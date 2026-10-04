# 4090 24GB：Stage-1 单人全量训练

> 历史方案，尚未据当前远程机器验证。用户已明确要求保留可用的 py38 + torch1.13.1+cu117；本文的旧安装建议不再适用。请先执行 [远程只读审计](12_remote_environment_audit.md)，不要根据本文估算修改环境或直接开训。

面向线上单卡 4090（24GB）的新入口 `scripts/train_full_single_4090.py`。它保留上游模型、损失、数据类与单人流程不变，只替换训练循环与显存/速度相关的配置。原来的 `scripts/train_overnight.py`（5060 8GB、microbatch=1、累积16、12 小时预算）保持原样，作为低显存记录，不再用于正式训练。

本轮只做 Stage-1 单人（用户确认）。交互阶段未改，也不在本文范围内。

## 为什么改：8GB 版在 4090 上既浪费又慢

实测与推导（5060 8GB，batch=1、300 帧上限）：

- 单步峰值 PyTorch reserved 约 3.6 GiB，其中 trainable 参数 154.6M 的 FP32 参数+梯度+AdamW 两个动量约 2.5 GiB、冻结 CLIP/文本适配层约 0.3 GiB、AdamW step 临时量约 0.6 GiB，**这部分不随 batch 增长**。
- 激活只占约 0.2–0.3 GiB/样本，近似线性。
- 数据 `__getitem__`（CACHE=True）实测 **1.78 ms/样本**，在单进程按需读取的循环里会和 GPU 计算串行。

也就是说 8GB 的限制几乎全部来自**固定开销**，而不是 batch。4090 上单步按 `3.6 + 0.32 × batch` GiB 粗估：batch 32 ≈ 13.8、40 ≈ 16.4、48 ≈ 19.0、64 ≈ 24.1 GiB。因此 4090 可以跑真实 minibatch，而不是 microbatch=1 累积 16 次。

## 与论文/上游的对应关系

保持与上游一致：模型与超参（LR 1e-4、weight_decay 2e-5、梯度裁剪 0.5、warmup 10 epoch、cosine 到 2500 epoch）、300 帧上限、冻结文本适配层、AdamW 且优化器只含 `requires_grad` 参数、每个 epoch 调一次 `CosineWarmupScheduler.step()`。

有意不同：

- 单卡无 DDP；上游 `train_single.sh` 用全部可见 GPU。
- 有效 batch 默认 = `--batch-size`（32）。论文单卡配置是 80，单步放不下；要贴合可 `--batch-size 40 --accumulation 2`（有效 80）。
- 上游 `tools/train.py` 开 `detect_anomaly=True`（逐算子检查，显著拖慢）；本入口不开。
- 每个 epoch 内每个样本的文本/人物/裁剪增强用 DataLoader worker 的 RNG，不保证逐位复现；epoch 顺序由保存的 generator 状态保证可复现。优化器/scheduler/模型/RNG 状态按 epoch 边界保存与恢复。

## 速度来源

1. 真实 minibatch（默认 32）：一次前/反向覆盖 32 条，去掉累积 16 次带来的 16 倍 Python/launch 开销。
2. DataLoader + `num_workers`（默认 4）+ `pin_memory` + CACHE：把 1.78 ms/样本的取数与 GPU 计算重叠，不再阻塞。
3. 默认开启 TF32 矩阵乘（`set_float32_matmul_precision("high")`，与上游 `medium` 同类）；`--no-tf32` 回到严格 FP32。
4. 可选 `--bf16`：只把解码器前/反向包进 `autocast(bfloat16)`，冻结的 fp16 CLIP 文本编码留在 autocast 之外，避免 dtype 混用。约 2x，但数值与原 FP32 不完全一致，作为对照更合适。
5. `cudnn.benchmark=True`（训练形状固定）。

## 用法

先做预检（会在目标机上打印峰值显存与每 epoch 秒数/ETA，并落一个 checkpoint）：

```bash
python scripts/train_full_single_4090.py \
  --output-dir artifacts/full_single_4090_preflight --batch-size 32 --max-updates 20
```

看峰值显存与 `seconds_per_epoch`：若 24GB 有余量可把 `--batch-size` 提到 40/48；若接近上限就下调。确认后正式跑（新目录，或 `--resume` 预检目录，会重放当前 epoch）：

```bash
python scripts/train_full_single_4090.py --output-dir artifacts/full_single_4090 --batch-size 32
# 更快（数值略有差异）
python scripts/train_full_single_4090.py --output-dir artifacts/full_single_4090 --batch-size 32 --bf16
# 贴合论文单卡 batch 80
python scripts/train_full_single_4090.py --output-dir artifacts/full_single_4090 --batch-size 40 --accumulation 2
```

续训与停止：

```bash
python scripts/train_full_single_4090.py --output-dir artifacts/full_single_4090 --resume
touch artifacts/full_single_4090/STOP      # 完成当前 epoch 后保存退出
```

先在 5060 上做小步流程验证（batch 2、无 worker、8GB 可承受）：

```bash
python scripts/train_full_single_4090.py \
  --output-dir artifacts/full_single_4090_smoke --batch-size 2 --num-workers 0 --max-updates 5
```

## 产出与监控

输出目录（默认 `artifacts/full_single_4090/`）：`protocol.json`（超参与数据签名，续训时严格比对）、`train_ids.json`、`progress.jsonl`（每 `--log-every` 个 update，默认 10）、`validation.jsonl`（固定 4 条 val 的去噪 MSE 诊断）、`status.json`（进程、update/epoch、峰值显存、ETA，供外部轮询）、`latest.ckpt`（原子替换，含模型/优化器/scheduler/RNG/generator/游标）、`best_validation.ckpt`（val MSE 最优时的快照）。`STOP` 文件用于安全停止。

## 规模与时间（需实测）

每 epoch 11258 条，`updates/epoch = 11258 // batch`（batch 32 → 351）。2500 epoch 合计约 87.7 万次优化器 update、2814 万条样本。总时长 ≈ 28.1M / 实测样本吞吐：60 samples/s → 约 5.5 天，100 samples/s → 约 3.3 天。预检的 `seconds_per_epoch` 给出可靠外推，不要用 5060 的数推算 4090。2500 epoch 是论文计划，可先按 epoch 存档、提前评估中间 checkpoint。

## 明确不做 / 不能声称

- 只训练 Stage-1 单人，未触及交互阶段。
- 本入口不计算 FID / R-Precision；`validation.jsonl` 的 120 帧固定去噪 MSE 只是趋势诊断，不是论文指标。
- 不是完成的论文复现；中间结果不当作最终结论。
- 显存估算是粗估，务必先跑 `--max-updates` 预检；OOM 就下调 `--batch-size`。
- 断点只在 epoch 边界完整；最后未 checkpoint 的多余部分在恢复时会被重放（同一 epoch 从头）。`--accumulation` 不能整除 `updates/epoch` 时，每 epoch 末尾会丢弃不足一个窗口的少量样本（约 0.3%）。

## 线上取代码

代码分两个公开仓库，克隆到同一父目录，保持本机的相对布局：

```bash
git clone https://github.com/skydreamorzzzz/freemotion-4090.git
git clone https://github.com/skydreamorzzzz/FreeMotion-Codes.git freemotion-4090/FreeMotion
```

即 `freemotion-4090/`（本仓库：scripts/docs/configs）+ 其下的 `FreeMotion/`（含 Python 3.12 / torch 2.11 兼容改动）。`FreeMotion/data`、`eval_model`、`artifacts`、`downloads` 都不在仓库里（体积大），需按 README 的下载脚本在服务器另行获取。

## 环境

原先建议升级 Python3.12/torch2.11 的方案已撤回：当前远程 py38 + torch1.13.1+cu117 已由用户验证GPU矩阵乘可用，先保留，不能根据显卡型号或驱动CUDA版本升级。依赖及数据准备以当前官方baseline审计结果为准，本节不提供安装命令。
