# FreeMotion baseline 重新复现

按根目录 AGENTS.md：先理解和验证官方 baseline，再讨论 RAG / guidance；不参考旧版最终实现补 baseline。

- `FreeMotion/`：官方仓库，初始 commit `df92bdd982e25363fd27c86899e22b3b40230de2`。
- `scripts/`：本次新增的下载、初始化和验证入口。
- `docs/`：面向研究理解的机制、证据和复现边界。
- `artifacts/random_init/`：未训练的单人 / 多人 checkpoint、采样数组和实测 JSON。
- `downloads/`：官方下载资产、清单与完整性记录。

当前随机 checkpoint 不是论文权重。CLIP 使用官方预训练权重；运动网络保留上游默认随机初始化及特殊零初始化。

已完成官方下载、全量数据审计、单人/双人随机模型 DDIM50 采样、真实样本反向更新和原始 Lightning 入口 3 步训练验证。正式两阶段收敛训练及论文指标复现尚未完成。

## 在本机运行

PowerShell，从本目录执行：

```powershell
& .\.venv\Scripts\python.exe scripts/bootstrap_smoke.py --stage single
& .\.venv\Scripts\python.exe scripts/bootstrap_smoke.py --stage multi
```

两个命令会重建对应的随机 checkpoint；single 必须先于 multi。采样使用原尺寸 8 层、1024 隐空间、1000 步训练扩散 / DDIM50 采样，batch=1、16 帧仅用于流程验证。脚本在合成数据更新**之前**保存 checkpoint，避免将调试更新误认为训练结果。

环境是 Python 3.12 + PyTorch 2.11/cu128 的 Windows 本机兼容环境；`.venv` 复用了系统包，不是原论文 Python 3.8 / torch 1.13.1 环境。实际依赖记录见 `requirements-local.lock.txt`，其中可能包含宿主已有的无关包，不应直接当作最小可移植依赖表。

直接依赖及已测版本另列于 `requirements-local.txt`。在新机器可使用 Python 3.12 创建独立 venv 后安装；尚未在第二台机器验证。

```powershell
# 下载完成后的数据审计（--extract 会执行 ZIP CRC 检查和解压）
& .\.venv\Scripts\python.exe scripts/audit_data.py --extract
# 根据最终审计生成显式的缺失资产排除名单；不修改官方划分
& .\.venv\Scripts\python.exe scripts/prepare_local_data.py
# 原始损失、真实样本的少步检查
& .\.venv\Scripts\python.exe scripts/real_data_smoke.py --steps 3
# 独立采样；重复 --text 可依次增加人物。随机模型输出不代表动作质量。
& .\.venv\Scripts\python.exe scripts/sample_random.py --text 'A person walks forward.'
# 原训练入口，Windows 单 GPU，最多 3 个更新用于检查
Set-Location FreeMotion
& ..\.venv\Scripts\python.exe tools/train.py --model_config configs/model_single.yaml --dataset_config ../configs/datasets_single_local.yaml --train_config ../configs/train_single_local.yaml --max_steps 3 --seed 20261003
```

本机训练配置使用 batch=1、workers=0、按需读取数据；不与上游 batch=80 的正式训练结果等同。去掉 `--max_steps 3` 会进入上游 2500 epoch 计划，实际耗时和收敛尚未验证。

## 数据状态

官方 train/val/test 文件保持原样。11 个 ID 缺必要动作或个人描述，列在本次生成的 `FreeMotion/data/split/ignore_list.txt`；它不是作者提供的排除表。再经过原代码的短动作过滤，可用训练/验证/测试序列为 5629/546/1099。训练集镜像增强后为 11258 条。详见 `artifacts/data_audit.json` 和 `artifacts/local_data_policy.json`。

## 重新下载官方资产

以下从本目录执行，标注下载会保留并验证同名文件变体：

```powershell
& .\.venv\Scripts\python.exe scripts/list_public_annotations.py
& .\.venv\Scripts\python.exe scripts/download_annotations.py
& .\.venv\Scripts\python.exe scripts/download_motion_archive.py
& .\.venv\Scripts\python.exe -m gdown 1ZtNZuctaIXMrxlWuQZS2eZhT0U6kAMoQ -O downloads/separate_annots.zip
& .\.venv\Scripts\python.exe -m gdown 1bJv5lTP7otJleaBYZ2byjru_k_wCsGvH -O downloads/interclip.ckpt
```

本机这两份小包/权重已分别解压到 `FreeMotion/data/separate_annots/` 和复制到 `FreeMotion/eval_model/interclip.ckpt`；新下载后也需放到这两个位置。重新下载不是必要的日常运行步骤。

## 研究阅读入口

先读 `docs/01_baseline_mechanism.md`，再读 `docs/02_reproduction_evidence.md`。未完成的真实训练与指标复现不会写成已完成。

8GB 显存的小规模试验已完成单人 200 步和交互 50 步：无 OOM / NaN，最大 PyTorch 预留显存约 4.62 GiB，固定验证去噪误差有所下降，但动作偏小、姿态仍不自然，尚未收敛。设置、对比动画说明及实测边界见 [小规模实验记录](docs/03_small_scale_trial.md)，权重和动画位于 `artifacts/small_scale_8gb/`。

后续仅将单人预算延长至 2000 步的对照已完成：固定验证去噪误差较 200 步下降约 38.8%，运动量增加，但姿态仍有缺陷。没有修改模型或重训交互阶段。见 [训练预算对照](docs/04_training_budget.md)；新权重及 200/2000 步对比动画位于 `artifacts/single_2000_8gb/`。

2000 步单人模型已补测官方 InterCLIP 下的 96 条测试样本：FID 161.678、R-Precision top-1 4.17%、MM Distance 4.179。与论文表 2 的对照、原始报告及有限样本限制见 [论文指标试评估](docs/05_paper_metrics_pilot.md)。这是单次小样本试评估，不是正式全量指标复现。

新增文本适配层解冻对照已完成：同初始化和 2000 步预算，固定 val96 的 FID 148.188 → 139.502，但 top-1 不变、MM Distance 略变差，尚不能认定解决了生成质量问题。原 CLIP 仍冻结，原 baseline 保留。见 [解冻实验](docs/08_unfreeze_text_adapter.md)。

2026-10-04 已启动原版冻结 baseline 的12小时全量单人训练：5629个train ID、镜像后11258条/轮、300帧上限、microbatch=1、累积16条。断点恢复试跑通过；是否仍在运行以 `artifacts/overnight_full_single/status.json` 及进程为准，不把启动当完成。配置、断点、停止与恢复方式见 [长时训练说明](docs/09_overnight_full_training.md)。

## 线上 4090 全量单人训练

8GB 低显存适配（microbatch=1、累积16）在 24GB 上是浪费且慢。新增 `scripts/train_full_single_4090.py`：真实 minibatch（默认 32）、worker 数据加载重叠、FP32 默认（可选 `--bf16`）、论文 2500 epoch warmup-10 cosine 计划、按 epoch 原子断点与 STOP 安全停止。模型/损失/数据不改。先 `--max-updates` 预检再正式跑；详见 [4090 全量训练说明](docs/11_full_training_4090.md)。
