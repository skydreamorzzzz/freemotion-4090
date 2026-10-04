# 文本适配层核查：源码事实与实测边界

本次是对原版 baseline 的核查，没有修改模型、重训、加入 RAG 或几何 guidance。CPU 核查入口为 `scripts/audit_text_adapter.py`，实测产物为 `artifacts/text_adapter_audit.json`。

## 没有找到默认配置中漏执行的加载步骤

2026-10-03 直接读取 [官方 models/intergen.py](https://github.com/VankouF/FreeMotion-Codes/blob/main/models/intergen.py)，换行规范化后与本地文件一致；该文件也没有本地 git diff。

构造函数先用 clip.load 加载官方 CLIP，再新建两层 `nn.TransformerEncoder` 和 `clip_ln`。前者使用 PyTorch 默认随机初始化；后者 LayerNorm 的 weight=1、bias=0，**并非随机参数**。此前笼统称两者“随机初始化”不够准确，应按此区分。

CLIP 本身及新增适配层均被设为 requires_grad=False。官方 `tools/train.py` 优化器只收集 requires_grad=True 的参数。加载路径有 FROM_PRETRAIN 和 Lightning RESUME，但官方单人训练配置两项都为空，model_single.CHECKPOINT 也为空；构造过程没有额外适配层权重加载。因此按公开默认配置训练，这两层 Transformer 就是随机且冻结的。交互阶段载入单人 checkpoint 只会继承这个状态。

结论是“公开源码默认路径没有提供适配层预训练加载”，而不是“已证明作者遗漏了步骤”。不能排除论文实验采用未公开配置或其他初始化；作者动机仍待核查。

## 本机已训练权重核验

从随机初始 checkpoint 与单人 2000 步 checkpoint 提取适配层所有 26 个 state tensor，逐项 `torch.equal` 均为 True。该模块合计 11,029,504 个参数，可训练参数数目为 0。实测 clip_ln 初始权重全 1、偏置全 0。这排除了本机训练期间曾悄悄更新适配层的可能。

## 冻结参数，但训练时 dropout 仍生效

对走路、坐下、跳跃三条固定文本，各连续编码 6 次；固定 CPU 随机种子后比较同一文本重复输出：

| 模式 | 重复输出最大绝对差 | 重复输出平均余弦相似度 |
|---|---:|---:|
| 原版 model.train() | 1.4850 | 0.96549 |
| model.eval() | 0 | 约 1 |
| train 模式下仅将适配层设 eval（诊断） | 0 | 约 1 |

因此，“冻结文本网络”不代表训练时获得固定条件向量。原版 train() 会打开适配层 dropout；推理 eval() 会关闭。本次仅在内存中的诊断模型切换模式，未改训练脚本或任何 checkpoint。

这种随机扰动可能是正则化，也可能增加从头训练难度；随机冻结适配器也仍可能保留有用语义信息。上述证据都**不能单独证明**它是当前低检索率或高 FID 的主因。需要控制实验才能判断影响。

## 外部问题记录与查询限制

GitHub issue 列表的 API 查询得到 [issue #5](https://github.com/VankouF/FreeMotion-Codes/issues/5)：有使用者报告依公开代码训练后 FID 高于论文。该 issue 的一条回复未成功读取（API rate limit），不能声称作者已确认问题或提供修复。未向作者发送消息。

## 下一步建议，不等同于论文实现

保留原版对照。在相同随机初始化、train 子集和预算下，先单独测试“仍冻结，但关闭适配层 dropout”，以隔离条件随机性。再单独测试允许新增适配层学习、保持预训练 CLIP 冻结；这属于复现修正候选，不能宣称是作者原方法。用固定 val 指标及原始动作案例判断，避免用 test 调参；暂不与几何修正同时开启。
