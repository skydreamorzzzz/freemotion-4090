# FreeMotion：先理解官方 baseline

证据类型：本文以官方源码固定版本为依据；公式是对源码计算的数学表述，尚未完成与论文逐式核对。不是对作者当年研究历程的复原。

## 核心结构和两阶段的必要性

原始源码 `models/intergen.py` 的 `InterGenSpatialControlNet` 包含文本编码器与扩散解码器。CLIP ViT-L/14@336px 的文本输出经过额外两层 Transformer 和 LayerNorm，再用 EOT 对应的 token 表示作为 768 维条件。若同时提供个人描述和整体描述，源码将两个表示相加。

需要特别核查：额外两层 Transformer 与 `clip_ln` 也被 `set_requires_grad(..., False)` 冻结。Transformer 来自默认随机初始化，LayerNorm 则是默认 weight=1、bias=0；CLIP 自身的预训练权重不包含这些新增层。stage 1 默认 `FROM_PRETRAIN` 为空。已实测这些参数在 2000 步后完全不变，训练模式下适配层 dropout 仍开启，见 `07_text_adapter_audit.md`。是否遗漏了作者的初始化/加载步骤尚待核查，不能编造成有意设计或已确认的错误。

`models/nets.py` 将每个人每帧的 262 维动作嵌入到 1024 维，用 8 个 Transformer block 处理；时间步嵌入与文本嵌入相加用于调制。单人阶段训练生成主干。交互阶段创建控制分支，沿用单人参数，冻结主干，再经逐层零初始化线性层把控制分支的特征加入主干。

训练交互模型时，第一个人是预测目标，其余人的干净动作作为 motion guidance。采样则先生成第一个人，再把已生成动作作为条件逐人生成人物；这与训练时使用真实条件动作有差别，误差传播是后续值得验证的问题，目前不能说已经通过实验证实。

原始模型的 motion guidance 与 classifier-free guidance 已属于 baseline；它们不能冒充我们后续新增的 RAG / guidance 模块。

## 扩散模型究竟预测什么

配置为 cosine beta schedule，1000 个扩散时间步；均匀采样训练时间步。前向加噪可表示为

`x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon`。

源码显式配置 `ModelMeanType.START_X`，所以网络预测的是干净动作 `x_0`，不能只根据通用扩散介绍说它预测 epsilon。`models/gaussian_diffusion.py` 中的 `training_losses` 会先按上游 mean/std 归一化动作，再拆出预测目标和其他人的条件动作。

采样使用 DDIM50。`models/cfg_sampler.py` 将带文本/空文本样本拼在 batch 中计算，最终输出为 `s*f_cond + (1-s)*f_uncond`，默认 `s=3.5`。motion guidance 在这两路都保留，并非同时清空文本和其他人物动作。

需核查的另一处源码行为：`InterDenoiser.forward` 调用 `mask_motion_guidance(..., 0.1)` 没有检查 training 状态，多人采样时也可能丢弃条件人物。当前保留原实现，未静默修正。

## 数据与损失

InterHuman 的官方预处理文件并不等同于模型最终输入。`utils/preprocess.py:load_motion` 先选 22 个关节位置和 21 个关节的 6D 旋转；`process_motion_np` 进一步生成位置、速度、旋转、脚接触等共 262 维表示。两人的坐标处理保留相对旋转与位移。需要通过数据检查验证配对、文本行对应、截断和镜像增强，不能只看文件数量。

`datasets/interhuman.py` 按官方 train/val/test 名单过滤，训练集加入镜像样本，随机交换人物；最长 300 帧、最短阈值 15 帧。每条整体标注与 text1/text2 使用同一行号，三者行数和对应关系是数据审计重点。

单人损失包含归一化动作重建及几何项；多人额外包含距离图、关节亲近关系和相对朝向项。几何项计算前反归一化。`T_BAR=700` 控制部分几何/交互项生效的时间段：一次随机采到高时间步的 loss 为零，并不代表该项不存在或失效。原配置学习率 `1e-4`、AdamW weight decay `2e-5`、梯度裁剪 `0.5`；这些是沿用上游的参数，不表示本次已经验证其最优性。

## 评价指标不是对原始坐标直接算

`datasets/evaluator.py` 依赖预训练 InterCLIP (`eval_model/interclip.ckpt`) 提取文本/动作嵌入。缺少评价权重时不可用随机 evaluator 冒充论文指标。

- Matching distance：配对文本与动作嵌入的欧氏距离均值。
- R-precision：在评价 batch 中将动作按与文本距离排序，统计匹配样本进入 top-1/2/3 的比例；原入口固定 batch=96，改变 batch 会改变候选集。
- FID：对动作嵌入拟合均值、协方差，计算 `||mu1-mu2||² + tr(C1+C2-2*sqrt(C1*C2))`。
- Diversity：不同样本的动作嵌入距离抽样均值。
- Multimodality：同一文本多次采样的动作嵌入距离均值。

注意 `utils/metrics.py` 中 FID 统计先将嵌入乘 6；Diversity 也乘 6，距离还除 2；Multimodality 函数没有这两步。不能用别的库的同名默认指标直接替换。评价入口设置重复 20 次、Diversity 抽样 300、100 条文本各生成 30 次用于多模态评估（每条抽 10 对）；这些协议和数据划分都需要一起固定。

## 后续研究问题的边界

当前尚未完成真实 baseline 训练和指标评估，不能声称“baseline 在某类动作失败，因此需要 RAG”。后续先检查真实生成质量、文本匹配、交互约束及逐人误差传播，形成带样例/指标的证据，再提出和比较候选改进。
