# 远程 py38 审计：先取证，不安装

当前用户已人工验证 RTX4090、torch1.13.1+cu117、CUDA runtime11.7、cuDNN8500 和GPU矩阵乘正常。本次尚未直接访问远程机器，不能将这些转述写成助手实测。torch/torchvision/CUDA 保持现状，驱动显示 CUDA13.0 不构成升级理由。

旧 `docs/11`、`requirements-local.txt` 和自定义4090训练入口的安装/性能建议不作为环境依据。先看官方当前源码及远程实际版本、diff；不要执行旧文档中的环境升级命令。

在服务器、已激活的 py38 环境中，从本仓库根目录执行：

```bash
python scripts/audit_remote_environment.py --repo ./FreeMotion --output ./artifacts/remote_environment_audit
```

脚本不安装依赖、不导入本地模型代码、不训练、不重置git。它用临时目录获取官方仓库，记录具体commit和官方README、requirements、train/test脚本、config与依赖调用；对比本地源码，并执行当前解释器的python版本、pip list、pip check、conda list和torch版本探针。仅写审计产物，临时官方checkout退出时删除。如果官方获取失败，会明确标记未完成，不把本地fork当官方。

查看 `artifacts/remote_environment_audit/summary.md`，把它和 `report.json` 提供给助手后，才能据远程实测生成安装决策。报告可能包含目录和本地代码diff，分享前检查敏感内容；不要将整份环境或密钥直接提交到公共仓库。

## 官方依赖初步核对

已从官方原始commit核对：torch==1.13.1+cu117、torchvision==0.14.1+cu117、torchaudio==0.13.1、lightning==1.9.1、mmcv~=1.6.2、tensorboard==2.14.0。其他精确范围以脚本捕获的官方requirements为准，不能用torch2.11时代的本机lock覆盖。

官方代码导入的是 `lightning.pytorch`，安装有 `pytorch-lightning` 不一定代表该命名空间可用。依赖树还需查torchmetrics、lightning-utilities等实际版本；本阶段不猜测或自动调整。

当前官方源码可见的MMCV调用是 `mmcv.runner.get_dist_info`、`mmcv.utils.Registry/build_from_cfg`，未发现 `mmcv.ops` 调用。[MMCV1.6.2文档](https://mmcv.readthedocs.io/en/v1.6.2/get_started/installation.html)区分无CUDA算子的mmcv与mmcv-full；不能因为机器有CUDA就默认编译mmcv-full，也不能两者并装。先查已装版本和实际import，再决定是否缺少轻量包。

## 审计后的边界

版本不同仅是差异，尚不是必须升级/降级的证据。每次拟安装前列出必要性、现有版本、目标版本及对受保护包的影响；缺包安装也要核对传递依赖，不能直接全量pip install -r。涉及源码编译先说明原因和风险。

远程源码diff需先保存再判定。仅为Python3.12/torch2.11添加的兼容补丁不应自动沿用到py38，但也不能未审查就reset用户修改。最终smoke test使用官方模型、loss、optimizer与数据处理，先核心包import、config、dataset、model，再单batch前反向、optimizer step和checkpoint保存，不启动长训练。
