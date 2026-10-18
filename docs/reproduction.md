# 完整复现流程

## 1. 复算策略实验

README命令使用仓库内的冻结预测与价格快照，重新执行四组策略、三档费用并输出逐日账本。该步骤覆盖本项目的信号与组合改进，无需下载神经网络依赖。

## 2. 重新生成Transformer预测

安装PyTorch2.6.0；CPU安装方式可参见[PyTorch官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。然后获取官方仓库与预处理数据：

```bash
git clone https://github.com/SJTU-DMTai/MASTER.git external/MASTER
git -C external/MASTER checkout de8f58557096abde4216a701b35fc4368158d111
```

从[MASTER数据说明](https://github.com/SJTU-DMTai/MASTER#usage)中的公开链接下载`opensource.zip`，解压并找到`csi300_dl_test.pkl`。数据量较大，因此没有把输入特征和权重重复提交到本仓库。

```bash
python -m signalcraft.inference --source external/MASTER --sampler external/data/csi300_dl_test.pkl --output output/predictions_test.csv.gz
```

脚本先核对源码、模型权重和采样文件的SHA256，之后才加载代码和反序列化数据。它使用受限的Qlib采样状态加载器，模型权重使用`weights_only=True`。不要将其他来源的pickle文件交给该入口。

生成的预测可替换自定义数据目录中的`predictions_test.csv.gz`，与同一份`prices.csv.gz`一起传给`--data`。

## 3. 从原始价格重新导出面板

从[2024-12-07数据发布页](https://github.com/chenditc/investment_data/releases/tag/2024-12-07)下载`qlib_bin.tar.gz`并解压。`--qlib-dir`应指向同时包含`features`、`calendars`和`instruments`的目录。

```bash
python scripts/prepare_prices.py --qlib-dir external/qlib_bin --predictions data/predictions_test.csv.gz --output output/rebuilt-data
```

脚本按信号区间与股票集合提取close、volume、change三个原始字段，核对历史成分，写出压缩面板与校验记录。将预测文件放入同一目录后，可运行：

```bash
python -m signalcraft.cli --data output/rebuilt-data --output output/rebuilt-study
```

## 文件校验

| 文件 | SHA256 |
|---|---|
| csi300_opensource_0.pkl | `581443312928ac27fab1c94482cea9e3e0a250f50948fe17e88d8ab5fa188a59` |
| csi300_dl_test.pkl | `d05194f2b8d0a1d94e79a13abc2702e1c16c9d8e514ead3ea1b41d4de07cfd95` |
| prices.csv.gz | `23bf121516fef7d6b116e65bb08c95b7a9cb049220d675b65c42d5db80b2ae6e` |

压缩文件字节可能随工具版本变化；复核数值时比较解压后的表格内容。推断和回测的本地复核记录见`results/verification.json`。
