# 研究输入

`predictions_test.csv.gz`为本项目运行官方MASTER冻结权重得到的预测输出，包含日期、股票、预测分数、前瞻标签及当日百分位。

`prices.csv.gz`为公开Qlib价格发布包中相关股票、日期和字段的提取快照。价格来自复权收盘序列；原始字段保留其发布时精度。`price_manifest.json`记录快照哈希和历史成分核对结果。

来源：

- [SJTU-DMTai/MASTER](https://github.com/SJTU-DMTai/MASTER)，代码采用MIT许可；官方模型与预处理数据来源见该项目说明。
- [chenditc/investment_data，2024-12-07](https://github.com/chenditc/investment_data/releases/tag/2024-12-07)，原始Qlib价格发布包。

请保留上述来源。第三方模型和数据的权利归原作者或提供者，本项目代码的许可不替代其使用条件。

引用：Li, T., Liu, Z., Shen, Y., Wang, X., Chen, H. & Huang, S. (2024). MASTER: Market-Guided Stock Transformer for Stock Price Forecasting. AAAI, 38(1), 162–170.
