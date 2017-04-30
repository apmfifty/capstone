# Rossmann药店销售额预测 

### 题目描述

Rossmann是欧洲的一家连锁药店。 在这个源自Kaggle比赛[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)中，我们需要根据Rossmann药妆店的信息（比如促销，竞争对手，节假日）以及在过去的销售情况，来预测Rossmann未来的销售额。

### 数据下载 
此数据集可以从Kaggle上[下载](https://www.kaggle.com/c/rossmann-store-sales/data)。


### 提交
* PDF 报告文件: capstone_report_Rossmann_Sales_Prediction_Pengju_Zhao.pdf

* 项目相关代码: Capstone_Project_Rossman_Sales_Prediction_1.ipynb

* 包含使用的库，机器硬件，机器操作系统，训练时间等数据的 README 文档

  * 函数库：pandas，numpy，matplotlib，seaborn，IPython.display，sklearn，xgboost，time，datetime，isoweek.Week，os，itertools，operator
  * 机器硬件：macbook pro，2.5 GHz Intel Core i7，16 GB 1600 MHz DDR3
  * 机器操作系统：macOS Sierra 10.12.3

* 训练时间：训练及优化XGBoost模型时间是最主要的时间开销，合计67518seonds（约18.8小时）

  * 初始模型训练：2744 seconds
  * 学习率优化：2676 seconds
  * 优化 max_depth 和 min_child_weight：

    * 第一次 8506 seconds
    * 第二次 3330 seconds
  * 优化 subsample 和 colsample_bytree
    * 第一次 13048 seconds
    * 第二次 10615 seconds
  * 优化gamma: 5903 seconds
  * 优化reg_alpha: 5188 seconds
  * 优化学习率和提前终止条件：15508 seconds


