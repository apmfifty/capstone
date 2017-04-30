# 机器学习纳米学位
##毕业项目
 赵鹏举

2017年4月30日

## Rossmann药店销售额预测

## I. 问题的定义

### 项目概述

本项目来自Kaggle比赛[Rossman Store Sales【1】](https://www.kaggle.com/c/rossmann-store-sales#description).  截至2015年，Rossmann在欧洲7个国家运行着超过3000家连锁药店，这些药店的营收会受到促销、竞争者、国家/学校假期、季节、地域等因素的影响。Rossmann希望参加比赛项目的选手，可以准确地预测出位于德国1115家药店在六周内每天的销售情况；进而利用可靠的销售预测情况帮助药店经理制定更加高效的工作安排。

本项目中，相关数据包含 train.csv和store.csv：

- train.csv是历史销售数据，每条信息包含了药店编号、日期、星期几、是否营业、是否节假日、是否促销、当日销售额以及客户数量；
- store.csv是药店数据，每条信息包含了药店编号、药店类型、商品组合、最近竞争者距离及开店时间、促销有无、促销间隔和开始时间。

输入的数据对于药店销售预测是非常有用的：日期和星期几等可以提供销售额周期性的时间标定；是否节假日和促销，以及每家药店的信息和竞争者的信息，对于销售额也会有一定影响。

销售预测对于每一个企业来说都非常重要，在这一领域，机器学习方法已经得到了广泛且重要的应用，因此通过完成本项目过程中，可以锻炼掌握机器学习的常用方法和工作流程，为从事数据分析工作做准备。销售额预测属于机器学习中有**监督学习**的回归问题，可以采用线性模型、决策树、SVM、神经网络、集成学习等方法进行建模和预测【2】；具体到本项目， train.csv的最后6周（2015-06-15之后）的数据将会被提前预留出来作为模型的验证集，而其他数据将会被作为模型的训练集。

### 问题陈述
本问题属于有监督机器学习中的回归问题：已知1115家药店的信息以及每家药店在2年多时间内每天的销售情况，需要对接下来6周内每家药店的销售状况进行预测。回归问题的常见机器学习方法有K近邻学习、线性回归、决策树、随机森林、XGBoost【3】、神经网络等；而实际中，为了训练出效果较好的模型，一般需要根据数据集的特点，进行特征工程，构造出有用的新特征，并对特征进行选择，同时注意防止过拟合。

本项目将采用XGBoost方法；XGBoost模型是一种有监督的集成学习方法，可以直观理解为对决策树的集成，是非常有效的解决非结构化数据的方法，在Kaggle比赛中得到广泛的应用；比赛过程中，第一名[Gert【4】](https://kaggle2.blob.core.windows.net/forum-message-attachments/102102/3454/Rossmann_nr1_doc.pdf)在原有数据集基础上，构造出临近信息、时间信息、趋势信息等特征，并采用XGBoost方法训练模型；

在本问题提供的数据集中，销售数据作为标记值，其他属性作为特征，对选择的模型进行训练；模型的预测销售数据与标记销售数据之间的差异可以用来对模型进行评估；训练好的模型，对测试数据的预测是可以再现的。

### 评价指标
本项目采用Kaggle比赛【1】的评估指标：RMSPE（误差百分比的均方差），可表示为
$$
RMSPE= \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\frac{y_i-\hat{y_i}}{y_i})^2}
$$
其中，任何当天销售额为0的数据在评估时将被忽略； $y_i$ 表示某药店在某天的实际销售额，而$\hat{y_i}$ 表示该药店在对应这一天的预测销售额。该评价指标的取值范围为[0,1]，数值越小表示结果越好。

该评价指标非常合理：

- 误差百分比相对于误差值更为合理，可以避免某些误差值数值较大导致所占比重过大
- 将实际销售额为0的数据剔除出去，避免过度预测（数据缺省导致无法判断该店当天是否营业，保险起见也对当天销售额进行预测）对结果的影响


## II. 分析
### 数据的探索
####  文件train.csv和store.csv探索
本项目中将会使用 train.csv和store.csv两个文件。

train.csv包含了1115家药店从2013-01-01到2015-07-01合计1017209条销售数据，其中每条销售数据的列名、类型、及其意义为：

| 列名            | 数据类型   | 意义                                       |
| :------------ | :----- | :--------------------------------------- |
| Store         | int64  | 药店编号，从1到1115                             |
| DayOfWeek     | int64  | 星期几，从1到7                                 |
| Date          | object | 日期的字符串形式，从2013-01-01到2015-07-01          |
| Sales         | int64  | 当日该店销售额，非负数；不营业时（即Open为0）为0              |
| Customers     | int64  | 当日该店客户数，非负数；不营业时（即Open为0）为0              |
| Open          | int64  | 当日该店是否营业，是为1，否为0；                        |
| Promo         | int64  | 当日该店是否参与促销Promo，是为1，否为0；                 |
| StateHoliday  | object | 用字符串表示类型，当日该店是否为StateHoliday，是可能为'a' 、'b'、 'c'，否为'0' |
| SchoolHoliday | int64  | 当日该店是否为SchoolHoliday，是为1，否为0；            |

store.csv包含了1115家药店各自的信息，其中每条药店信息的列名、类型、及其意义为：

| 列名                        | 数据类型    | 意义                                       |
| :------------------------ | :------ | :--------------------------------------- |
| Store                     | int64   | 药店编号，从1到1115；与train.csv中Store意义一样        |
| StoreType                 | object  | 表示药店某种类型划分，可选值为字符'a'、 'b'、'c'、 'd'四者之一   |
| Assortment                | object  | 表示药店某种类型划分，可选值为字符'a'、 'b'、'c'三者之一        |
| CompetitionDistance       | float64 | 有缺省值，表示距离最近竞争者到该药店的距离                    |
| CompetitionOpenSinceMonth | float64 | 有缺省值，表示竞争者开始营业的月份                        |
| CompetitionOpenSinceYear  | float64 | 有缺省值，表示竞争者开始营业的年份                        |
| Promo2                    | int64   | 该药店是否参与促销活动Promo2                        |
| Promo2SinceWeek           | float64 | 有缺省值，该药店参与促销活动Promo2的起始周                 |
| Promo2SinceYear           | float64 | 有缺省值，该药店参与促销活动Promo2的起始年                 |
| PromoInterval             | object  | 有缺省值，该药店参与促销活动Promo2的月份，可选值为'Jan,Apr,Jul,Oct' ， 'Feb,May,Aug,Nov' ，'Mar,Jun,Sept,Dec'三者之一 |

### 探索性可视化

销售额Sales是需要进行预测的特征，因其特别重要，所以首先对其进行可视化分析:

- 通过销售额的柱状图和箱框图可以发现：销售额分布范围大约从0到42000，主要大约集中在4000到8000；
- 通过销售的月份平均图可以发现：销售额在月份尺度上体现出一定的周期性，每年的12月份销售额最高，每年的1-3月份销售最低；

| ![](./Rossmann_Store_Sales/insertedPics/2_6_1_SalesDistribution.png) | ![](./Rossmann_Store_Sales/insertedPics/2_6_2_BoxplotOfSales.png) |
| :--------------------------------------: | :--------------------------------------: |
|               销售额Sales的柱状图               |               销售额Sales的箱框图               |

| ![](./Rossmann_Store_Sales/insertedPics/2_6_4_AverageSalesOverMonth.png) |
| :--------------------------------------: |
|              销售额Sales的月份平均图              |

DayOfWeek、Promo、StateHoliday、SchoolHoliday、StoreType、Assortment等特征与销售额Sales有非常重要的关系，因此将其之间的关系进行可视化：

- DayOfWeek为1和7（即周一和周日）时，销售额较高；
- Promo为1（即进行促销Promo时），销售额较高；
- 处于国家假期StateHoliday（即StateHoliday不为0时）销售额高于非国家假期（即StateHoliday为0时），处于假期b和c时，销售额的平均值（大约10000）比非假期销售额平均值（大约7000）高出30%；
- SchoolHoliday对于销售额的平均值影响不大；
- StoreType为b的药店平均销售额（大约9500）比其他StoreType的药店平均销售额（略低于6000）高出大约50%；
- Assortment为b的平均销售额（大约8000）比其他Assortment的药店平均销售额（大约6000）高出大约30%；

| ![](./Rossmann_Store_Sales/insertedPics/2_2_MeanofSalesOnDayOfWeek.png) | ![](./Rossmann_Store_Sales/insertedPics/2_3_MeanOfSalesOnPromo.png) |
| :--------------------------------------: | :--------------------------------------: |
|             DayOfWeek的销售额平均值             |               Promo的销售额平均值               |

| ![](./Rossmann_Store_Sales/insertedPics/2_4_MeanSalesOnStateHoliday.png) | ![](./Rossmann_Store_Sales/insertedPics/2_5_MeanSalesOnSchoolHoliday.png) |
| :--------------------------------------: | :--------------------------------------: |
|           StateHoliday的销售额平均值            |           SchoolHoliday的销售额平均值           |

| ![](./Rossmann_Store_Sales/insertedPics/2_7_SalesAverageOnStoreType.png) | ![](./Rossmann_Store_Sales/insertedPics/2_8_SalesAverageOnAssortment.png) |
| :--------------------------------------: | :--------------------------------------: |
|             StoreType的销售额平均值             |            Assortment的销售额平均值             |

Competition对于销售额也有非常大的影响：

- 以6号药店为例：2013年每月日均销售额范围为5800到7200；而从2013年12月出现竞争者后，2014年每月日均销售额范围为4500到6000，下降非常明显；
- 从CompetitionDistance的KDE分布图可以发现：CompetitionDistance绝大多数分布在0-8000范围内；
- 从CompetitionDistance-Sales的散点图可以发现：CompetitionDistance和Sales绝大多数分布与CompetitionDistance（0，10000），Sales（4000，10000）的范围内。

| ![](./Rossmann_Store_Sales/insertedPics/2_9_2CompetitionEffectOnSalesOnStore6.png) |
| :--------------------------------------: |
|           Competition对6号药店的影响            |

| ![](./Rossmann_Store_Sales/insertedPics/2_9_1_KDEOfCompetitionDistance.png) | ![](./Rossmann_Store_Sales/insertedPics/2_9_1ScatterOfCompetitionDistanceSales.png) |
| :--------------------------------------: | :--------------------------------------: |
|        CompetitionDistance的KDE分布图        |      CompetitionDistance-Sales的散点图       |

### 算法和技术

本项目训练模型将采用XGBoost方法【3】，该方法是数据挖掘和机器学习中最常用的算法之一，因为效果好，对于输入要求不敏感，是从统计学家到数据科学家必备的工具之一，同时也是kaggle比赛冠军选手最常用的工具。最后，因为效果好，计算复杂度不高，也在工业界有大量的应用。

- XGBoost方法是集成学习的一种；集成学习通过构建并结合多个学习器来完成学习任务，而按照个体学习生成器的生成方式，主要分为序列化方法和并行化方法。XGBoost方法是序列化方法代表Boosting的重要算法。 
- Boosting【2】的工作机制是指：从初始训练集中训练一个基学习器，再根据基学习器的表现对训练样本分布进行调整，使得先前基学习器做错的训练样本在后续受到更多关注，然后基于调整后的样本分布来训练下一个基学习器。如此重复直至基学习器训练结束。
- XGBoost方法的基学习器是回归树（CART）【3】。

XGBoost方法对数据的学习能力非常强，在学习过程中比较容易遇到过拟合的问题；为解决[XGBoost的过拟合问题【6】](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)，需要对learning rate学习率、max_depth树最大深度、min_child_weight、subsample、colsample_bytree等参数进行调优；此外，可选不同的特征、random seed训练多个XGBoost模型，然后进行集成，以提高模型性能；

### 基准模型

本项目的基准模型采用具有相同特征参数（Store，DayOfWeek，Promo和InPromo2Today等四个特征）数据子集的中位数；该模型在训练集上的RMSPE数值为**0.225588**，在验证集上的RMSPE数值为**0.140907**；

采用该模型作为基准模型，是因为该方法简单有效：

- 模型训练过程非常简单，其核心思路就是创建一个字典dict：从任一特征组合的参数映射到销售额的中位数；而预测过程也非常简单，就是对照训练好的字典dict，找出训练样本所对应参数组合的销售额。
- 该模型之所以有效，是因为对构造模型的特征组合进行了挑选，并用中位数来泛化每一种特征组合（虽然忽略了一些特征）的对应销售额；

## III. 方法
### 数据预处理

#### 类别数据信息处理

- 将train.csv中用字符串格式表示的特征StateHoliday的'0','a','b','c'转变为数值0,1,2,3表示的特征StateHoliday_cat；
- 将store.csv中用字符串格式表示的特征StoreType的'a','b','c','d'转变为数值0,1,2,3表示的StoreType_cat；
- 将store.csv中用字符串格式表示的特征Assortment的'a','b','c'转变为数值0,1,2表示的Assortment_cat；
- 将store.csv中用字符串格式表示的特征PromoInterval的'Jan,Apr,Jul,Oct' ， 'Feb,May,Aug,Nov' ，'Mar,Jun,Sept,Dec'转变为相应的月份1/4/7/10，2/5/8/11和3/6/9/12。

#### 时间特征处理

train.csv数据的Date为日期的字符串形式，难以直接进行处理，需要先转换为python的标准时间格式datetime；而datetime格式作为一个特征依然比较难以直接处理，所以从中提取为年Year，月Month，日Day三个独立特征；此外，从1月1日算起，当日位于当年第几天DayOfYear作为独立特征提取出来；

#### 竞争Competition相关特征处理

store.csv数据中关于竞争Competition有三个特征CompetitionDistance，CompetitionOpenSinceMonth，CompetitionOpenSinceYear；后两者在是否为缺省值时具有一致性。

- 缺省值
  - 当CompetitionDistance为缺省值时，CompetitionOpenSinceMonth，CompetitionOpenSinceYear亦同时为缺省值，在这种情况下，假定该药店开始竞争的时间为train.csv统计时间结束时（取为2016-01-01），而竞争距离为CompetitionDistance的中位数；
  - 当CompetitionDistance不为缺省值，而CompetitionOpenSinceMonth，CompetitionOpenSinceYear为缺省值时，假定Competition开始时间太早而无法统计到，将其统一设置为CompetitionOpenSinceMonth/Year的最早值1961-01-01。
- 后两者合在一起表示了该店竞争Competition开始的时间，首先取该月份的1日作为该店竞争Competition开始的日期CompetitionSinceDate，进而将该店遇到竞争Competition的天数累积值DaysCountSinceCompetition作为独立特征。
- 对于train.csv任一行数据，如果其日期Date在该店的CompetitionSinceDate后，则构建用来表示当天是否处于竞争的特征InCompetitionToday为1，否则为0；

#### 促销Promo2相关特征处理

store.csv中关于促销Promo2有Promo2，Promo2SinceWeek，Promo2SinceYear，PromoInterval等四个特征；当Promo2为0时，后三者为缺省值；

- 缺省值处理
  - Promo2SinceWeek和Promo2SinceYear的缺省值取train.csv统计结束后的某时间（取为2016-01-01）
  - PromoInterval的缺省值替代为''
- Promo2SinceWeek，Promo2SinceYear合在一起表示该店Promo2开始的时间，首先取该周周一作为该店Promo2开始的日期Promo2SinceDate；
- 对于train.csv任一行数据，如果其日期Date在该店的Promo2SinceDate之后，且其月份Month在该药店的PromoInterval表示的月份内，则构建用来表示当天是否处于Promo2的特征InPromo2Today为1，否则为0；
- 对于train.csv中InPromo2Today为1的数据，可以计算出该店从Promo2SinceDate到当天Dat的天数累积值DaysCountSincePromo2作为独立特征。

#### 特征范围调整

对特征Sales、CompetitionDistance、DaysCountSinceCompetition、DaysCountSincePromo2进行对数处理后并取整，得到新特征SalesLog, CompetitionDistance_log，DaysCountSinceCompetition_log和DaysCountSincePromo2_log。经过处理后的数据为离散值，便于进行训练。

#### Sales数据的异常值处理

对特征Sales进行对数处理得到特征SalesLog；本文采用MAD（median absolute deviation）方法来标记特征SalesLog异常值；注意：训练模型时不使用异常值，但是对数据预测时需要使用异常值。

### 执行过程

- 准备模型的训练数据dtrain和验证数据dvalid

  - 由评价指标的定义可知，训练数据和验证数据的特征Sales应不为0；同时，由于特征Open为0时，Sales也为0，所以训练数据和验证数据的特征Open应不为0；
  - 训练数据和验证数据将会用到的特征feature_x_list包含'Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'DayOfYear', 'StoreType_cat', 'Assortment_cat', 'StateHoliday_cat', 'SchoolHoliday', 'Promo', 'Promo2', 'InPromo2Today', 'DaysCountSinceCompetition_log', 'InCompetition', 'InCompetitionToday', 'CompetitionDistance_log', 'DaysCountSincePromo2_log'等18个特征；

- 构造评估函数rmspe_xg

  - 设置rmspe评估函数，输入值为不为零的销售额及相应销售额预测值

  ```python
  import numpy as np
  def rmspe(y,yhat):
      return np.sqrt(np.mean((yhat/y-1)**2))
  ```

  - 设置rmspe_xg，作为XGBoost模型的评估函数，输入值为不为零的销售额对数值及相应销售额预测值对数值

  ```python
  import numpy as np
  def rmspe_xg(yhat,y):
      y=np.expm1(y.get_label())
      yhat=np.expm1(yhat)
      return 'rmspe',rmspe(y,yhat)
  ```

- 设置模型的watchlist：训练数据dtrain作为训练集，验证数据dvalid作为验证集

  ```python
  watchlist=[(dtrain,'train'),(dvalid,'eval')]
  ```

- 设置模型初始参数值params；需要注意，通过设置seed，可以保证训练模型的可再现性。

  | 参数                            | 初始参数值 |
  | ----------------------------- | ----- |
  | 学习率learning_rate              | 0.1   |
  | 提前终止循环次数early_stopping_rounds | 100   |
  | 模型深度max_depth                 | 6     |
  | min_child_weight              | 1     |
  | subsample                     | 0.8   |
  | colsample_bytree              | 0.8   |
  | gamma                         | 0     |
  | 正则化强度reg_alpha                | 0     |
  | 最大训练次数num_boost_round         | 20000 |
  | seed                          | 42    |

- 训练模型gbm，模型在3409次循环后提前停止，此时训练集上的rmspe值为**0.113855**，验证集上的rmspe值为**0.127431**

```python
import xgboost as xgb
gbm=xgb.train(params,dtrain,num_boost_round,evals=watchlist,early_stopping_rounds=early_stopping_round,feval=rmspe_xg,verbose_eval=True)
```

- 使用训练好的模型gbm对验证集进行预测，预测的rmspe值为**0.127445**

```python
import import xgboost as xgb
import numpy as np
yhat=gbm.predict(xgb.DMatrix(X_valid[feature_x_list]))
error=rmspe(np.exp(y_valid),np.exp(yhat))
```

### 完善

在项目完成过程中，按照以下方法对模型的参数进行优化：

- 对学习率learning_rate进行优化，优化后选择**0.3**；

  - 优化过程需同时考虑模型训练效果和训练时间；
  - 设置提前终止循环次数early_stopping_round与学习率learning_rate关系为

  ```python
  early_stopping_round=int(10/learning_rate)
  ```

  - 分别采用以下学习率learning_rate，得到结果如下

  | 学习率learning_rate | 验证集上rmspe | 训练时间(秒) |
  | ---------------- | --------- | ------- |
  | 0.5              | 0.138914  | 约300    |
  | 0.45             | 0.151644  | 149     |
  | 0.35             | 0.142147  | 292     |
  | 0.3              | 0.125566  | 777     |
  | 0.25             | 0.126492  | 约1100   |
  | 0.1              | 0.127445  | 2744    |

  - 通过上表可发现学习率learning_rate在设置为0.3时，可以使得验证集上rmspe最小；
    - 原则上学习率learning_rate越小，相应模型在预测结果上也应该越好；
    - 此处发现使用0.3的学习率比0.1的学习率所训练出的模型更好，最主要原因是因为提前终止循环次数early_stopping_round设置较小，导致使用0.1学习率所训练模型提前终止；
    - 将学习率设置为0.3时效果最佳，这样可以同时兼顾训练效果和训练时间：在相同的提前终止条件下，相比于学习率0.3，学习率大于0.3时，所训练模型的泛化能力明显变差；学习率小于0.3时，所训练模型花费时间增加非常多，所训练模型的泛化能力略微下降；

- 对max_depth和min_child_weight同时进行优化，优化后max_depth选择6，min_child_weight选择1；

  - 优化过程仅考虑模型训练效果；
  - 第一次以较粗的网格进行优化，max_depth可选范围为4,6,8,10，min_child_weight可选范围为1,3,5；通过下表发现，max_depth=6，min_child_weight=1时，模型泛化能力最优；

  | 验证集上rmspe          | max_depth=4 | max_depth=6 | max_depth=8 | max_depth=10 |
  | ------------------ | ----------- | ----------- | ----------- | ------------ |
  | min_child_weight=1 | 0.153025    | 0.125566    | 0.127068    | 0.130251     |
  | min_child_weight=3 | 0.136483    | 0.128622    | 0.127268    | 0.129370     |
  | min_child_weight=5 | 0.144115    | 0.130610    | 0.127755    | 0.128333     |

  - 第二次以较细的网格进行优化，max_depth可选范围为5,6,7，min_child_weight可选范围为1,2；通过下表发现，max_depth=6，min_child_weight=1时，模型泛化能力最优；

  | 验证集上rmspe          | max_depth=5 | max_depth=6 | max_depth=7 |
  | ------------------ | ----------- | ----------- | ----------- |
  | min_child_weight=1 | 0.144219    | 0.125566    | 0.129337    |
  | min_child_weight=2 | 0.147406    | 0.132562    | 0.128306    |


- 对subsample和colsample_bytree同时进行优化，优化后subsample选择0.9，colsample_bytree选择0.7；

  - 优化过程仅考虑模型训练效果；
  - 第一次以较粗的网格进行优化，subsample可选范围为0.6,0.7,0.8,0.9，colsample_bytree可选范围为0.6,0.7,0.8,0.9；通过下表发现，subsample=0.9，colsample_bytree=0.7时，模型泛化能力最优；

  | 验证集上rmspe            | subsample=0.6 | subsample=0.7 | subsample=0.8 | subsample=0.9 |
  | -------------------- | ------------- | ------------- | ------------- | ------------- |
  | colsample_bytree=0.6 | 0.131553      | 0.129450      | 0.160291      | 0.129152      |
  | colsample_bytree=0.7 | 0.134882      | 0.132771      | 0.136842      | 0.122155      |
  | colsample_bytree=0.8 | 0.131032      | 0.125431      | 0.125566      | 0.130173      |
  | colsample_bytree=0.9 | 0.133560      | 0.133070      | 0.127994      | 0.128874      |

  - 第二次以较细的网格进行优化，subsample可选范围为0.85,0.9,0.95，colsample_bytree可选范围为0.65,0.7,0.75；通过下表发现，subsample=0.9，colsample_bytree=0.7时，模型泛化能力最优；

  | 验证集上rmspe             | subsample=0.85 | subsample=0.9 | subsample=0.95 |
  | --------------------- | -------------- | ------------- | -------------- |
  | colsample_bytree=0.65 | 0.148758       | 0.146673      | 0.132153       |
  | colsample_bytree=0.7  | 0.124185       | 0.122155      | 0.131080       |
  | colsample_bytree=0.75 | 0.129030       | 0.132518      | 0.130167       |


- 对gamma进行优化，优化后gamma选择0

  - 优化过程仅考虑模型训练效果
  - gamma可选范围为0,0.05,0.1,0.2,0.3,0.4；通过下表发现，gamma=0时，模型泛化能力最优；

  | gamma     | 0        | 0.05     | 0.1      | 0.2      | 0.3      | 0.4      |
  | --------- | -------- | -------- | -------- | -------- | -------- | -------- |
  | 验证集上rmspe | 0.122155 | 0.123412 | 0.124999 | 0.133499 | 0.129251 | 0.128723 |

- 对reg_alpha进行优化，优化后reg_alpha选择0

  - 优化过程仅考虑模型训练效果
  - reg_alpha可选范围为0,0.00001,0.01,0.1,1,100；通过下表发现，reg_alpha=0时，模型泛化能力最优；

  | reg_alpha | 0        | 0.00001  | 0.01     | 0.1      | 1        | 100      |
  | --------- | -------- | -------- | -------- | -------- | -------- | -------- |
  | 验证集上rmspe | 0.122155 | 0.123422 | 0.124343 | 0.130531 | 0.130173 | 0.229874 |

- 调整learning_rate，并设置合适的提前终止条件early_stopping_round；通过调整发现，当learning_rate=0.1，early_stopping_round=200时，模型在训练集的RMSPE可以达到**0.120141**

优化后的模型的参数汇总如下：

| 参数                            | 初始参数值 |
| ----------------------------- | ----- |
| 学习率learning_rate              | 0.05  |
| 提前终止循环次数early_stopping_rounds | 400   |
| 模型深度max_depth                 | 6     |
| min_child_weight              | 1     |
| subsample                     | 0.9   |
| colsample_bytree              | 0.7   |
| gamma                         | 0     |
| 正则化强度reg_alpha                | 0     |
| 最大训练次数num_boost_round         | 20000 |
| seed                          | 42    |

## IV. 结果
### 模型的评价与验证

最终模型为XGBoost模型：为了得到最终模型，

- 需要对特征进行选择，对特征数据进行预处理，满足模型训练要求
- 需要对学习率learning_rate、提前终止循环次数early_stopping_rounds、模型深度max_depth、min_child_weight、subsample、colsample_bytree、gamma、正则化强度reg_alpha等参数进行调整和优化，以得到最终模型；
- 最终模型参数在参数常见范围内，比较合理；

通过销售额，最终模型的预测效果可以更加直观的得到衡量，优于优于基准模型:

- 验证集上实际集的平均值、中位数、求和分别为7144.5，6564.0，327588433.0；
- 基准模型、最终模型的统计数据如下所示，可以发现：预测值与实际值之差绝对值，与实际值之比，基准模型为10.58%，而最终模型为9.16%，下降了1.43%。

| 验证集:预测值与实际值之差的绝对值 | 基准模型       | 最终模型       |
| ----------------- | ---------- | ---------- |
| 平均值               | 756.3      | 654.5      |
| 中位数               | 534.0      | 469.2      |
| 求和                | 34669773.0 | 30009543.0 |

| 验证集：预测值与实际值之差的绝对值/实际值 | 基准模型   | 最终模型  |
| --------------------- | ------ | ----- |
| 平均值                   | 10.59% | 9.16% |
| 中位数                   | 8.14%  | 7.15% |
| 求和                    | 10.59% | 9.16% |

最终模型较为合理，与期待结果一致，在训练集和验证集上的RMSPE分别为0.095274和0.120141，虽然有一定的过拟合，但是泛化能力依然较强，具有不错的鲁棒性。

### 合理性分析

基准模型与最终模型的训练集rmspe和验证集rmspe结果如下：

| 结果       | 基准模型     | 最终模型     |
| -------- | -------- | -------- |
| 训练集rmspe | 0.225588 | 0.095274 |
| 验证集rmspe | 0.140907 | 0.120141 |

- 基准模型的训练集rmspe 0.225588大于验证集rmspe 0.140907，该模型为欠拟合；
- 最终模型的训练集rmspe 0.095274略小于验证集rmspe 0.120141，该模型为过拟合，但是泛化能力依然较强；
- 最终模型的训练集rmspe 0.095274和验证集rmspe 0.120141分别小于基准模型的训练集rmspe 0.225588和验证集rmspe 0.140907，因此对于最终模型，其模型对于数据的学习程度以及泛化能力，都远优于基准模型。

## V. 项目结论
### 结果可视化

XGBoost模型在训练过程中，可以得到不同特征的重要性，重要性前三的特征为Store, DayOfYear和DayOfWeek.

![](./Rossmann_Store_Sales/insertedPics/5_0_xgboost_feature_importance.png)

### 对项目的思考

本项目主要实现过程包括：

- 数据探索与可视化
- 选择评价指标与基准模型
- 数据预处理以及训练模型
- 结果与分析

在实现过程中，模型的选择与优化是比较困难的地方：为了训练出学习能力好、泛化能力强的模型，在模型初选阶段，需要对多种模型进行大量尝试；除了项目最后选用的XGBoost方法，本项目还尝试了随机森林、深度学习embedding方法，但是训练出的模型效果相比基准模型进步不大，但是各种方法（尤其是深度学习embedding方法）的实现以及优化是需要较多的知识储备以及模型训练时间；

最终模型采用了Kaggle比赛中针对回归预测类非常通用XGBoost方法，其预测效果也显出了很强的泛化能力，相比基准模型也有很大的提高；此外，本项目的实现方案对于回归预测类问题具有一定的通用性。

### 需要作出的改进

本项目可以从以下方面实现进一步的完善：

- XGBoost模型的集成：既可以使用相同特征、采用不同随机seed训练出XGBoost模型，也可以使用不同特征训练出XGBoost模型，然后对这些模型的结果进行集成；
- 采用深度学习embedding方法进行实现。Kaggle比赛中，第三名[Cheng Guo【5】](https://arxiv.org/pdf/1604.06737.pdf) 将原本主要用于自然语言处理的深度学习entity embedding模型应用到该项目，取得第三名，由此可见该方法的强大之处。

参考文献：

【1】Kaggle比赛-Rossman Store Sales: https://www.kaggle.com/c/rossmann-store-sales#description
【2】周志华. 机器学习[M]. 北京：清华大学出版社, 2016.
【3】Tianqi Chen. XGBoost- A Scalable Tree Boosting System[J].  arXiv,2016. 
【4】Gert. Winning Model Documentation-describing my solution for the Kaggle competition-“Rossmann Store Sales”.2015. https://kaggle2.blob.core.windows.net/forum-message-attachments/102102/3454/Rossmann_nr1_doc.pdf.
【5】[Cheng Guo.Entity Embeddings of Categorical Variables.[J].  arXiv,2016.]((https://arxiv.org/pdf/1604.06737.pdf)) 	
【6】Arshay Jain.  Complete Guide to Parameter Tuning in XGBoost (with codes in Python). 2016: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python

