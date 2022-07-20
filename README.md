
1、离线模型的执行
运行run.py，执行“预处理-二部图推荐-图卷积推荐”3步

2、在线接口的调用
interface.py提供了接口，基于from OutputFusion.outputFusion中的format_result文件实现结果输出，即给定用户ID，返回推荐结果。

注意outputFusion汇聚了多个源头的数据，除了读取数据库，还包括来自online.onlineRecommend.py的online_run()函数
该函数基于指定的**时间周期参数**执行了离线推荐与在线推荐的结合。
