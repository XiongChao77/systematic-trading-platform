# 马丁机器学习实验入口

## Context
新增 `experiment/martingale_ml.py`，一条命令执行马丁网格数据准备与模型训练。现有 common 已提供纯价格0～9标签，通用训练链仍属于Signal任务，本次不移除其保护、不改写标签逻辑、不新增回测。

## 实施方案
1. **独立实验入口与参数**
   - 默认 DOGEUSDT、5m、long、1m细周期、网格步长0.01、初始价格止盈比例0.01。
   - CLI支持资产、方向、周期、价格参数、时间范围与development/forward分界、滚动窗口和少量XGBoost参数。
   - 使用独立唯一运行目录，保存prep、模型、OOS预测、指标、配置及日志，不覆盖旧产物。
2. **复用数据准备**
   - 调用 `data_process/common.py` 的 `MartingaleDefine` 和 `data_process/preparation.py` 的 `main`。
   - 用 `load_train_df_from_dir`、`load_data_manifest_from_dir`读取development数据和持久化边界；默认不分析forward。
   - 特征采用 `FeatureFactory` 注册白名单与实际列的交集，保持固定顺序；排除label、全部martingale结果列及其它未来结果列。第一版采用当前已收盘K线的二维特征，不加入序列模型。
   - 保留完整时间轴，空标签只在选择监督样本时排除，不填充为有效类别。
3. **滚动训练与防泄漏**
   - 复用 `model/train.py::rolling_windows` 与 `model/train_config.py::RollingConfig`；默认expanding，训练/验证/OOS为12/2/1个月，支持sliding。
   - 标签成熟时间保守取 `martingale_exit_time + 主周期长度`。Train事件必须在验证起点前成熟，Valid事件必须在OOS起点前成熟。
   - OOS所有入场行保留预测；仅对在development结束前成熟的有效标签评分。
   - 原生 `xgboost.train`，固定 `multi:softprob`、`num_class=10`，避免现有sklearn适配器在某折缺类时改变概率维度。不伪造缺失类别样本。
   - 仅用验证集早停；预测使用最佳迭代。保留通用训练对MARTINGALE的拒绝保护。
4. **结果与可运行性**
   - 每折保存模型、最佳迭代、边界、样本数、成熟过滤数和0～9类别分布。
   - OOS CSV包含symbol、时间、fold、真实标签、可评分标记、预测类别、p_0～p_9。
   - 输出逐折及整体OOS logloss、accuracy、macro-F1、逐类precision/recall/support和10×10混淆矩阵，附训练类别先验基线。
   - manifest记录特征顺序、类别语义、来源data_id、配置、随机种子和库版本。缺少足够时间窗口或有效训练/验证样本时明确报错。

## 关键文件
- 新增 `experiment/martingale_ml.py`。
- 直接复用 common、preparation、rolling_windows、RollingConfig；原则上不修改既有训练、扫描及报告。

## 验证
- CLI帮助和语法检查。
- 临时验证文件仅放顶层test目录且以test_开头，验证后移除。
- 合成数据覆盖缺失类别仍有10列概率、概率归一、事件成熟过滤、结果列隔离、空标签保留OOS预测、时间不重叠、模型重载预测一致。
- 小范围端到端准备及滚动训练冒烟，核对产物与指标；不自动运行全历史重训练，不改写现有研究报告，不新增永久孤立测试。
