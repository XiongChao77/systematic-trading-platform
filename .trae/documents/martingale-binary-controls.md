# 马丁未击穿二分类对照实验

## Context
用户明确指定：假设最大层数 X 分别为 4、5、6，直接按已有原始深度标签划分，标签 <=X 为 Y（未击穿），标签 >X 为 N（击穿）。不做额外的层数偏移。新增三个独立二分类组，保留原回归基线。

## 标签和预测语义
| 最大层数 X | 正类 1：Y，未击穿 | 负类 0：N，击穿 |
|---|---|---|
| 4 | label <=4 | label >4 |
| 5 | label <=5 | label >5 |
| 6 | label <=6 | label >6 |

模型输出 survival_probability=P(Y)=P(label<=X)，越高表示预测未击穿概率越高；breach_probability=1-survival_probability。这里未击穿指完整标签周期内未越过指定深度，不是固定未来时间范围内的判断。

## 实现方案
1. 在 model/train_config.py 和 model/tasks/strategies.py 中增加参数化 MARTINGALE_BINARY 任务与 max_layers 配置。在 task.dataset_split 内将原始有效 0～9 标签映射为 (label<=X)，不覆盖制备数据；无效或未完成标签不得变成负类。不复用交易 Signal 或方向惩罚。
2. 复用现有 XGBoost、train_rolling、模型工厂和标准保存机制。三个实验使用相同数据、特征、窗口、stride、随机种子及现有标签成熟边界；默认排除 Forward。采用二分类交叉熵，logloss 早停与选模。主对照不做类别重采样或逆频率加权，保留原始概率估计口径。
3. 在 model/model_loader.py、model/artifacts.py 和 model/train.py 中接入加载和预测：保存 X、比较运算符 <=、Y/N 与类别编码，输出 survival_probability 及其补数 breach_probability，保留原始深度标签、时间与 fold。不映射成交易方向，保持原回归功能不变。
4. 在 experiment/martingale_ml.py 增加实验选择，支持回归及 X=4/5/6 三个二分类组，保存到独立且不覆盖的目录。实际对照复用已有 preparation 产物和原回归结果，只新增训练三个二分类组。
5. 复用分类评估并补充 logloss、Brier、ROC-AUC；同时报告击穿类的 AP 和原始击穿占比，避免正类 Y 占比很高导致准确率、Y 类 AP 看起来虚高。单类区间的 AUC 明确记为不可计算。
6. 比较共同 OOS 时间点上的原始 Y/N 占比、按 survival_probability 从低到高分组后的真实 Y/N 比例、高分放行组的击穿率及相对基准降幅、各 fold 稳定性。全段 OOS 分位数仅作为事后诊断，不充当可部署阈值。

## 验证与执行
- 使用 test/ 下 test_ 开头的临时验证文件核对每个 X 的 X-1、X、X+1 标签，确保边界 X 属于 Y；核对空标签、成熟时间剔除、保存加载后概率一致及两类概率互补。完成后删除临时验证文件。
- 检查原回归入口与现有模型加载不受影响，执行编译与 diff 检查。
- 验证通过后复用已有 DOGEUSDT 5m 制备目录，运行 X=4、5、6 三个滚动实验，输出 OOS 对比，不分析 Forward。
