# DOGEUSDT 15m Trigger/Direction research

## Conclusions from 2026-11-21

Use `stop_multiplier_rate = 0.618` and `fixed_hold_bars = 96` as the baseline for the next experiment. Continue exploring `xgb_depth` in `[4, 5, 6]`; depth 6 performs better in some configurations, so the preferred depth remains open.

| Parameter | Conclusion | Next step |
| --- | --- | --- |
| `stop_multiplier_rate` | 0.618 performs better in this round. | Use 0.618 as the baseline. |
| `xgb_depth` | Depths 4, 5 and 6 remain candidates; 6 is sometimes better. | Further explore `[4, 5, 6]` and their Trigger/Direction combinations. |
| `min_child_weight` | This round shows no clear differentiation among the tested values. | Do not select a preferred value from these results. |
| `learning_rate` | At 0.01, 100 training rounds are insufficient. | Increase the training-round cap before judging 0.01 against other learning rates. |
| `fixed_hold_bars` | 96 performs better in this round. | Use 96 as the baseline. |

This entry records the researcher's conclusions for the round documented on 2026-11-19; the detailed metrics and evidence links remain in that entry. It supersedes that entry's narrower depth and learning-rate recommendations for the next exploration. The earlier 4+5 configuration remains a historical reference, and Forward evaluation remains open.

## Conclusions from 2026-11-19

Keep the earlier 4+5 configuration as the provisional reference. This sweep favors a stop ratio of 0.618 and identifies Trigger depth 6 as a strong candidate under the 100-round cap. The best individual run reached 19.24% CAGR, compared with the earlier reference's recorded 24.49% at caps 500/200. Both figures use the same Development OOS dates. The old run's linked files are unavailable at their original paths on the current mount.

| Parameter | Conclusion | Key evidence |
| --- | --- | --- |
| Stop ratio | Prefer 0.618 for the combined label and execution setup. | It beats 0.8 in 80.6% of matched runs. Median CAGR is +0.52% versus −5.54% and median profit factor is 1.002 versus 0.972, despite median win rates of 43.16% versus 45.82%. |
| Holding period | Use 96 as the baseline for the 0.618 group. Holding 48 bars has the highest median CAGR in the 0.8 group, where all three hold medians are negative. | At stop 0.618, hold 96 beats 48 in 57.1% of matched pairs, and hold 128 beats 96 in 41.2%. |
| Trigger depth | Depth 6 leads this sweep across both stop ratios. | It beats depth 4 in 76.4% of matched runs and beats depth 5 in 69.8% and 74.8% of matched runs at stops 0.618 and 0.8, respectively. |
| Direction and joint depth | Direction depth 3 leads marginally across the whole grid. At stop 0.618 and hold 96, 6+5 leads the joint comparison. | At those execution settings median CAGR is 8.99% for 6+5, 4.88% for 6+3 and 0.90% for 4+5. |
| Learning rate | Use 0.02 for both components as the baseline for 6+5 at stop 0.618 and hold 96. Direction 0.05 usually underperforms 0.02 across the full grid, with a stronger penalty at stop 0.618. | The 0.02/0.02 pair leads the 6+5 subgroup by median CAGR (+11.81%, nine child-weight pairs). Direction 0.05 beats 0.02 in 19.4% of matched runs at stop 0.618 and 44.1% at stop 0.8. |
| Minimum child weight | Results are mixed across 1, 2 and 5. | Changing 1 to 2 gives median matched CAGR changes of −0.21 percentage points for Trigger and +0.18 for Direction. |
| Actual boosting rounds | Lower Direction learning rates increase retained rounds without a clear return gain over 0.02. The 100-round cap constrains the learning-rate comparison. | Direction retains a median of 61 / 29 / 11 rounds at rates 0.01 / 0.02 / 0.05. At 0.01, 44.4% of Direction fits and 97.4% of Trigger fits reach the execution cap. Early-stopping patience is fixed at 30. |

For the next broad sweep, drop stop 0.8 and deprioritize Direction depth 4 and hold 128. Keep the best 6+4/hold-128 run as a specific comparison. Within stop 0.618 and hold 96, retain 6+5 across learning rates and 6+6 at 0.02/0.02. With both rates fixed at 0.02, 6+6 beats 6+5 on CAGR in seven of nine matched child-weight pairs and reduces drawdown in all nine. Keep 5+3 with both rates at or below 0.02 as a shallower comparison: all 36 configurations are profitable. Direction rate 0.05 can be dropped for Trigger depths 4–5 in the core stop/hold region, but remains competitive for 6+5 with Trigger rate 0.02.

Of 5,832 backtests, 1,932 were profitable (33.13%). Median CAGR was −2.93% and median maximum drawdown was 50.67%. The top run `e93694222ed6` used stop 0.618 and hold 128. Its Trigger used depth 6, rate 0.05 and child weight 1, and its Direction used depth 4, rate 0.01 and child weight 5. It returned +127.31% total with Sharpe 0.715, maximum drawdown 29.87% and Calmar 0.644. Its gains were concentrated in 2022 and 2025, with negative returns in 2023, 2024 and partial 2026. The full Development reports logged unexplained meltdown days in 5,703 runs. Those events have no stage attribution in the aggregate report.

Run `DOGEUSDT_15m/2026-11-19/00_49_44` used 126 component configurations and 1,944 fusion pairs. The sweep fixed sequence length at 48, the estimator cap at 100 and the label horizon at 96. Development OOS covers `[2021-11-27 23:45, 2026-07-27 23:45)` UTC. Metrics use continuous account slices, and matched comparisons change one parameter at a time. The trials share dates and component models. Forward evaluation remains open. [Task specification](../../../quant_output/batch_experiments/DOGEUSDT_15m/2026-11-19/00_49_44/tasks_spec.json) · [OOS metrics](../../../quant_output/batch_experiments/DOGEUSDT_15m/2026-11-19/00_49_44/analysis/oos_completed/oos_metrics.csv) · [Matched comparisons](../../../quant_output/batch_experiments/DOGEUSDT_15m/2026-11-19/00_49_44/analysis/oos_completed/paired_cagr.csv) · [Boosting histories](../../../quant_output/batch_experiments/DOGEUSDT_15m/2026-11-19/00_49_44/analysis/oos_completed/actual_boosting_rounds.csv) · [Top run equity](../../../quant_output/batch_experiments/DOGEUSDT_15m/2026-11-19/00_49_44/sim_output/e93694222ed6/development/equity_curve_absolute.png)

## Conclusions from 2026-11-18

**Keep Trigger depth 4, Direction depth 5, input sequence 48 bars and maximum hold 96 bars as the provisional reference.** The reference run uses estimator caps of 500 for Trigger and 200 for Direction. It earned 24.49% annualized with 30.10% maximum drawdown on the Development OOS selection period. Forward performance remains unevaluated in this experiment.

| Parameter | Conclusion | Key evidence |
| --- | --- | --- |
| Input sequence length | Prefer 48 among the tested lengths. The improvement over 32 comes with a modest increase in win rate and fewer trades. The 64-bar sequence trades least and underperforms 48. | For lengths 32 / 48 / 64, median CAGR was −5.88% / −3.08% / −4.44%, median win rate was 43.24% / 44.00% / 43.85%, and median trades/day was 1.198 / 1.157 / 1.092. |
| Maximum holding period | Moving from 32 to 48 bars improves returns across most model pairs. Moving from 48 to 96 gives mixed results overall. Keep 96 for the reference model, where fewer trades and a better payoff ratio accompany higher returns despite a lower win rate. | 48 beats 32 in 90.7% of matched pairs, and 96 beats 48 in 51.5%. For the reference model, 48 → 96 raises CAGR from 16.20% to 24.49% and the mean net win/loss ratio from 1.251 to 1.389, while trades fall from 1,979 to 1,541 and win rate from 45.98% to 44.19%. |
| Trigger depth | Depths 4 and 5 are the stronger choices across the grid. The improvement over depth 1 does not come with higher trading win rates or fewer trades. | Depth 4 has the best marginal median CAGR (−2.00%). Depth 5 beats depth 1 in 69.7% of matched runs, with a median change of +144 trades and −0.13 percentage points in win rate. |
| Direction depth | Depth 5 leads the marginal comparison. Compared with depth 1, it generally improves returns, trading win rate and drawdown even though conditional classification accuracy falls. Classification accuracy alone can misrank the trading models. | Depth 5 beats depth 1 on CAGR in 81.1% of matched runs. Median changes are +1.02 percentage points in trading win rate, −6.61 points in maximum drawdown and −2.03 points in conditional Direction accuracy. |
| Joint Trigger + Direction depth | Prefer 4+5 for sequence 48 and hold 96. The preferred pair depends on sequence and hold. | With sequence 48 and hold 96 fixed, median CAGR is 23.42% for 4+5, 10.78% for 5+5 and 9.46% for 5+3. Across the full grid, 5+3 has the highest median CAGR (+3.15%). |
| Estimator cap | Raising either cap gives no consistent return improvement. For Direction, 500 and 1,000 are almost always equivalent in this grid. | Direction caps 500 and 1,000 give identical reported CAGR, trade count, win rate and profit factor in 899 of 900 matched comparisons. |
| Actual boosting rounds | More executed rounds have no consistent relationship with returns across complete backtests. For the reference setup, additional Trigger rounds help up to the middle of the observed range, then returns flatten or decline. Direction can execute extra patience rounds with the same saved predictor and return. | With depths, sequence, hold and the Direction model fixed, Trigger totals of 5,343 / 8,683 / 11,840 / 13,125 executed rounds give 4.86% / 22.77% / 24.49% / 23.52% OOS CAGR. Each total covers the same 56 updates. |

The actual rounds comparison pairs each complete backtest's OOS return with its total executed rounds. The saved reference models retain an average of 183.6 Trigger rounds and 31.1 Direction rounds per update. Whether changing learning rate or early stopping improves Direction performance remains open.

## Overall result

Only 1,042 of 3,600 configurations were profitable in Development OOS. The median CAGR was −4.62% and median maximum drawdown was 56.54%. The reference returned +177.92% in total, with Sharpe 0.877 and maximum drawdown 30.10%. Its annual returns were positive in 2022, 2024, 2025 and partial 2026, and negative in 2023 (−7.11%). These results support keeping the reference as a candidate selected on Development OOS.

## Run and evidence

- Run `DOGEUSDT_15m/2026-11-16/16_15_30`. Monthly expanding training starts with 12 months of training and 4 months of validation. The sweep covers 120 component configurations, 1,200 fusion pairs and 3,600 backtests. [Task specification](../../../quant_output/batch_experiments/DOGEUSDT_15m/2026-11-16/16_15_30/tasks_spec.json) · [Frozen resume plan](../../../quant_output/batch_experiments/DOGEUSDT_15m/2026-11-16/16_15_30/resume_plan.json)
- Development OOS is `[2021-11-27 23:45, 2026-07-27 23:45)` UTC. The corrected metrics slice one continuous account and include positions carried into the OOS period. Earlier full Development figures included first-model training and validation and are superseded for OOS comparisons. [Corrected results](../../../quant_output/batch_experiments/DOGEUSDT_15m/2026-11-16/16_15_30/analysis/oos_completed/analysis.md) · [Full metric table](../../../quant_output/batch_experiments/DOGEUSDT_15m/2026-11-16/16_15_30/analysis/oos_completed/oos_metrics.csv)
- Parameter comparisons and supporting figures. [Parameter analysis](../../../quant_output/batch_experiments/DOGEUSDT_15m/2026-11-16/16_15_30/analysis/oos_completed/parameter_analysis.md) · [Joint depth heatmap](../../../quant_output/batch_experiments/DOGEUSDT_15m/2026-11-16/16_15_30/analysis/oos_completed/depth_joint_heatmap.png) · [Actual rounds across complete backtests](../../../quant_output/batch_experiments/DOGEUSDT_15m/2026-11-16/16_15_30/analysis/oos_completed/actual_rounds_between_backtests.md) · [Equity curve index](../../../quant_output/batch_experiments/DOGEUSDT_15m/2026-11-16/16_15_30/equity_curve_index.csv)
- Reference backtest `f793d7d0b0ce`, fusion `5a134e39e2c9`. Data ID `8f15cff10ec780fcdc2c331505f4dfb72c4d368a63807fb62ecbe023d9a319a4`. The run reported Git revision `10c743a50ddc63a79d4fdba255b10fe6533c081e` and also used local resume and accounting repairs.
