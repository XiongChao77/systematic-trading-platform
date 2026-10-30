# Strategy Validation Pipeline

## Objective

The purpose of this pipeline is to eliminate obvious overfitting, parameter sensitivity, time-split dependence, concentrated returns, and execution-dependent strategies before using the final OOS dataset.

The model and all decision rules must remain frozen during each validation stage. The OOS dataset is reserved for auditing the complete screening process rather than selecting individual strategies.

## Stage 1: Train / Valid / Test Performance Retention

Treat Valid and Test as one continuous post-training period and calculate:

- Train CAGR
- Combined Valid+Test CAGR

A strategy passes when:

\[
CAGR_{Train} > CAGR_{min}
\]

and

\[
CAGR_{Valid+Test} \ge CAGR_{Train} \times retention\_ratio
\]

Example thresholds:

- Minimum Train CAGR: 30%
- Minimum Valid+Test retention: 20% of Train CAGR

This stage removes strategies that perform well during Train but collapse after leaving the training region. Once this screening has been applied, additional highly correlated filters based on the same Train/Valid/Test equity curves should be limited.

## Stage 2: Strategy-Parameter Robustness

Keep the trained model and its predictions fixed. Perturb only downstream strategy and execution parameters, such as:

- Holding period
- Take-profit and stop-loss distances
- Entry threshold
- Position sizing
- Cooldown period
- Fees, slippage, and execution delay

The objective is to determine whether the model's predictive advantage remains usable across a reasonable neighborhoos of trading parameters. A credible strategy should lie on a stable performance region rather than depend on one isolated parameter combination.

This test does not require deploying multiple strategies or models.

## Stage 3: Training-Pipeline Stability

Model stability is a separate experiment from strategy-parameter robustness.

Do not manually perturb learned model weights. Instead, repeat the normal training process with controlled changes to:

- Random seed
- Training sample or time-block resampling
- Regularization strength
- Model complexity
- Other relevant training hyperparameters

Different runs may produce structurally different models. Therefore, compare their outputs rather than their internal weights:

- Prediction or ranking correlation
- Signal-direction agreement
- Trade overlap
- Valid/Test performance distribution
- Proportion of successful training runs

The purpose is to determine whether the training pipeline repeatedly produces useful models, not whether every run produces the same fitted model.

## Stage 4: Time-Boundary Robustness

Keep the model specification and evaluation process unchanged while making small shifts to:

- The Train end date
- The Valid/Test boundary
- The start of the training window

If a small boundary change turns a profitable strategy into a clear failure, the result may depend excessively on a specific market interval or split location.

## Stage 5: Null and Placebo Tests

Run the complete training and backtesting pipeline against controls that should contain no genuine predictive information, such as:

- Time-block label permutations
- Large circular shifts of the signal
- Mismatched asset-signal pairs
- Random signals with similar trading frequency

The real strategy should outperform the resulting null distribution. This stage tests whether the research pipeline itself can easily manufacture attractive results from noise.

## Stage 6: Return-Concentration Analysis

Determine whether performance depends excessively on a small number of trades or market intervals:

- Largest winning month's share of total profit
- Largest trades' share of total profit
- Performance after removing the best 1% or 5% of trades
- Return contribution by quarter or market regime
- Long-side and short-side contribution

Confidence is lower when removing a very small number of trades eliminates the strategy's entire advantage.

## Stage 7: Cost and Execution Robustness

Apply more conservative execution assumptions:

- Higher fees and slippage
- Delayed execution
- Adverse price movement before fills
- Missed orders
- Take-profit and stop-loss execution errors

This stage does not directly prove or disprove statistical overfitting. It removes strategies whose apparent advantage cannot survive realistic execution.

## Stage 8: Frozen Cross-Period and Cross-Asset Validation

Freeze the following before testing other historical periods or economically related assets:

- Features
- Label definition
- Model specification
- Strategy parameters
- Screening criteria
- Risk rules

Different assets do not need to produce identical CAGR. The main question is whether the predictive direction, source of returns, and advantage over the null baseline can be reproduced without adaptation.

These datasets are part of the screening process and are not the final OOS audit dataset.

## Final Stage: OOS Process Audit

Before revealing the OOS results, freeze:

- The complete candidate strategy set
- Every model and parameter
- All screening criteria
- The OOS success definition
- The expected OOS success rate

OOS evaluates the effectiveness of the complete strategy-discovery and screening process. Its primary outputs are:

- Number of strategies entering OOS
- OOS success count and success rate
- Performance retention relative to Valid+Test
- Common failure modes across the candidate set

OOS must not be used to keep individual winners and discard individual losers. Doing so would turn OOS into another selection dataset.

The valid conclusions are process-level conclusions:

- If the predefined OOS success rate is achieved, the screening pipeline receives independent support.
- If the success rate is materially below expectation, the screening pipeline is not validated and should be revised as a whole.

Once OOS results influence any rule, parameter, or model decision, that OOS period has been consumed. A revised pipeline requires new future data for another independent audit.

## Complete Workflow

\[
Train/Valid/Test\ Retention
\rightarrow Strategy\ Parameter\ Robustness
\rightarrow Training\ Pipeline\ Stability
\rightarrow Time\ Boundary\ Robustness
\rightarrow Null\ Tests
\rightarrow Return\ Concentration
\rightarrow Execution\ Robustness
\rightarrow Cross\ Period/Asset\ Validation
\rightarrow Freeze\ Candidate\ Set
\rightarrow OOS\ Process\ Audit
\]

The central principle is that each stage should contribute a different type of evidence instead of repeatedly extracting highly correlated metrics from the same equity curve.
