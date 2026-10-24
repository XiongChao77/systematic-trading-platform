# architecture
```mermaid
flowchart TD
    A[Model Design]

    A --> A1[Single Model]
    A1 --> A11[Direct Three-Class]
    A1 --> A12[Two-Head Three-Class]

    A --> A2[1+1 Two-Model Combination]
    A2 --> A21[Trigger + Direction]
    A2 --> A22[Long/Short OVR]

    A11 --> B[Trained Model and Backtest]
    A12 --> B
    A21 --> B
    A22 --> B

    B --> C[Result Analysis]
    C --> C1[Model Analysis]
    C --> C2[Financial Analysis]
```

***************Model******************
•	Convolution Neural Networks
•	Recurrent Neural Networks (RNN)
•	Deep Autoencoders (unsupervised learning)
•	GAN (2014)
•	Deep Forrest (2017)
•	Transformer (GPT3, 2020)


--------------------Problem---------------------------
*Should early stopping follow va_loss or va_macroF1? How to choose
    Should a trading proxy metric be used during validation as well?
*unbalanced class

*tr_loss training loss
*va_loss validation loss
*va_macroF1 harmonic mean of precision + recall, 0 = pure guessing, 1 = perfect classification.
    Rules of thumb:
    Random (multi-class): ~ 0.2~0.25
    Barely useful: >0.3
    Some signal: >0.35
    Fairly strong: >0.45
Overfitting: va_loss >> tr_loss and va_macroF1 does not rise while tr_loss falls -- the model learned the training set's patterns / noise
Feed part of the magnitude information into the loss function and see whether the model can learn more

## Training and evaluation plots

Each single-model training run saves `precision_recall_curve.png` using the
final test-set predictions. Binary tasks plot the positive class; multiclass
tasks plot one-vs-rest curves for each class.

Runs also save `threshold_precision_recall_curve.png`, showing precision and
recall as the decision threshold varies for the positive class or each
one-vs-rest class.

Runs with per-epoch or per-boosting-round metrics also save `training_curve.png`
and `history.json`, containing train and validation loss. XGBoost prints train
and validation log loss for each boosting round and stops after 30 rounds
without validation log-loss improvement. The saved XGBoost booster is sliced
to the lowest-validation-loss round. The test set is evaluated only after
training and is not used for per-epoch monitoring or model selection.
