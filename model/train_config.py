import hashlib
from dataclasses import dataclass, field
from typing import List, Optional

from data_process import feature


# ==============================================================================
# 1. Configuration
# ==============================================================================
@dataclass
class DataConfig:
    label_col: str = "label"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    purge_overlap: bool = True

    def __post_init__(self):
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        if not 0 < self.val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must leave a non-empty test split")


@dataclass
class BaseModelConfig:
    seq_len: int = 96
    model_type: str = "base"
    model_version: int = 1


class TrainTask:
    DIRECT_3CLASS = "DIRECT_3CLASS"
    DUAL_HEAD_3CLASS = "DUAL_HEAD_3CLASS"
    TRIGGER_DIRECTION = "TRIGGER_DIRECTION"
    LONG_SHORT_OVR = "LONG_SHORT_OVR"

    TRIGGER = "TRIGGER"
    DIRECTION = "DIRECTION"
    LONG_OVR = "LONG_OVR"
    SHORT_OVR = "SHORT_OVR"

    BINARY_TASKS = {TRIGGER, DIRECTION, LONG_OVR, SHORT_OVR}
    COMBO_TASKS = {TRIGGER_DIRECTION, LONG_SHORT_OVR}


@dataclass
class LSTMConfig(BaseModelConfig):
    seq_len: int = 216  # 160 best for LSTM
    model_type: str = "lstm"
    model_version: int = 1
    hidden_size: int = 32
    num_layers: int = 2
    bidirectional: bool = True
    lstm_dropout: float = 0.2
    head_dropout: float = 0.2
    p_drop: float = 0.3
    readout: str = ["last", "meanmax", "attn", "mix"][1]
    head: str = ["linear", "mlp"][0]
    in_locked_p: float = 0.1  # V4 locked dropout on inputs
    out_locked_p: float = 0  # V4 locked dropout on LSTM outputs (before pooling)
    input_norm: bool = True  # V4 LayerNorm on input features
    input_proj_dim: int | None = None  # V4 optional projection before LSTM: a linear layer mapping raw feature dim (e.g. 48) to a new dim D (dim reduction)
    logit_clip: float | None = None  # V4


@dataclass
class TransformerConfig(BaseModelConfig):
    seq_len: int = 216  # 160 best for LSTM
    model_type: str = "transformer"
    model_version: int = 1
    d_model: int = 64
    nhead: int = 8
    num_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.3
    attn_dropout: float = 0.1
    drop_path: float = 0
    in_locked_p: float = 0
    use_alibi: bool = True
    pos_encoding: str = "none"
    cls_token: bool = False
    readout: str = "meanmax"  # "cls" | "meanmax" | "attn" | "mix"
    head: str = "linear"
    ffn_type: str = "swiglu"
    use_feature_weighting: bool = False


@dataclass
class ConvLSTMConfig(BaseModelConfig):
    seq_len: int = 160  # 160 best for LSTM
    model_type: str = "conv_lstm"
    model_version: int = 1
    d_model: int = 64
    hidden_size: int = 32
    conv_kernel: int = 3
    conv_dropout: float = 0.2
    conv_layers: int = 1
    conv_dilations: tuple[int, ...] | None = None
    bidirectional: bool = True
    lstm_dropout: float = 0.2
    input_norm: bool = True
    in_locked_p: float = 0.1
    out_locked_p: float = 0.1
    head_dropout: float = 0.3
    readout: str = "meanmax"  # 'last'|'meanmax'|'attn'|'mix'
    head: str = "linear"  # 'linear'|'mlp'
    logit_clip: Optional[float] = None
    p_drop: Optional[float] = None
    task_proj_dim: int = 32
    use_feature_weighting: bool = False


@dataclass
class TCNConfig(BaseModelConfig):
    seq_len: int = 216  # 160 best for LSTM
    model_type: str = "tcn"
    model_version: int = 1
    num_channels: list = field(default_factory=lambda: [64, 128, 256])
    kernel_size: int = 3
    dropout: float = 0.2
    readout: str = "mix"
    logit_clip: Optional[float] = None


@dataclass
class MambaConfig(BaseModelConfig):
    seq_len: int = 216  # 160 best for LSTM
    model_type: str = "mamba"
    model_version: int = 1
    d_model: int = 128
    n_layers: int = 4
    d_state: int = 16
    expand: int = 2
    dropout: float = 0.1
    readout: str = "mix"  # 'last' | 'meanmax' | 'mix'
    logit_clip: Optional[float] = None


@dataclass
class XGBoostConfig(BaseModelConfig):
    seq_len: int = 96
    model_type: str = "xgboost"
    model_version: int = 2

    xgb_depth: int = 4
    xgb_estimators: int = 300
    xgb_early_stopping_rounds: int = 30
    learning_rate: float = 0.03

    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.5
    reg_alpha: float = 0.0

    tree_method: str = "hist"
    xgb_device: str = "cuda"  # "cpu" or "cuda"
    eval_metric: str = "logloss"
    use_scaler: bool = False  # tree models need no standardization


@dataclass
class LightGBMConfig(BaseModelConfig):
    seq_len: int = 96
    model_type: str = "lightgbm"
    model_version: int = 1

    num_leaves: int = 31
    max_depth: int = -1
    n_estimators: int = 300
    learning_rate: float = 0.03
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_child_samples: int = 20
    use_scaler: bool = False


@dataclass
class SVCConfig(BaseModelConfig):
    seq_len: int = 96
    model_type: str = "svc"
    model_version: int = 1

    C: float = 1.0
    kernel: str = "rbf"
    gamma: str = "scale"
    use_scaler: bool = True


@dataclass
class LogisticRegressionSklearnConfig(BaseModelConfig):
    seq_len: int = 96
    model_type: str = "logistic_regression_sklearn"
    model_version: int = 1

    C: float = 1.0
    penalty: str = "l2"
    solver: str = "lbfgs"
    max_iter: int = 1000
    use_scaler: bool = True


@dataclass
class CNNConfig(BaseModelConfig):
    seq_len: int = 216  # 160 best for LSTM
    model_type: str = "cnn"
    model_version: int = 1
    p_drop: float = 0.3
    tau: float = 16.0
    use_tpool: bool = False


@dataclass
class TrainConfig:
    model_cfg: BaseModelConfig = field(default_factory=lambda: ConvLSTMConfig(model_version=3))
    data_cfg: DataConfig = field(default_factory=DataConfig)
    feature_conf_list: Optional[List[str]] = field(default_factory=lambda: feature.FEATURE_LIST_CRYPTOCURRENCY)
    epochs: int = 100
    batch_size: int = 256  # 256
    lr: float = 3e-4
    gate_lr: float = 3e-4
    weight_decay: float = 5e-4
    patience: int = 8
    seed: int = 42
    stride: int = 4
    use_cache: bool = False
    lambda_dir: float = 0.1  # Importance of long/short direction
    lambda_main: float = 0.7
    lambda_cost: float = 1  # Flip / missed trend / noisy trades
    flip_penalty: float = 1
    miss_penalty: float = 1
    false_trade_penalty: float = 1
    minority_sampling_ratio: Optional[float] = None
    mag_warmup_epochs: int = 8
    temperature: float = 2.0
    selection_metric: str = "macro_f1"  # "macro_f1" or "loss"
    label_smoothing: float = 0.02
    train_compatibility: str = ""
    train_task: str = TrainTask.DIRECT_3CLASS

    def __post_init__(self):
        if self.selection_metric not in {"macro_f1", "loss"}:
            raise ValueError("selection_metric must be 'macro_f1' or 'loss'")
        if self.epochs <= 0 or self.batch_size <= 0 or self.patience <= 0:
            raise ValueError("epochs, batch_size and patience must be positive")
        if self.minority_sampling_ratio is not None and not 0 < self.minority_sampling_ratio < 0.5:
            raise ValueError("minority_sampling_ratio must be between 0 and 0.5")
        feature_hash_input = (
            "\0<ALL_FEATURES>"
            if self.feature_conf_list is None
            else "\0".join(self.feature_conf_list)
        )
        feature_hash = hashlib.sha256(feature_hash_input.encode("utf-8")).hexdigest()[:12]
        self.train_compatibility = f"{self.model_cfg.seq_len}_{self.stride}_{feature_hash}"


@dataclass
class LogisticConfig(BaseModelConfig):
    seq_len: int = 96  # 160 best for LSTM
    model_type: str = "logistic_regression"
    model_version: int = 1


COMBO_SUB_TASKS = {
    TrainTask.TRIGGER_DIRECTION: (TrainTask.TRIGGER, TrainTask.DIRECTION),
    TrainTask.LONG_SHORT_OVR: (TrainTask.LONG_OVR, TrainTask.SHORT_OVR),
}

seq_len = 48
SingleModelConfig = TrainConfig(
    model_cfg=XGBoostConfig(model_version=2, seq_len=seq_len),
    feature_conf_list=None,#feature.FEATURE_LIST_CRYPTOCURRENCY,
    train_task=TrainTask.DIRECT_3CLASS,
    stride=2,
)
