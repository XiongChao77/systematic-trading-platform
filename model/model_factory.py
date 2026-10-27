import json
from typing import Any, Type, Tuple, Dict

import torch

from model.models.model_base import BaseTimeSeriesModel

# =============================
# Explicitly import all models
# =============================

# ---- generic: single-head, n_classes-configurable (binary OR direct 3-class) ----
from model.models.generic.xgboost_model import XGBoostAdapter
from model.models.generic.lightgbm_model import LightGBMAdapter
from model.models.generic.sklearn_adapter import SVCAdapter, LogisticRegressionSklearnAdapter
from model.models.generic.lstm import LSTM1D_V1
from model.models.generic.lstm_v2 import LSTM1D_V2
from model.models.generic.conv_lstm_v1 import ConvLSTM1D_V1
from model.models.generic.transformer_v1 import Transformer1D_V1
from model.models.generic.logistic_regression import LogisticRegressionTS_V1

# ---- dual_head: fixed trigger(2)+direction(2) heads, fused by an inference wrapper ----
# Dedicated to the "Two-Head Three-Class" branch only; not usable as a plain
# binary or single-head 3-class classifier.
from model.models.dual_head.tcn_v1 import TCN1D_V1
from model.models.dual_head.mamba_v1 import Mamba1D_V1
from model.models.dual_head.lstm_v3 import LSTM1D_V3
from model.models.dual_head.conv_lstm_v3 import ConvLSTM1D_V3
from model.models.dual_head.cnn import CNN1D_V1
from model.models.dual_head.transformer_v3 import Transformer1D_V3

from model.models.fusion_wrapper import FusionWrapper
# Add new model imports here

class ModelFactory:
    """
    Centralized model factory.

    - All available models are explicitly listed here
    - Selection is based on (model_type, model_version)
    - Models must inherit BaseTimeSeriesModel
    """

    model_list = [
        # ---- generic (single-head, n_classes-configurable) ----
        XGBoostAdapter,
        LightGBMAdapter,
        SVCAdapter,
        LogisticRegressionSklearnAdapter,
        LSTM1D_V1,
        LSTM1D_V2,
        ConvLSTM1D_V1,
        Transformer1D_V1,
        LogisticRegressionTS_V1,

        # ---- dual_head (fixed 2+2 trigger/direction heads) ----
        TCN1D_V1,
        Mamba1D_V1,
        LSTM1D_V3,  #good
        ConvLSTM1D_V3,
        CNN1D_V1,
        Transformer1D_V3,
    ]

    # Internal index
    _index: Dict[Tuple[str, int], Type[BaseTimeSeriesModel]] = {}

    # =============================
    # Build index (only once)
    # =============================
    @classmethod
    def _build_index(cls) -> None:
        if cls._index:
            return

        for model_cls in cls.model_list:
            if not issubclass(model_cls, BaseTimeSeriesModel):
                raise TypeError(
                    f"{model_cls.__name__} must inherit BaseTimeSeriesModel"
                )

            key = (model_cls.MODEL_TYPE, model_cls.MODEL_VERSION)

            if key in cls._index:
                raise ValueError(f"Duplicate model registered: {key}")

            cls._index[key] = model_cls

    # =============================
    # Lookup model class
    # =============================
    @classmethod
    def get_model_class(cls, model_type: str, model_version: int) -> Type[BaseTimeSeriesModel]:
        cls._build_index()

        key = (model_type, model_version)
        if key not in cls._index:
            available = sorted(cls._index.keys())
            raise KeyError(
                f"Model not found: {key}. "
                f"Available models: {available}"
            )
        return cls._index[key]

    # =============================
    # Build for training
    # =============================
    @classmethod
    def build_for_training(
        cls,
        model_type: str,
        model_version: int,
        device: torch.device,
        **model_kwargs: Any,
    ) -> BaseTimeSeriesModel:
        model_cls = cls.get_model_class(model_type, model_version)
        if model_cls.TRAINING_BACKEND == "sklearn":
            model = model_cls(device=device, **model_kwargs)
        else:
            model = model_cls(**model_kwargs)
        return model.to(device)

    # =============================
    # Load checkpoint
    # =============================
    @classmethod
    def load_from_checkpoint(
        cls,
        model_path: str,
        meta_path: str,
        device: torch.device,
    ) -> Tuple[BaseTimeSeriesModel, dict]:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model_cls = cls.get_model_class(
            meta["model_type"],
            meta["model_version"],
        )

        model, meta = model_cls.load_checkpoint(
            model_path=model_path,
            meta_path=meta_path,
            device=device,
        )
        return model, meta

    # =============================
    # Debug / visualization
    # =============================
    @classmethod
    def list_models(cls) -> list:
        cls._build_index()
        return sorted(cls._index.keys())
