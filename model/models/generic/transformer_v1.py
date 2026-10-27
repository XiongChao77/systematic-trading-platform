import math

import torch
import torch.nn as nn

from model.models.model_base import BaseTimeSeriesModel


# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(
            position
            * div_term[:pe[:, 1::2].shape[1]]
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
            persistent=False,
        )

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds "
                f"max_len={self.pe.size(1)}"
            )

        x = x + self.pe[:, :x.size(1)].to(
            dtype=x.dtype
        )

        return self.dropout(x)


# ============================================================
# Local Temporal Block
# ============================================================

class TemporalLocalBlock(nn.Module):
    """
    Capture short-range temporal structure before Transformer.

    Input / output:
        [B, T, D]
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert kernel_size % 2 == 1

        padding = kernel_size // 2

        # depthwise temporal convolution
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,
        )

        # channel mixing
        self.pointwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=1,
        )

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x

        # [B,T,D] -> [B,D,T]
        y = x.transpose(1, 2)

        y = self.depthwise(y)
        y = self.pointwise(y)

        y = self.activation(y)

        # [B,D,T] -> [B,T,D]
        y = y.transpose(1, 2)

        y = self.dropout(y)

        # residual local block
        return self.norm(residual + y)


# ============================================================
# Mean / Last Gated Pooling
# ============================================================

class MeanLastGatedPooling(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()

        self.gate = nn.Linear(
            d_model * 2,
            d_model,
        )

    def forward(self, x):
        """
        x: [B,T,D]
        """

        mean_state = x.mean(dim=1)
        last_state = x[:, -1, :]

        gate = torch.sigmoid(
            self.gate(
                torch.cat(
                    [mean_state, last_state],
                    dim=-1,
                )
            )
        )

        return (
            gate * last_state
            +
            (1.0 - gate) * mean_state
        )


# ============================================================
# Transformer1D_V1
# ============================================================

class Transformer1D_V1(BaseTimeSeriesModel):
    """
    Local-Global Transformer for financial time-series
    classification.

    Architecture:

        Feature Projection
            ↓
        Local Temporal Conv
            ↓
        Sinusoidal Position Encoding
            ↓
        Transformer Encoder
            ↓
        Mean + Last gated pooling
            ↓
        Linear classifier

    Input:
        [B,T,F]

    Output:
        [B,n_classes]
    """

    MODEL_TYPE = "transformer"
    MODEL_VERSION = 1

    supports_lengths = False

    def __init__(
        self,
        input_size: int,
        n_classes: int,

        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,

        dropout: float = 0.1,

        local_kernel_size: int = 3,

        seq_len: int = 5000,

        **kwargs,
    ):
        super().__init__()

        if kwargs:
            print(
                f"[Transformer1D_V3] Ignored kwargs: "
                f"{list(kwargs.keys())}"
            )

        assert d_model % nhead == 0

        # ----------------------------------------------------
        # Meta
        # ----------------------------------------------------

        self.input_size = input_size
        self.n_classes = n_classes

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward

        self.dropout_p = dropout

        self.local_kernel_size = local_kernel_size
        self.seq_len = seq_len

        # ----------------------------------------------------
        # Feature projection
        # ----------------------------------------------------

        self.input_proj = nn.Linear(
            input_size,
            d_model,
        )

        # ----------------------------------------------------
        # Local temporal representation
        # ----------------------------------------------------

        self.local_block = TemporalLocalBlock(
            d_model=d_model,
            kernel_size=local_kernel_size,
            dropout=dropout,
        )

        # ----------------------------------------------------
        # Position
        # ----------------------------------------------------

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=seq_len,
            dropout=dropout,
        )

        # ----------------------------------------------------
        # Transformer
        #
        # Keep V1 behavior intentionally
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,

            dim_feedforward=dim_feedforward,

            dropout=dropout,

            activation="relu",

            batch_first=True,

            # Keep V1 Post-Norm
            norm_first=False,
        )

        self.transformer_encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        )

        # ----------------------------------------------------
        # Readout
        # ----------------------------------------------------

        self.pool = MeanLastGatedPooling(
            d_model
        )

        # ----------------------------------------------------
        # Single classification head
        # ----------------------------------------------------

        self.fc = nn.Linear(
            d_model,
            n_classes,
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(
                f"Expected [B,T,F], "
                f"got {tuple(x.shape)}"
            )

        if x.size(-1) != self.input_size:
            raise ValueError(
                f"Expected input_size={self.input_size}, "
                f"got {x.size(-1)}"
            )

        # Feature projection
        x = self.input_proj(x)

        # Local temporal pattern extraction
        x = self.local_block(x)

        # Position
        x = self.pos_encoder(x)

        # Global temporal interaction
        x = self.transformer_encoder(x)

        # Current + historical state
        x = self.pool(x)

        # Classification
        return self.fc(x)

    # ========================================================
    # Meta
    # ========================================================

    def export_meta(self, **extra):

        return {
            "model_type": self.MODEL_TYPE,
            "model_version": self.MODEL_VERSION,

            "input_size": self.input_size,
            "n_classes": self.n_classes,

            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward":
                self.dim_feedforward,

            "dropout": self.dropout_p,

            "local_kernel_size":
                self.local_kernel_size,

            "seq_len": self.seq_len,

            **extra,
        }

    @classmethod
    def build_from_meta(
        cls,
        meta: dict,
        state: dict,
        device,
    ):

        model = cls(
            input_size=meta["input_size"],

            n_classes=(
                len(meta["classes"])
                if "classes" in meta
                else meta["n_classes"]
            ),

            d_model=meta["d_model"],
            nhead=meta["nhead"],
            num_layers=meta["num_layers"],

            dim_feedforward=
                meta["dim_feedforward"],

            dropout=meta["dropout"],

            local_kernel_size=
                meta.get(
                    "local_kernel_size",
                    3,
                ),

            seq_len=meta["seq_len"],
        )

        model.load_state_dict(
            state["state_dict"]
        )

        return model.to(device)