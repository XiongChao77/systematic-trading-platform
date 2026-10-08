import os, sys, time, json,hashlib
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score,confusion_matrix
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))
# Import project modules
from data_process.common import *
from model.model_factory import ModelFactory
from model.data_loader import TimeSeriesWindowDataset
from model.models.fusion_wrapper import (
    BinaryInferenceWrapper,
    DirectThreeClassInferenceWrapper,
    DualHeadInferenceWrapper,
    FusionWrapper,
)
from model.train_config import *
# -----------------------------------------------------------------------------
# Encapsulated Model Handler
# -----------------------------------------------------------------------------
# model_loader.py

class MetaConfig:
    def _init_config_from_meta(self, meta):
        """
        Extract dataset configuration from a meta dict.
        """
        self.feature_cols = meta["feature_cols"]
        self.seq_len = int(meta["seq_len"])
        # In pipeline mode, final output is typically mapped back to 3 classes.
        # If a sub-model is binary, the wrapper will handle mapping to 3 classes.
        self.classes = meta.get("classes", [0, 1]) 
        self.label_col = meta.get("label_col", "label")
        
        self.raw_config = meta.get("feature_group_list", [])
        self.feature_group_list = []
        for class_name, params in self.raw_config:
            if class_name in globals():
                cls = globals()[class_name] 
                self.feature_group_list.append(FeatureContainer(cls, **params))

TASK_DESCRIPTION_FILENAME = "task_description.json"

class ModelHandler(MetaConfig):
    
    def __init__(self,tarin_out_path , device, task_desc_path = None):
        self.device = device
        self.logger = logging.getLogger("trade")
        
        # 1. Read task index
        if task_desc_path is None:
            task_desc_path = os.path.join(tarin_out_path, TASK_DESCRIPTION_FILENAME)
            
        if not os.path.exists(task_desc_path):
            raise FileNotFoundError(f"Task Description not found: {task_desc_path}")
        
            
        with open(task_desc_path, "r", encoding="utf-8") as f:
            self.task_desc = json.load(f)
            
        self.task_type = self.task_desc.get("task_type", "single")
        self.base_dir = os.path.dirname(task_desc_path)
        self.sub_model_conf :dict[str,MetaConfig]= {}
        # For composite tasks (TRIGGER_DIRECTION/LONG_SHORT_OVR): the nn.Module of every sub-model and its
        # task_type (e.g. TRIGGER/DIRECTION), used by evaluate_sub_models() for separate evaluation.
        # Always an empty dict for non composite tasks.
        self.loaded_sub_models: dict = {}
        self.sub_model_task_type: dict = {}

        # 2. Initialize by task type
        self.logger.info(f"🚀 Loading Task: {self.task_type.upper()} from {task_desc_path}")

        if self.task_type in TrainTask.COMBO_TASKS:
            # produced by save_fusion_run: "models": {role: relative_dir_to_a_save_single_run_output}
            self._load_fusion_mode()
        elif self.task_type in TrainTask.BINARY_TASKS:
            # produced by save_single_run for a lone TRIGGER/DIRECTION/LONG_OVR/SHORT_OVR model
            self._load_binary_mode()
        elif self.task_type in (TrainTask.DIRECT_3CLASS, TrainTask.DUAL_HEAD_3CLASS):
            # produced by save_single_run for a 3-class model
            self._load_3class_mode()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def _load_3class_mode(self):
        files = self.task_desc["models"]["main"]
        meta_path = os.path.join(self.base_dir, files["meta"])
        model_path = os.path.join(self.base_dir, files["model"])
        
        # 1. Read meta and initialize configuration
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._init_config_from_meta(meta)
        self.classes = meta["classes"]  # In single mode, use classes from meta directly

        # 2. Load the raw training model, then adapt it to the common inference
        # contract: (three_class_logits, three_class_probabilities).
        raw_model, _ = ModelFactory.load_from_checkpoint(
            model_path=model_path,
            meta_path=meta_path,
            device=self.device
        )
        raw_model.eval()
        if self.task_type == TrainTask.DIRECT_3CLASS:
            self.model = DirectThreeClassInferenceWrapper(raw_model)
        else:
            self.model = DualHeadInferenceWrapper(raw_model)
        self.model.to(self.device)
        self.model.eval()

    def _load_binary_mode(self):
        """
        Load a lone TRIGGER/DIRECTION/LONG_OVR/SHORT_OVR model.

        Direction and OvR binary models are adapted to the common inference
        contract: (three_class_logits, three_class_probabilities). A lone
        TRIGGER model still cannot be a tradable signal because it has no
        direction; BinaryInferenceWrapper raises a clear error if it is used
        for backtesting.
        """
        files = self.task_desc["models"]["main"]
        meta_path = os.path.join(self.base_dir, files["meta"])
        model_path = os.path.join(self.base_dir, files["model"])

        # 1. Read meta and initialize configuration
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._init_config_from_meta(meta)
        self.classes = meta["classes"]  # In single mode, use classes from meta directly

        # 2. Load the raw binary model and adapt it to the shared 3-class
        # inference contract expected by predict()/predict_with_ds().
        raw_model, _ = ModelFactory.load_from_checkpoint(
            model_path=model_path,
            meta_path=meta_path,
            device=self.device
        )
        raw_model.eval()
        self.model = BinaryInferenceWrapper(raw_model, task_type=self.task_type)
        self.model.to(self.device)
        self.model.eval()

    def _load_fusion_mode(self):
        sub_models_map = self.task_desc["models"]
        loaded_sub_models = {}
        shared_input_signature = None
        
        # 2. Load all sub-models
        for name, path in sub_models_map.items():
            task_desc_dir = os.path.join(self.base_dir, path)
            with open(os.path.join(task_desc_dir, TASK_DESCRIPTION_FILENAME), "r", encoding="utf-8") as f:
                task_desc = json.load(f)
            models = task_desc["models"]

            sub_model_meta_path = os.path.join(task_desc_dir, models["main"]["meta"])
            sub_model_model_path = os.path.join(task_desc_dir, models["main"]["model"])
            with open(sub_model_meta_path, "r", encoding="utf-8") as f:
                sub_model_meta = json.load(f)
                self.sub_model_conf[name] = MetaConfig()
                self.sub_model_conf[name]._init_config_from_meta(sub_model_meta)
                conf = self.sub_model_conf[name]
                input_signature = (
                    tuple(conf.feature_cols),
                    conf.seq_len,
                    conf.label_col,
                )
                if shared_input_signature is None:
                    shared_input_signature = input_signature
                    self._init_config_from_meta(sub_model_meta)
                elif input_signature != shared_input_signature:
                    raise ValueError(
                        "Fusion sub-model input schemas differ; a shared input "
                        "tensor cannot be used safely. "
                        f"model={name}, expected={shared_input_signature}, "
                        f"actual={input_signature}"
                    )
            self.sub_model_task_type[name] = task_desc.get("task_type")

            self.logger.info(f"   🔄 Loading sub-model '{name}'...")
            model, _ = ModelFactory.load_from_checkpoint(
                model_path=sub_model_model_path,
                meta_path=sub_model_meta_path,
                device=self.device
            )
            model.eval()
            loaded_sub_models[name] = model

        # 3. Assemble wrapper
        self.model = FusionWrapper(loaded_sub_models, task_type=self.task_type)
        self.model.to(self.device)
        self.model.eval()
        self.loaded_sub_models = loaded_sub_models

    def evaluate_sub_models(self, df, kline_interval_ms, batch_size=2048):
        """
        Composite tasks (TRIGGER_DIRECTION/LONG_SHORT_OVR) only: on the given df (usually the backtest range),
        run every sub-model separately and compute its metrics, instead of only looking at the fused 3-class result.

        The label mapping reuses model.tasks.strategies.BinaryTask.dataset_split, so it matches the binary task
        definitions used during training (trigger: is there an action; direction: long/short; long_ovr/short_ovr: OvR)
        exactly, instead of inventing a mapping on the spot.

        Returns {} for non composite tasks (self.loaded_sub_models empty).
        """
        if not self.loaded_sub_models:
            return {}

        from model.tasks.strategies import BinaryTask
        from model.train_config import TrainConfig
        from model.training_types import TensorSplit

        results = {}
        for name, sub_model in self.loaded_sub_models.items():
            conf = self.sub_model_conf.get(name)
            sub_task_type = self.sub_model_task_type.get(name)
            if conf is None or sub_task_type is None:
                self.logger.warning(f"⚠️ evaluate_sub_models: missing conf/task_type for sub-model '{name}', skip")
                continue

            ds = TimeSeriesWindowDataset(
                df=df,
                kline_interval_ms=kline_interval_ms,
                feature_cols=conf.feature_cols,
                label_col=conf.label_col,
                seq_len=conf.seq_len,
                is_live=False,
            )
            if len(ds) == 0:
                self.logger.warning(f"⚠️ evaluate_sub_models: no valid windows for sub-model '{name}', skip")
                results[name] = {}
                continue

            transformed = BinaryTask(sub_task_type, TrainConfig(), self.device).dataset_split(
                TensorSplit(ds.X, ds.y, ds.returns)
            )
            X_t, y_t = transformed.X, transformed.y
            if len(y_t) == 0:
                results[name] = {}
                continue

            sub_model.eval()
            preds_list = []
            with torch.no_grad():
                for start in range(0, len(X_t), batch_size):
                    xb = X_t[start:start + batch_size].to(self.device)
                    logits = sub_model(xb)
                    preds_list.append(torch.argmax(logits, dim=1).cpu().numpy())
            y_pred = np.concatenate(preds_list)
            y_true = y_t.cpu().numpy()

            stats = self.evaluate_performance(y_true, y_pred)
            # "signal" (short/long win rate etc.) is written against the 3-class Short/Neutral/Long
            # Signal semantics, while a sub-model here has its own binary labels (0/1). The semantics
            # do not line up and showing them would mislead, so only the generic classification metrics are kept.
            stats.pop("signal", None)

            results[name] = {
                "task_type": sub_task_type,
                **stats,
            }

        return results

    def predict(self, df, kline_interval_ms, is_live=True, batch_size=2048, diff_thresh=None, min_thresh=0.3, stride =1,
                   cache_path = '', use_cache= False, show_feature_distribution = True):
        """
        Run inference with optional strategy enhancement based on probability differences.
        
        :param df: input DataFrame (raw features included)
        :param is_live: whether running in live mode.
                        - True (live): optimize memory; only output the latest bar's signal
                        - False (backtest): keep index mapping; align signals strictly with the time axis
        :param batch_size: batch size
        :param diff_thresh: probability difference threshold (P_long - P_short)
        :param min_thresh: minimum probability gate
        :return: (df_out, stats) 
                 df_out includes full bars and columns like 'pred', 'pred_prob', 'net_score', etc.
        """
        self.logger.debug(f"Starting inference pipeline (Mode={'Live' if is_live else 'Backtest'}, diff_thresh={diff_thresh})...")
        
        # 1. Prepare dataset: pass is_live to control index recording behavior
        ds = TimeSeriesWindowDataset(
            df=df, 
            kline_interval_ms = kline_interval_ms,
            feature_cols=self.feature_cols, 
            label_col=self.label_col,
            seq_len=self.seq_len,
            is_live=is_live,
            stride= stride,
            cache_path = cache_path,
            use_cache = use_cache,
            show_feature_distribution = show_feature_distribution
        )
        
        # Check whether any valid windows were generated (data too short or discontinuous windows may be dropped)
        if len(ds) == 0:
            self.logger.warning("No valid windows generated after continuity check!")
            df_empty = df.copy()
            for c in ['pred', 'pred_prob', 'prob_short', 'prob_neutral', 'prob_long', 'net_score']:
                df_empty[c] = np.nan
            return df_empty, {}

        self.logger.info(f"Dataset created. Valid windows: {len(ds)}")
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # 2. Inference loop (logits -> probabilities)
        probs_list = []
        with torch.no_grad():
            for xb, _, _ in dl:
                xb = xb.to(self.device)
                _, fused_probs = self.model(xb)
                
                # Convert to numpy for downstream processing
                probs_list.append(fused_probs.cpu().numpy())

        # Concatenate all batches
        probs_all = np.concatenate(probs_list)
        p_short = probs_all[:, 0]    # Down probability
        p_neutral = probs_all[:, 1]  # Neutral/range probability
        p_long = probs_all[:, 2]     # Up probability
        net_score = p_long - p_short # Net score

        # 3. Final signal logic
        if diff_thresh is not None:
            final_pred = np.full(len(probs_all), int(Signal.NEUTRAL))
            final_conf = np.zeros(len(probs_all))
            
            # Long logic
            mask_long = (net_score > diff_thresh) & (p_long > min_thresh)
            final_pred[mask_long] = int(Signal.POSITIVE )
            final_conf[mask_long] = net_score[mask_long]
            
            # Short logic
            mask_short = (net_score < -diff_thresh) & (p_short > min_thresh)
            final_pred[mask_short] = int(Signal.NEGATIVE)
            final_conf[mask_short] = -net_score[mask_short]
        else:
            final_pred = probs_all.argmax(axis=1)
            final_conf = probs_all.max(axis=1)

        # 4. Core fix: precise alignment and backfill
        # Create a copy and initialize columns with NaN so discontinuity "gaps" are preserved for position management
        df_out = df.copy()
        cols_to_init = ['pred', 'pred_prob', 'prob_short', 'prob_neutral', 'prob_long', 'net_score']
        for c in cols_to_init:
            df_out[c] = np.nan
        
        if not is_live:
            # === Backtest mode: pin signals to the correct original timestamps using ds.indices ===
            if ds.indices is not None:
                # Ensure index and prediction lengths align
                valid_len = min(len(ds.indices), len(final_pred))
                active_indices = ds.indices[:valid_len]
                
                df_out.loc[active_indices, 'pred'] = final_pred[:valid_len]
                df_out.loc[active_indices, 'pred_prob'] = final_conf[:valid_len]
                df_out.loc[active_indices, 'prob_short'] = p_short[:valid_len]
                df_out.loc[active_indices, 'prob_neutral'] = p_neutral[:valid_len]
                df_out.loc[active_indices, 'prob_long'] = p_long[:valid_len]
                df_out.loc[active_indices, 'net_score'] = net_score[:valid_len]
        else:
            # === Live mode: only fill the latest bar ===
            if len(final_pred) > 0:
                last_idx = df.index[-1]
                df_out.at[last_idx, 'pred'] = final_pred[-1]
                df_out.at[last_idx, 'pred_prob'] = final_conf[-1]
                df_out.at[last_idx, 'net_score'] = net_score[-1]

        # 5. Compute evaluation metrics (only when not live and label column exists)
        stats = {}
        if not is_live and self.label_col in df_out.columns:
            # Only evaluate rows that have both predictions and labels
            df_valid = df_out.dropna(subset=['pred', self.label_col])
            if not df_valid.empty:
                y_true = df_valid[self.label_col].values.astype(int)
                y_pred = df_valid['pred'].values.astype(int)
                stats = self.evaluate_performance(y_true, y_pred)
                self._log_performance_summary(stats)
        stats['feature_config'] = self.raw_config
        stats['feature_cols']   = self.feature_cols
        self.logger.info(f"Inference complete. Valid signals: {len(final_pred)}")
        return df_out, stats
    
    def predict_with_ds(self, ds, df, is_live=True, batch_size=512, diff_thresh=None, min_thresh=0.3):
        self.logger.info(f"Starting inference pipeline (Mode={'Live' if is_live else 'Backtest'}, diff_thresh={diff_thresh})...")
        self._validate_inference_dataset(ds)
        # Check whether any valid windows were generated (may be dropped if too short or discontinuous)
        if len(ds) == 0:
            self.logger.warning("No valid windows generated after continuity check!")
            df_empty = df.copy()
            for c in ['pred', 'pred_prob', 'prob_short', 'prob_neutral', 'prob_long', 'net_score']:
                df_empty[c] = np.nan
            return df_empty, {}

        self.logger.info(f"Dataset created. Valid windows: {len(ds)}")
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # 2. Inference loop (logits -> probabilities)
        probs_list = []
        with torch.no_grad():
            for xb, _, _ in dl:
                xb = xb.to(self.device)
                _, fused_probs = self.model(xb)
                
                # Convert to numpy for downstream processing
                probs_list.append(fused_probs.cpu().numpy())

        # Concatenate all batches
        probs_all = np.concatenate(probs_list)
        p_short = probs_all[:, 0]    # Down probability
        p_neutral = probs_all[:, 1]  # Neutral/range probability
        p_long = probs_all[:, 2]     # Up probability
        net_score = p_long - p_short # Net score

        # 3. Final signal logic
        if diff_thresh is not None:
            final_pred = np.full(len(probs_all), int(Signal.NEUTRAL))
            final_conf = np.zeros(len(probs_all))
            
            # Long logic
            mask_long = (net_score > diff_thresh) & (p_long > min_thresh)
            final_pred[mask_long] = int(Signal.POSITIVE )
            final_conf[mask_long] = net_score[mask_long]
            
            # Short logic
            mask_short = (net_score < -diff_thresh) & (p_short > min_thresh)
            final_pred[mask_short] = int(Signal.NEGATIVE)
            final_conf[mask_short] = -net_score[mask_short]
        else:
            final_pred = probs_all.argmax(axis=1)
            final_conf = probs_all.max(axis=1)

        # 4. Core fix: precise alignment and backfill
        # Create a copy and initialize columns with NaN so discontinuity "gaps" are preserved for position management
        df_out = df.copy()
        cols_to_init = ['pred', 'pred_prob', 'prob_short', 'prob_neutral', 'prob_long', 'net_score']
        for c in cols_to_init:
            df_out[c] = np.nan
        
        if not is_live:
            # === Backtest mode: pin signals to correct original timestamps using ds.indices ===
            if ds.indices is not None:
                # Ensure index and prediction lengths align
                valid_len = min(len(ds.indices), len(final_pred))
                active_indices = ds.indices[:valid_len]
                
                df_out.loc[active_indices, 'pred'] = final_pred[:valid_len]
                df_out.loc[active_indices, 'pred_prob'] = final_conf[:valid_len]
                df_out.loc[active_indices, 'prob_short'] = p_short[:valid_len]
                df_out.loc[active_indices, 'prob_neutral'] = p_neutral[:valid_len]
                df_out.loc[active_indices, 'prob_long'] = p_long[:valid_len]
                df_out.loc[active_indices, 'net_score'] = net_score[:valid_len]
        else:
            # === Live mode: only fill the latest bar ===
            if len(final_pred) > 0:
                last_idx = df.index[-1]
                df_out.at[last_idx, 'pred'] = final_pred[-1]
                df_out.at[last_idx, 'pred_prob'] = final_conf[-1]
                df_out.at[last_idx, 'net_score'] = net_score[-1]

        # 5. Compute evaluation metrics (only when not live and label column exists)
        stats = {}
        if not is_live and self.label_col in df_out.columns:
            # Only evaluate rows that have both predictions and labels
            df_valid = df_out.dropna(subset=['pred', self.label_col])
            if not df_valid.empty:
                y_true = df_valid[self.label_col].values.astype(int)
                y_pred = df_valid['pred'].values.astype(int)
                stats = self.evaluate_performance(y_true, y_pred)
                self.logger.info("\n" + classification_report(y_true, y_pred, digits=4, zero_division=0))
                self._log_performance_summary(stats)
                self.logger.info(f"Label distribution (true): {stats.get('label_proportions_true', {})}")
        stats['feature_config'] = self.raw_config
        stats['feature_cols']   = self.feature_cols
        self.logger.info(f"Inference complete. Valid signals: {len(final_pred)}")
        return df_out, stats

    def _validate_inference_dataset(self, ds) -> None:
        """Require exact feature identity/order before accepting a prebuilt dataset."""
        expected_features = list(self.feature_cols)
        actual_features = getattr(ds, "feature_names", None)
        if actual_features is None:
            raise ValueError(
                "Inference dataset does not expose feature_names; exact model "
                "input compatibility cannot be verified"
            )
        actual_features = list(actual_features)
        if actual_features != expected_features:
            missing = [name for name in expected_features if name not in actual_features]
            unexpected = [name for name in actual_features if name not in expected_features]
            raise ValueError(
                "Inference dataset feature schema mismatch; refusing to infer. "
                f"missing={missing}, unexpected={unexpected}, "
                f"expected_order={expected_features}, actual_order={actual_features}"
            )

        inputs = getattr(ds, "X", None)
        if inputs is None or inputs.ndim != 3:
            shape = None if inputs is None else tuple(inputs.shape)
            raise ValueError(
                f"Inference dataset must expose X with shape [N, T, F], got {shape}"
            )
        if inputs.shape[1] != self.seq_len or inputs.shape[2] != len(expected_features):
            raise ValueError(
                "Inference dataset tensor shape is incompatible with model metadata: "
                f"expected [N, {self.seq_len}, {len(expected_features)}], "
                f"got {tuple(inputs.shape)}"
            )
    
    def scan_thresholds(self, df, kline_interval_ms, thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3], batch_size=1024):
        """
        Scan multiple thresholds in one run to compare model performance.
        
        Idea:
        1. Run inference once to get raw probabilities.
        2. Apply different diff_thresh logic in memory.
        3. Compute per-threshold signal density, precision, and recall.
        """
        self.logger.info(f"🔍 Scanning thresholds: {thresholds}...")
        
        # 1. Prepare data & run inference (reuse dataset logic)
        ds = TimeSeriesWindowDataset(
            df=df, 
            kline_interval_ms = kline_interval_ms,
            feature_config_list = self.feature_group_list,
            feature_cols=self.feature_cols, 
            label_col=self.label_col,
            seq_len=self.seq_len,
            is_live = False,
        )
        # shuffle=False to keep order consistent
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # 2. Get raw probabilities
        probs_list = []
        self.model.eval()
        with torch.no_grad():
            for xb, _ in dl:
                xb = xb.to(self.device)
                # Common inference interface returns a [B, 3] probability distribution.
                _, fused_probs = self.model(xb)
                probs_list.append(fused_probs.cpu().numpy())
        
        if not probs_list:
            self.logger.warning("No predictions generated!")
            return pd.DataFrame()

        # Concatenate all batches: [N, 3]
        probs_all = np.concatenate(probs_list)
        
        # 3. Align true labels (y_true)
        # TimeSeriesWindowDataset outputs start from seq_len-1
        valid_idx = df.index[self.seq_len-1:]
        
        # Truncate to avoid length mismatch
        min_len = min(len(valid_idx), len(probs_all))
        valid_idx = valid_idx[:min_len]
        probs_all = probs_all[:min_len]
        
        # Check label existence for evaluation
        if self.label_col not in df.columns:
            self.logger.warning("No label column found in df, cannot evaluate performance.")
            return pd.DataFrame()
            
        y_true = df.loc[valid_idx, self.label_col].values.astype(int)

        # 4. Precompute net score
        # Assume 0:Short, 1:Neutral, 2:Long
        p_short = probs_all[:, 0]
        p_long = probs_all[:, 2]
        net_score = p_long - p_short  # range [-1, 1]

        # 5. Evaluate each threshold
        results = []
        
        for th in thresholds:
            # Initialize predictions to 1 (neutral/range)
            preds = np.full(len(y_true), int(Signal.NEUTRAL))
            
            # Apply threshold logic
            preds[net_score > th] = int(Signal.POSITIVE )
            preds[net_score < -th] = int(Signal.NEGATIVE)
            
            # Compute metrics
            # output_dict=True returns a dict for easy extraction
            report = classification_report(y_true, preds, output_dict=True, zero_division=0)
            
            # Count signals (non-neutral)
            n_short = np.sum(preds == int(Signal.NEGATIVE))
            n_long = np.sum(preds == int(Signal.POSITIVE ))
            total_signals = n_short + n_long
            coverage = total_signals / len(y_true)
            
            # Extract key metrics
            res_row = {
                "Threshold": th,
                "Signals": total_signals,   # Signal count
                "Coverage": coverage,       # Coverage (signal frequency)
                
                # Precision: how many taken signals are correct?
                "Prec_Short": report[str(int(Signal.NEGATIVE))]['precision'],
                "Prec_Long": report[str(int(Signal.POSITIVE ))]['precision'],
                
                # Recall: how many opportunities were captured?
                "Rec_Short": report[str(int(Signal.NEGATIVE))]['recall'],
                "Rec_Long": report[str(int(Signal.POSITIVE ))]['recall'],
                
                # Macro F1
                "Macro_F1": report['macro avg']['f1-score']
            }
            results.append(res_row)

        # 6. Build summary DataFrame
        df_res = pd.DataFrame(results)
        
        # Print ASCII table
        print("\n" + "="*80)
        print("📊 Threshold Scan Report (search best diff threshold)")
        print("="*80)
        # Formatted printing
        print(df_res.to_string(formatters={
            'Threshold': '{:.2f}'.format,
            'Coverage': '{:.2%}'.format,
            'Prec_Short': '{:.2%}'.format,
            'Prec_Long': '{:.2%}'.format,
            'Rec_Short': '{:.2%}'.format,
            'Rec_Long': '{:.2%}'.format,
            'Macro_F1': '{:.4f}'.format
        }))
        print("="*80 + "\n")
        
        return df_res

    def _log_performance_summary(self, stats: dict) -> None:
        """Log the headline metrics using the same keys stored in backtest JSON."""
        self.logger.info(
            "Backtest model metrics | Accuracy: %.4f | Macro-F1: %.4f | Weighted-F1: %.4f",
            stats["accuracy"],
            stats["f1_macro"],
            stats["f1_weighted"],
        )

    def evaluate_performance(self, y_true, y_pred):
        """
        Returned stats are guaranteed to be directly serializable by json.dumps.
        """
        # Ensure numpy arrays for unified processing
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        stats = {}

        # ===== Basic metrics =====
        stats["accuracy"] = float(accuracy_score(y_true, y_pred))
        stats["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        stats["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        stats["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        stats["recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

        # ===== Classification report (dict) =====
        stats["classification_report"] = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        # sklearn keys are typically strings ('0','1','2' / 'accuracy' / 'macro avg'); we will json_safe at the end

        # ===== Confusion matrix =====
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        stats["confusion_matrix"] = {
            "labels": [int(x) for x in labels],     # Force Python int
            "matrix": cm.tolist(),                  # list[list[int]]
        }

        # ===== Precision lift vs base rate =====
        precision_lift = {}

        n_total_true = len(y_true)

        for cls in labels:
            cls_int = int(cls)

            true_count = int(np.sum(y_true == cls))
            base_rate = float(true_count / n_total_true) if n_total_true > 0 else 0.0

            cls_key = str(cls_int)
            cls_report = stats["classification_report"].get(cls_key, {})
            precision = float(cls_report.get("precision", 0.0))

            lift = precision / base_rate if base_rate > 0 else None

            precision_lift[cls_int] = {
                "precision": precision,
                "base_rate": base_rate,
                "lift": lift,
                "lift_pct": float((lift - 1.0) * 100.0) if lift is not None else None,
            }

        stats["precision_lift"] = precision_lift
        
        # ===== Distribution info =====
        unique_t, cnt_t = np.unique(y_true, return_counts=True)
        unique_p, cnt_p = np.unique(y_pred, return_counts=True)

        stats["label_distribution_true"] = {int(k): int(v) for k, v in zip(unique_t, cnt_t)}
        stats["label_distribution_pred"] = {int(k): int(v) for k, v in zip(unique_p, cnt_p)}

        # 2. Core change: add the class proportions
        n_total_true = len(y_true)
        if n_total_true > 0:
            stats["label_proportions_true"] = {
                int(k): float(v / n_total_true) for k, v in zip(unique_t, cnt_t)
            }
        else:
            stats["label_proportions_true"] = {}
        # ===== Signal metrics (based on project's Signal definition) =====
        NEUTRAL = int(Signal.NEUTRAL)
        NEG = int(Signal.NEGATIVE)
        POS = int(Signal.POSITIVE)

        mask_signal = (y_pred != NEUTRAL)
        n_total = int(len(y_pred))
        n_signal = int(mask_signal.sum())

        signal = {
            "total_samples": n_total,
            "signal_count": n_signal,
            "coverage": float(n_signal / n_total) if n_total > 0 else 0.0,
        }

        if n_signal > 0:
            y_true_sig = y_true[mask_signal]
            y_pred_sig = y_pred[mask_signal]
            signal["directional_accuracy"] = float(np.mean(y_true_sig == y_pred_sig))

            # Per-direction stats (based on predicted trigger)
            for name, cls in [("short", NEG), ("long", POS)]:
                m = (y_pred == cls)
                cnt = int(m.sum())
                signal[f"{name}_count"] = cnt
                signal[f"{name}_win_rate"] = float(np.mean(y_true[m] == cls)) if cnt > 0 else None

        stats["signal"] = signal

        # ===== Final step: force JSON-safe =====
        stats = json_safe(stats)

        return stats

    def construct_dataset(self, df, kline_interval_ms, is_live=True):
        """
        Construct dataset using the primary sub-model's configuration.
        This is useful for external modules that want to reuse the same dataset logic.
        """ 
        # This line replaces your original ds = [None, None] plus the loop
        ds = {}
        for key, conf in self.sub_model_conf.items():
            self.logger.info(f"Constructing dataset using sub-model '{key}' configuration...")
            hash_value  = self.generate_config_hash(conf.feature_cols, conf.seq_len)
            self.logger.info(f"Dataset Config Hash (SHA256): {hash_value}")
            if hash_value not in ds:
                ds[hash_value] = TimeSeriesWindowDataset(
                    df=df,
                    kline_interval_ms = kline_interval_ms,
                    feature_cols=conf.feature_cols,
                    label_col=conf.label_col,
                    seq_len=conf.seq_len,
                    is_live=is_live,
                )
        return ds

    def generate_config_hash(feature_cols: list, seq_len: int):
        """
        Combine the feature list and the window size into a unique configuration hash
        """
        # 1. Build a dict or tuple carrying every piece of core information
        # Make sure the feature list is ordered (if the model input depends on the order)
        config_dict = {
            "features": feature_cols,
            "seq_len": seq_len
        }
        
        # 2. Serialize into a stable JSON string
        # sort_keys=True is essential, so the dict key order cannot change the result
        config_str = json.dumps(config_dict, sort_keys=True).encode('utf-8')
        
        # 3. Compute the SHA256 (safer than MD5, more collision resistant)
        return hashlib.sha256(config_str).hexdigest()
