"""
Spiking Neural Network Classifier.
Built with snntorch and PyTorch.
Accepts spike-encoded inputs and produces class predictions.
"""

import logging
from typing import Optional
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import snntorch as snn
    from snntorch import surrogate
    HAS_SNNTORCH = True
except ImportError:
    HAS_SNNTORCH = False
    logger = logging.getLogger(__name__)
    logger.warning("snntorch not available. SNN will use a PyTorch fallback.")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# snntorch SNN model
# ─────────────────────────────────────────────────────────────────────

class SNNTorchModel(nn.Module):
    """
    Fully-connected SNN using Leaky Integrate-and-Fire neurons (snntorch).
    Processes spike trains over `time_steps` and uses rate-coded output.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        beta: float = 0.9,
        threshold: float = 1.0,
        dropout: float = 0.3,
        num_hidden_layers: int = 2,
        use_residual: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_hidden_layers = num_hidden_layers
        self.use_residual = use_residual

        # Build fully-connected layers with batch normalization
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()
        self.residual_fc = nn.ModuleList()  # For residual connections

        in_dim = input_size
        for i in range(num_hidden_layers):
            self.fc_layers.append(nn.Linear(in_dim, hidden_size))
            self.bn_layers.append(nn.BatchNorm1d(hidden_size))
            self.lif_layers.append(
                snn.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate.fast_sigmoid())
            )
            # Add residual connection if dimensions match
            if use_residual and in_dim == hidden_size:
                self.residual_fc.append(nn.Identity())
            elif use_residual:
                self.residual_fc.append(nn.Linear(in_dim, hidden_size))
            else:
                self.residual_fc.append(None)
            in_dim = hidden_size

        self.fc_out = nn.Linear(hidden_size, num_classes)
        self.lif_out = snn.Leaky(
            beta=beta, threshold=threshold, spike_grad=surrogate.fast_sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (time_steps, batch, input_size) spike trains

        Returns:
            spike_out: (time_steps, batch, num_classes)
            mem_out:   (batch, num_classes)  — membrane potential at last step
        """
        time_steps = x.shape[0]

        # Initialize membrane potentials
        mems = [lif.init_leaky() for lif in self.lif_layers]
        mem_out = self.lif_out.init_leaky()

        spike_rec = []
        for t in range(time_steps):
            xt = x[t]  # (batch, input_size)
            spk = xt
            for i, (fc, bn, lif, res_fc) in enumerate(zip(self.fc_layers, self.bn_layers, self.lif_layers, self.residual_fc)):
                cur = fc(self.dropout(spk))
                cur = bn(cur)  # Apply batch normalization
                
                # Add residual connection if enabled (skip connection on currents)
                if self.use_residual and res_fc is not None:
                    residual = res_fc(spk)
                    cur = cur + residual
                
                spk, mems[i] = lif(cur, mems[i])

            cur_out = self.fc_out(spk)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            spike_rec.append(spk_out)

        spike_out = torch.stack(spike_rec, dim=0)  # (T, batch, num_classes)
        return spike_out, mem_out


# ─────────────────────────────────────────────────────────────────────
# Pure-PyTorch fallback (no snntorch required)
# ─────────────────────────────────────────────────────────────────────

class SimpleSNNFallback(nn.Module):
    """
    Simplified SNN fallback without snntorch.
    Integrates spike trains across time, then applies a linear classifier.
    """

    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout: float = 0.3, num_hidden_layers: int = 2, use_residual: bool = True):
        super().__init__()
        layers = []
        in_dim = input_size
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        # x: (time_steps, batch, features) → integrate over time
        rate = x.mean(dim=0)  # (batch, features)
        return self.net(rate), None


# ─────────────────────────────────────────────────────────────────────
# High-level classifier wrapper
# ─────────────────────────────────────────────────────────────────────

class SNNClassifier:
    """
    High-level SNN classifier wrapper.
    Accepts spike trains (numpy arrays) and returns classification metrics.
    """

    def __init__(self, config: dict, encoding_name: str = "poisson_rate"):
        self.config = config
        self.encoding_name = encoding_name
        snn_cfg = config.get("snn", {})
        arch = snn_cfg.get("architecture", {})
        self.hidden_size = arch.get("hidden_size", 256)
        self.num_hidden_layers = arch.get("num_hidden_layers", 2)
        self.dropout = arch.get("dropout", 0.3)
        self.use_residual = arch.get("use_residual", True)  # Enable residual connections by default
        neuron_cfg = snn_cfg.get("neuron", {})
        self.beta = neuron_cfg.get("beta", 0.9)
        self.threshold = neuron_cfg.get("threshold", 1.0)
        train_cfg = snn_cfg.get("training", {})
        self.lr = train_cfg.get("learning_rate", 1e-3)
        self.batch_size = train_cfg.get("batch_size", 32)
        self.num_epochs = train_cfg.get("num_epochs", 10)
        self.time_steps = config.get("encoding", {}).get("time_steps", 50)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.num_classes: int = 2

    def build_model(self, input_size: int, num_classes: int):
        """Instantiate the SNN architecture."""
        self.num_classes = num_classes
        if HAS_SNNTORCH:
            self.model = SNNTorchModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_classes=num_classes,
                beta=self.beta,
                threshold=self.threshold,
                dropout=self.dropout,
                num_hidden_layers=self.num_hidden_layers,
                use_residual=self.use_residual,
            ).to(self.device)
        else:
            logger.warning("Using SimpleSNNFallback (snntorch not installed)")
            self.model = SimpleSNNFallback(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_classes=num_classes,
                dropout=self.dropout,
                num_hidden_layers=self.num_hidden_layers,
            ).to(self.device)

    def train(
        self,
        spike_trains: np.ndarray,
        labels: np.ndarray,
        val_spikes: np.ndarray,
        val_labels: np.ndarray,
    ) -> dict:
        """
        Train the SNN classifier.

        Args:
            spike_trains: (n_train, time_steps, features)
            labels:       (n_train,)
            val_spikes:   (n_val, time_steps, features)
            val_labels:   (n_val,)

        Returns:
            dict with training history and final validation metrics
        """
        n_train, T, features = spike_trains.shape
        num_classes = int(labels.max()) + 1
        self.build_model(features, num_classes)

        # Tensors: transpose to (time_steps, batch, features) for SNN
        X_train = torch.tensor(spike_trains, dtype=torch.float32)  # (N, T, F)
        y_train = torch.tensor(labels, dtype=torch.long)
        X_val = torch.tensor(val_spikes, dtype=torch.float32)
        y_val = torch.tensor(val_labels, dtype=torch.long)

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)

        # [FIX, 2026-09]: class weights computed from the observed training
        # distribution. The CaseHOLD training split is close to balanced
        # (~19-21% per class), so this is a small, defensive correction --
        # NOT the primary fix for the collapse (see mem_out change below) --
        # but it costs nothing and guards against imbalance on other datasets.
        class_counts = torch.bincount(y_train, minlength=num_classes).float()
        class_weights = (class_counts.sum() / (num_classes * class_counts.clamp(min=1))).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        history = []
        best_val_metric = 0.0  # Using macro-F1 instead of accuracy
        best_model_state = None
        patience_counter = 0
        early_stopping_patience = self.config.get("snn", {}).get("training", {}).get("early_stopping_patience", 5)
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss, correct = 0.0, 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)  # (batch, T, F)
                y_batch = y_batch.to(self.device)

                # Transpose: (T, batch, F)
                x_t = X_batch.permute(1, 0, 2)
                spike_out, mem_out = self.model(x_t)

                if HAS_SNNTORCH:
                    # [FIX, 2026-09]: the previous readout used discrete,
                    # heavily-quantised spike-count sums (spike_out.sum),
                    # which produced weak/vanishing gradients under both
                    # sparse and dense encodings alike and caused training
                    # to collapse onto a constant-output solution (verified:
                    # 4 of 5 encodings converged to predicting a single
                    # class regardless of input, on a near-perfectly
                    # balanced training set). mem_out is the continuous
                    # membrane potential of the final LIF layer and is not
                    # subject to this quantisation; we combine it with the
                    # spike-count signal rather than discarding the latter
                    # entirely, since spike counts still encode useful
                    # rate information once gradients are no longer stuck.
                    logits = spike_out.sum(dim=0) + mem_out
                    # Add debugging for first batch of first epoch
                    if epoch == 0 and len(history) == 0:
                        logger.debug(f"First batch logits range: [{logits.min():.4f}, {logits.max():.4f}]")
                        logger.debug(f"First batch logits mean: {logits.mean():.4f}, std: {logits.std():.4f}")
                        logger.debug(f"First batch predictions distribution: {torch.bincount(logits.argmax(1))}")
                else:
                    logits = spike_out

                loss = criterion(logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                correct += (logits.argmax(1) == y_batch).sum().item()

            train_acc = correct / len(train_ds)
            val_metrics = self.evaluate(X_val, y_val)
            val_macro_f1 = val_metrics.get("f1_macro", 0.0)

            # [FIX, 2026-09]: log the FULL validation prediction distribution
            # every epoch (not just the first training batch of epoch 0), so
            # a collapse to a constant prediction is visible immediately in
            # logs rather than requiring forensic analysis of raw results.
            val_pred_dist = torch.bincount(
                torch.as_tensor(val_metrics["predictions"]), minlength=num_classes
            ) if "predictions" in val_metrics else None
            if val_pred_dist is not None and val_pred_dist.max().item() > 0.9 * len(y_val):
                logger.warning(
                    f"[COLLAPSE WARNING] Epoch {epoch+1}: {val_pred_dist.max().item()}/"
                    f"{len(y_val)} validation predictions are the same class "
                    f"({val_pred_dist.argmax().item()}). Model may have collapsed."
                )
            
            # Add diagnostic logging for convergence analysis
            if epoch == 0 or epoch == self.num_epochs - 1:
                logger.info(f"[DIAGNOSTIC] Epoch {epoch+1} - Train acc: {train_acc:.4f}, Val F1: {val_macro_f1:.4f}")
                logger.info(f"[DIAGNOSTIC] Train loss: {total_loss / len(train_loader):.4f}")
            
            # Early stopping logic with best model saving (using macro-F1)
            if val_macro_f1 > best_val_metric:
                best_val_metric = val_macro_f1
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                logger.info(f"New best model saved with val_f1_macro: {val_macro_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            scheduler.step()
            
            epoch_stats = {
                "epoch": epoch + 1,
                "train_loss": total_loss / len(train_loader),
                "train_acc": train_acc,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(epoch_stats)
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} "
                f"loss={epoch_stats['train_loss']:.4f} "
                f"train_acc={train_acc:.4f} "
                f"val_acc={val_metrics.get('accuracy', 0):.4f}"
            )

        # Restore best model before final evaluation
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Restored best model with val_f1_macro: {best_val_metric:.4f}")

        return {
            "history": history,
            "encoding": self.encoding_name,
            "model": "snn",
            "final_val": self.evaluate(X_val, y_val),
            "best_val_metric": best_val_metric,
        }

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        """Run inference and return metrics."""
        self.model.eval()
        all_preds = []
        all_logits = []
        loader = DataLoader(TensorDataset(X, y), batch_size=64)

        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                x_t = X_batch.permute(1, 0, 2)
                spike_out, mem_out = self.model(x_t)
                if HAS_SNNTORCH:
                    # [FIX, 2026-09]: match the training-loop readout
                    # (spike-count + membrane potential) so evaluation is
                    # consistent with what the model was actually optimised
                    # against. Previously this discarded mem_out entirely.
                    logits = spike_out.sum(dim=0) + mem_out
                else:
                    logits = spike_out
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_logits.append(logits.cpu().numpy())

        y_np = y.cpu().numpy()
        preds_np = np.array(all_preds)
        logits_np = np.concatenate(all_logits, axis=0)
        
        # Add diagnostic information
        pred_distribution = Counter(preds_np)
        logger.debug(f"Prediction distribution: {dict(pred_distribution)}")
        logger.debug(f"Logits statistics - mean: {logits_np.mean():.4f}, std: {logits_np.std():.4f}")
        logger.debug(f"Logits range: [{logits_np.min():.4f}, {logits_np.max():.4f}]")
        
        from sklearn.metrics import accuracy_score, f1_score
        return {
            "accuracy": float(accuracy_score(y_np, all_preds)),
            "f1_macro": float(f1_score(y_np, all_preds, average="macro", zero_division=0)),
            "f1_micro": float(f1_score(y_np, all_preds, average="micro", zero_division=0)),
            "predictions": preds_np,  # [FIX, 2026-09]: needed for per-epoch collapse detection
        }


    def count_synaptic_operations(
        self, spike_trains: np.ndarray
    ) -> dict:
        """
        Count Synaptic Operations (SOPs) for energy analysis.
        SOPs = sum of spikes × fan-out (number of downstream connections).
        """
        n, T, features = spike_trains.shape
        snn_cfg = self.config.get("snn", {})
        arch = snn_cfg.get("architecture", {})
        hidden = arch.get("hidden_size", 256)
        n_layers = arch.get("num_hidden_layers", 2)

        total_spikes = float(spike_trains.sum())
        avg_spikes_per_sample = total_spikes / n

        # Layer sizes: input → hidden → ... → hidden → output
        layer_sizes = [features] + [hidden] * n_layers + [self.num_classes]
        total_sops_per_sample = 0
        
        # Compute actual input firing rate (spikes per time step per feature)
        input_firing_rate = avg_spikes_per_sample / (spike_trains.shape[1] * spike_trains.shape[2])
        
        for i in range(len(layer_sizes) - 1):
            # SOPs = spikes_at_layer_i × fan_out
            # For input layer: spikes = avg_spikes_per_sample
            # For hidden layers: estimate based on input firing rate
            spikes_at_layer = avg_spikes_per_sample if i == 0 else avg_spikes_per_sample * input_firing_rate
            total_sops_per_sample += spikes_at_layer * layer_sizes[i + 1]

        return {
            "total_spikes": total_spikes,
            "avg_spikes_per_sample": avg_spikes_per_sample,
            "avg_sops_per_sample": float(total_sops_per_sample),
            "sparsity": float(1.0 - spike_trains.mean()),
        }