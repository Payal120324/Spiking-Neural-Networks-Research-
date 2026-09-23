"""
Transformer Baseline Model for Legal Text Classification.
Supports fine-tuning and embedding extraction for:
  - LegalBERT, BERT, RoBERTa, DeBERTa-v3, Sentence-BERT
"""

import logging
import pickle
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class LegalTextDataset(Dataset):
    """PyTorch Dataset wrapping a list of {text, label} dicts."""

    def __init__(self, rows: list[dict], tokenizer, max_length: int = 512):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(
            row["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": row["label"],
        }


class TransformerBaseline:
    """
    Transformer-based legal text classifier.

    Supports two modes:
      - "finetune": full end-to-end fine-tuning with classification head
      - "embedding": extract frozen embeddings → train linear classifier
    """

    def __init__(self, config: dict, model_key: str = "legal_bert"):
        self.config = config
        self.model_key = model_key
        models_cfg = config.get("models", {})
        transformer_cfg = models_cfg.get("transformers", {})
        self.model_name = transformer_cfg.get(model_key, {}).get(
            "name", "nlpaueb/legal-bert-base-uncased"
        )
        training_cfg = models_cfg.get("training", {})
        self.lr = training_cfg.get("learning_rate", 2e-5)
        self.batch_size = training_cfg.get("batch_size", 16)
        self.num_epochs = training_cfg.get("num_epochs", 3)
        self.warmup_ratio = training_cfg.get("warmup_ratio", 0.1)
        self.weight_decay = training_cfg.get("weight_decay", 0.01)
        self.max_length = config.get("preprocessing", {}).get("max_length", 512)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._label_encoder = LabelEncoder()
        self._embedding_cache: dict = {}
        storage = config.get("storage", {})
        self._emb_cache_dir = Path(storage.get("embeddings_cache", "storage/embeddings"))
        self._emb_cache_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def load(self, num_labels: int = 2, mode: str = "embedding"):
        """Load tokenizer and model from HuggingFace (auto-cached)."""
        logger.info(f"Loading tokenizer: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=False,
                trust_remote_code=False
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from remote: {e}, trying local only")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=True
            )
        
        if mode == "finetune":
            logger.info(f"Loading classification model ({num_labels} labels)")
            try:
                # ignore_mismatched_sizes=True allows loading with new classification head
                # This will produce a warning about missing parameters, which is expected during fine-tuning
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, 
                    num_labels=num_labels, 
                    ignore_mismatched_sizes=True,
                    local_files_only=False
                ).to(self.device)
            except Exception as e:
                logger.warning(f"Failed to load model from remote: {e}, trying local only")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, 
                    num_labels=num_labels, 
                    ignore_mismatched_sizes=True,
                    local_files_only=True
                ).to(self.device)
        else:
            logger.info("Loading base model for embedding extraction")
            try:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    local_files_only=False
                ).to(self.device)
            except Exception as e:
                logger.warning(f"Failed to load model from remote: {e}, trying local only")
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    local_files_only=True
                ).to(self.device)
        logger.info(f"Model loaded on {self.device}")

    def get_embeddings(
        self,
        rows: list[dict],
        batch_size: int = 32,
        cache_key: Optional[str] = None,
        layer: str = "cls",
    ) -> np.ndarray:
        """
        Extract [CLS] embeddings for a list of text samples.
        Results are cached to disk to avoid recomputation.

        Args:
            rows: list of {text, label}
            batch_size: inference batch size
            cache_key: optional string key for disk cache
            layer: 'cls' | 'mean' (pooling strategy)

        Returns:
            embeddings: (n_samples, hidden_size)
        """
        if cache_key:
            # Sample count is part of the key so that changing sample caps
            # never reuses embeddings computed for a different subset.
            cache_path = (
                self._emb_cache_dir / f"{self.model_key}_{cache_key}_n{len(rows)}_{layer}.pkl"
            )
            if cache_path.exists():
                logger.info(f"Loading cached embeddings: {cache_path}")
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

        if self.model is None or self.tokenizer is None:
            self.load(mode="embedding")

        self.model.eval()
        texts = [r["text"] for r in rows]
        all_embeddings = []

        progress = tqdm(
            range(0, len(texts), batch_size),
            desc=f"Embedding {cache_key or 'batch'} ({len(texts)} samples)",
            unit="batch",
        )
        with torch.no_grad():
            for i in progress:
                batch_texts = texts[i : i + batch_size]
                enc = self.tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                outputs = self.model(**enc)

                if layer == "cls":
                    # [CLS] token representation
                    emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                else:
                    # Mean pooling over non-padding tokens
                    hidden = outputs.last_hidden_state
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    emb = (hidden * mask).sum(1) / mask.sum(1)
                    emb = emb.cpu().numpy()

                all_embeddings.append(emb)

        embeddings = np.concatenate(all_embeddings, axis=0)

        if cache_key:
            with open(cache_path, "wb") as f:
                pickle.dump(embeddings, f)
            logger.info(f"Saved embeddings → {cache_path}")

        return embeddings

    def train_linear_probe(
        self,
        train_rows: list[dict],
        val_rows: list[dict],
        cache_key: Optional[str] = None,
        val_split: str = "validation",
    ) -> dict:
        """
        Train a logistic regression classifier on frozen embeddings.
        Returns evaluation metrics on the validation set.
        """
        logger.info(f"Extracting train embeddings [{self.model_key}]")
        X_train = self.get_embeddings(
            train_rows, cache_key=f"{cache_key}_train" if cache_key else None
        )
        logger.info(f"Extracting val embeddings [{self.model_key}]")
        X_val = self.get_embeddings(
            val_rows, cache_key=f"{cache_key}_{val_split}" if cache_key else None
        )

        logger.info(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
        y_train = [r["label"] for r in train_rows]
        y_val = [r["label"] for r in val_rows]
        logger.info(f"y_train len: {len(y_train)}, y_val len: {len(y_val)}")

        # Handle multi-label as multi-class via label string
        if isinstance(y_train[0], list):
            y_train = [str(sorted(lbl)) for lbl in y_train]
            y_val = [str(sorted(lbl)) for lbl in y_val]

        self._label_encoder.fit(y_train)
        y_train_enc = self._label_encoder.transform(y_train)
        y_val_enc = self._label_encoder.transform(
            [y if y in self._label_encoder.classes_ else self._label_encoder.classes_[0]
             for y in y_val]
        )

        logger.info("Training logistic regression probe…")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_train_s, y_train_enc)

        from sklearn.metrics import accuracy_score, f1_score
        y_pred = clf.predict(X_val_s)
        metrics = {
            "accuracy": float(accuracy_score(y_val_enc, y_pred)),
            "f1_macro": float(f1_score(y_val_enc, y_pred, average="macro", zero_division=0)),
            "f1_micro": float(f1_score(y_val_enc, y_pred, average="micro", zero_division=0)),
            "n_train": len(y_train),
            "n_val": len(y_val),
            "model": self.model_key,
            "mode": "linear_probe",
        }
        logger.info(
            f"[{self.model_key}] acc={metrics['accuracy']:.4f} "
            f"f1_macro={metrics['f1_macro']:.4f}"
        )
        self._clf = clf
        self._scaler = scaler
        return metrics

    def finetune(
        self,
        train_rows: list[dict],
        val_rows: list[dict],
        num_labels: int = 2,
    ) -> dict:
        """
        Full fine-tuning of the transformer with a classification head.
        Returns per-epoch training history.
        """
        self.load(num_labels=num_labels, mode="finetune")
        train_ds = LegalTextDataset(train_rows, self.tokenizer, self.max_length)
        val_ds = LegalTextDataset(val_rows, self.tokenizer, self.max_length)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        history = []
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].clone().detach().to(self.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            val_metrics = self._evaluate_classification(val_loader)
            epoch_stats = {
                "epoch": epoch + 1,
                "train_loss": total_loss / len(train_loader),
                **val_metrics,
            }
            history.append(epoch_stats)
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} "
                f"loss={epoch_stats['train_loss']:.4f} "
                f"val_acc={epoch_stats.get('accuracy', 0):.4f}"
            )

        return {"history": history, "model": self.model_key, "mode": "finetune"}

    def count_mac_operations(self, seq_length: int = 512) -> int:
        """
        Estimate number of MAC operations for a forward pass.
        Used for energy analysis.
        """
        if self.model is None:
            self.load(mode="embedding")
        cfg = self.model.config
        hidden = cfg.hidden_size
        layers = cfg.num_hidden_layers
        heads = cfg.num_attention_heads
        head_dim = hidden // heads
        ffn = getattr(cfg, "intermediate_size", hidden * 4)

        # Self-attention: Q, K, V projections + attention scores + output
        attn_macs = layers * (
            3 * seq_length * hidden * hidden  # QKV projections
            + seq_length * seq_length * hidden  # attention scores
            + seq_length * hidden * hidden  # output projection
        )
        # FFN
        ffn_macs = layers * (
            seq_length * hidden * ffn  # up
            + seq_length * ffn * hidden  # down
        )
        return int(attn_macs + ffn_macs)

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _evaluate_classification(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                labels = batch["label"]
                if isinstance(labels, torch.Tensor):
                    labels = labels.numpy()
                all_labels.extend(labels)
        from sklearn.metrics import accuracy_score, f1_score
        return {
            "accuracy": float(accuracy_score(all_labels, all_preds)),
            "f1_macro": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        }
