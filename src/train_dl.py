"""
GeoMind AI - Deep Learning Pipeline (PyTorch LSTM & GRU)
Author: Applied Scientist Candidate
Role Target: Amazon Applied Scientist I Intern

Implements:
1. Vectorized sequence generation from historical observations.
2. Target normalization & inverse scaling for numerical stability.
3. PyTorch Dataset & DataLoader utilities.
4. Deep Sequential Architectures: LSTM and GRU with dropout and multi-layer heads.
5. Training loop with early stopping, learning rate scheduling, and checkpointing.
6. Controlled context length experiments (L=6, L=12, L=24).
7. Experiment registration into 'experiments/results.csv'.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.logger import logger
from src.exception import CustomException
from src.data_preprocessing import prepare_datasets
from src.evaluate import compute_metrics, log_experiment


class TrafficSequenceDataset(Dataset):
    """PyTorch Dataset for multivariate sequential time-series forecasting."""
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs sliding window sequences of shape (N - seq_len, seq_len, num_features).
    At step i, sequence is X[i : i + seq_len], target is y[i + seq_len - 1].
    """
    n = len(X)
    if n <= seq_len:
        raise ValueError(f"Dataset length ({n}) must be greater than sequence length ({seq_len})")
        
    num_samples = n - seq_len
    X_seq = np.empty((num_samples, seq_len, X.shape[1]), dtype=np.float32)
    y_seq = np.empty((num_samples,), dtype=np.float32)
    
    for i in range(num_samples):
        X_seq[i] = X[i : i + seq_len]
        y_seq[i] = y[i + seq_len]
        
    return X_seq, y_seq


class TrafficLSTM(nn.Module):
    """Deep 2-Layer LSTM with Dropout and Regression Head."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (hn, cn) = self.lstm(x)
        # Take representation from last time step in sequence
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)


class TrafficGRU(nn.Module):
    """Gated Recurrent Unit (GRU) with fewer gating parameters for efficient temporal modeling."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, hn = self.gru(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)


class DeepLearningTrainer:
    """Orchestrates sequence modeling, early stopping, and checkpoint saving."""
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        seq_len: int,
        device: str = "cpu",
        learning_rate: float = 1e-3,
        batch_size: int = 128
    ):
        self.model = model.to(device)
        self.model_name = model_name
        self.seq_len = seq_len
        self.device = device
        self.batch_size = batch_size
        self.criterion = nn.HuberLoss() # Robust to extreme residual outliers
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)
        self.target_scaler = StandardScaler()

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 15,
        patience: int = 4,
        save_path: Path = None
    ) -> float:
        """Executes training loop with early stopping."""
        best_val_loss = float("inf")
        patience_counter = 0
        
        start_time = time.time()
        for epoch in range(1, max_epochs + 1):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item() * len(batch_x)
                
            train_loss /= len(train_loader.dataset)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    preds = self.model(batch_x)
                    loss = self.criterion(preds, batch_y)
                    val_loss += loss.item() * len(batch_x)
            val_loss /= len(val_loader.dataset)
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "model_state_dict": self.model.state_dict(),
                        "seq_len": self.seq_len,
                        "model_class": self.model.__class__.__name__,
                        "target_scaler_mean": self.target_scaler.mean_[0],
                        "target_scaler_scale": self.target_scaler.scale_[0]
                    }, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at Epoch {epoch} (Best Val Loss: {best_val_loss:.4f})")
                    break
                    
        total_time = time.time() - start_time
        return total_time

    def predict(self, loader: DataLoader) -> np.ndarray:
        """Inference on DataLoader, inverting target scaling to unscaled traffic counts."""
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                preds = self.model(batch_x).cpu().numpy()
                all_preds.append(preds)
                
        preds_scaled = np.vstack(all_preds)
        # Invert target scaling: unscaled = scaled * scale + mean
        preds_unscaled = self.target_scaler.inverse_transform(preds_scaled).flatten()
        return preds_unscaled


def run_dl_experiments() -> pd.DataFrame:
    """
    Executes controlled context-length experiments (L=6, 12, 24) and architecture comparisons (LSTM vs GRU).
    """
    try:
        logger.info("Initializing Deep Learning Experimentation Suite...")
        X_train, y_train, X_val, y_val, X_test, y_test, feat_names = prepare_datasets()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using compute device: {device.upper()}")
        
        # Scale targets using training distribution
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        # Define controlled research experiments
        experiments = [
            {"name": "LSTM (Context L=6)", "arch": "LSTM", "seq_len": 6, "hidden_dim": 64},
            {"name": "LSTM (Context L=12)", "arch": "LSTM", "seq_len": 12, "hidden_dim": 64},
            {"name": "LSTM (Context L=24)", "arch": "LSTM", "seq_len": 24, "hidden_dim": 64},
            {"name": "GRU (Context L=12)", "arch": "GRU", "seq_len": 12, "hidden_dim": 64},
        ]
        
        dl_results = []
        save_dir = Path("models/dl")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for exp in experiments:
            name = exp["name"]
            seq_len = exp["seq_len"]
            arch = exp["arch"]
            hidden_dim = exp["hidden_dim"]
            
            logger.info(f"=== Running Experiment: {name} (Arch: {arch}, Lookback: {seq_len} hrs) ===")
            
            # Generate temporal sequences
            X_tr_seq, y_tr_seq = build_sequences(X_train, y_train_scaled, seq_len=seq_len)
            X_va_seq, y_va_seq = build_sequences(X_val, y_val_scaled, seq_len=seq_len)
            X_te_seq, y_te_seq = build_sequences(X_test, y_test_scaled, seq_len=seq_len)
            
            # Ground-truth targets in original unscaled domain
            _, y_tr_orig = build_sequences(X_train, y_train, seq_len=seq_len)
            _, y_va_orig = build_sequences(X_val, y_val, seq_len=seq_len)
            _, y_te_orig = build_sequences(X_test, y_test, seq_len=seq_len)
            
            train_loader = DataLoader(TrafficSequenceDataset(X_tr_seq, y_tr_seq), batch_size=128, shuffle=True)
            val_loader = DataLoader(TrafficSequenceDataset(X_va_seq, y_va_seq), batch_size=256, shuffle=False)
            test_loader = DataLoader(TrafficSequenceDataset(X_te_seq, y_te_seq), batch_size=256, shuffle=False)
            
            input_dim = X_train.shape[1]
            if arch == "LSTM":
                net = TrafficLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=2, dropout=0.2)
            else:
                net = TrafficGRU(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=2, dropout=0.2)
                
            trainer = DeepLearningTrainer(model=net, model_name=name, seq_len=seq_len, device=device)
            trainer.target_scaler = target_scaler
            
            ckpt_path = save_dir / f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '_')}.pt"
            train_time = trainer.train_model(train_loader, val_loader, max_epochs=12, patience=3, save_path=ckpt_path)
            
            # Predict and evaluate on original vehicle counts
            pred_train = trainer.predict(DataLoader(TrafficSequenceDataset(X_tr_seq, y_tr_seq), batch_size=256, shuffle=False))
            pred_val = trainer.predict(val_loader)
            pred_test = trainer.predict(test_loader)
            
            train_m = compute_metrics(y_tr_orig, pred_train)
            val_m = compute_metrics(y_va_orig, pred_val)
            test_m = compute_metrics(y_te_orig, pred_test)
            
            logger.info(f"{name} -> Val MAE: {val_m['mae']} | Test MAE: {test_m['mae']} | Test R2: {test_m['r2']} (Time: {train_time:.2f}s)")
            
            # Log to experiment registry
            log_experiment(
                model_name=name,
                model_family="Deep Learning",
                hyperparameters={"arch": arch, "seq_len": seq_len, "hidden_dim": hidden_dim, "num_layers": 2},
                train_metrics=train_m,
                val_metrics=val_m,
                test_metrics=test_m,
                training_time_sec=train_time,
                sequence_length=seq_len,
                promoted=False
            )
            
            dl_results.append({
                "Model": name,
                "Sequence (L)": seq_len,
                "Val MAE": val_m["mae"],
                "Val RMSE": val_m["rmse"],
                "Val R2": val_m["r2"],
                "Test MAE": test_m["mae"],
                "Test RMSE": test_m["rmse"],
                "Test R2": test_m["r2"],
                "Train Time (s)": round(train_time, 2)
            })
            
        leaderboard_dl = pd.DataFrame(dl_results).sort_values(by="Val MAE").reset_index(drop=True)
        return leaderboard_dl
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    df_dl = run_dl_experiments()
    print("\n" + "=" * 85)
    print(" GeoMind AI: Deep Learning (LSTM & GRU) Controlled Context Experiments")
    print("=" * 85)
    print(df_dl.to_string(index=False))
    print("=" * 85)
