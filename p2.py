# train_physzero.py
# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

# ---------------- DataModule ----------------
class EnhancedDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, feature_columns, target_column, batch_size=256, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scalers = {}

    def setup(self, stage=None):
        train_path = os.path.join(self.data_dir, "My_Training_data_enhanced.csv")
        val_path = os.path.join(self.data_dir, "My_Test_data_enhanced.csv")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        # 依据训练集计算 min-max（掩码列需保持0/1，不做归一化）
        mask_cols = {"Is_Cooling", "Is_Defrosting", "Controller_On"}
        for col in self.feature_columns + [self.target_column]:
            if col in mask_cols:
                self.scalers[col] = {"kind": "mask"}
            else:
                min_val, max_val = train_df[col].min(), train_df[col].max()
                range_val = max_val - min_val if max_val > min_val else 1.0
                self.scalers[col] = {'min': min_val, 'max': max_val, 'range': range_val, "kind": "minmax"}

        X_train, y_train = self._normalize(train_df)
        X_val, y_val = self._normalize(val_df)

        self.train_dataset = TensorDataset(X_train, y_train)
        self.val_dataset = TensorDataset(X_val, y_val)

    def _normalize(self, df):
        X = df[self.feature_columns].copy()
        for col in self.feature_columns:
            sc = self.scalers[col]
            if sc.get("kind") == "mask":
                X[col] = df[col].astype(float)
            else:
                X[col] = (df[col] - sc['min']) / sc['range']
        y = (df[self.target_column] - self.scalers[self.target_column]['min']) / self.scalers[self.target_column]['range']
        return torch.tensor(X.values, dtype=torch.float32), torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

# ---------------- Model ----------------
class PhysZeroMLP(pl.LightningModule):
    def __init__(self, input_size, feature_map, hidden_size, depth, lr, lambda_constraint=1.0):
        super().__init__()
        self.save_hyperparameters()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)
        self.feature_map = feature_map

    def forward(self, x):
        return self.network(x)

    def _calculate_constrained_loss(self, x, y_hat, y):
        prediction_loss = nn.MSELoss()(y_hat, y)

        current_state_norm = x[:, self.feature_map['Current_State']]
        is_cooling = x[:, self.feature_map['Is_Cooling']]
        is_defrosting = x[:, self.feature_map['Is_Defrosting']]

        delta_t_predicted = y_hat.squeeze() - current_state_norm

        # 约束：制冷降温
        cooling_mask = is_cooling > 0.5
        cooling_violation = torch.relu(delta_t_predicted[cooling_mask]).mean() if cooling_mask.any() else torch.tensor(0., device=x.device)

        # 约束：化霜升温
        defrost_mask = is_defrosting > 0.5
        defrost_violation = torch.relu(-delta_t_predicted[defrost_mask]).mean() if defrost_mask.any() else torch.tensor(0., device=x.device)

        # 约束：空闲不降温
        natural_mask = (is_cooling < 0.5) & (is_defrosting < 0.5)
        natural_violation = torch.relu(-delta_t_predicted[natural_mask]).mean() if natural_mask.any() else torch.tensor(0., device=x.device)

        constraint_loss = cooling_violation + defrost_violation + natural_violation
        total_loss = prediction_loss + self.hparams.lambda_constraint * constraint_loss
        return total_loss, prediction_loss, constraint_loss

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss, pred_loss, cons_loss = self._calculate_constrained_loss(x, y_hat, y)
        self.log_dict({'train/total_loss': loss,
                       'train/prediction_loss': pred_loss,
                       'train/constraint_loss': cons_loss}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss, pred_loss, cons_loss = self._calculate_constrained_loss(x, y_hat, y)
        self.log_dict({'val/total_loss': loss,
                       'val/prediction_loss': pred_loss,
                       'val/constraint_loss': cons_loss}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# ---------------- Train runner ----------------
def main():
    ap = argparse.ArgumentParser(description="PhysZeroMLP 训练（业务规则约束版）")
    # 数据与特征
    ap.add_argument("--model", type=str, default="Train",help="Train/Test")
    ap.add_argument("--data_dir", type=str, default="data/pd", help="处理后CSV所在目录")
    ap.add_argument("--history_depth", type=int, default=12, help="历史步长（需与数据处理一致）")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    # 模型与优化
    ap.add_argument("--hidden_size", type=int, default=64)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--lambda_constraint", type=float, default=1.5)
    ap.add_argument("--max_epochs", type=int, default=75)
    ap.add_argument("--seed", type=int, default=42)
    # 日志与存储
    ap.add_argument("--log_dir", type=str, default="logs")
    ap.add_argument("--log_name", type=str, default="PhysZeroMLP_final")
    ap.add_argument("--model_out", type=str, default="models/PhysZeroMLP_final.pth")
    args = ap.parse_args()

    pl.seed_everything(args.seed)

    base_features = [
        'Current_State', 'Current_Action_Power', 'Defrost_Temp',
        'Is_Cooling', 'Is_Defrosting', 'Controller_On',
        'Start_Temp_Setting', 'Stop_Temp_Setting', 'Outside_Air_Temperature'
    ]
    history_features = [f'Previous_State_t-{i}' for i in range(1, args.history_depth + 1)] + \
                       [f'Previous_Action_Power_t-{i}' for i in range(1, args.history_depth + 1)]
    feature_columns = base_features + history_features
    target_column = 'Next_State'
    feature_map = {name: i for i, name in enumerate(feature_columns)}

    data_module = EnhancedDataModule(
        data_dir=args.data_dir,
        feature_columns=feature_columns,
        target_column=target_column,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = PhysZeroMLP(
        input_size=len(feature_columns),
        feature_map=feature_map,
        hidden_size=args.hidden_size,
        depth=args.depth,
        lr=args.learning_rate,
        lambda_constraint=args.lambda_constraint
    )

    early_stop = EarlyStopping(monitor="val/total_loss", min_delta=1e-5, patience=20, verbose=True, mode="min")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        callbacks=[early_stop],
        logger=pl.loggers.TensorBoardLogger(args.log_dir, name=args.log_name)
    )

    trainer.fit(model, data_module)

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    torch.save(model.state_dict(), args.model_out)
    print(f"\n--- ✅ 训练完成，模型已保存至: {args.model_out} ---")

if __name__ == "__main__":
    main()
