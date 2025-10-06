# train_physzero.py
# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
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


class fc_module(nn.Module):

    def __init__(self, layer_params, activation='tanh', dropout_rate=0.0):
        """
        :type layer_params: list[int]
        """
        super(fc_module, self).__init__()
        if activation == 'tanh':
            activation_function = nn.Tanh

        elif activation == 'relu':
            activation_function = nn.ReLU

        elif activation == 'sigmoid':
            activation_function = nn.Sigmoid

        else:
            raise ValueError(f"Unknown Activation function: {activation}")
        self.fc_module = nn.Sequential(
            nn.Linear(layer_params[0], layer_params[1]),
            activation_function(),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x):
        
        return self.fc_module(x)

# ---------------- Model ----------------
class PhysZeroMLP(pl.LightningModule):
    def __init__(self, input_size, feature_map, hidden_size, depth, lr, lambda_constraint=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.network = nn.ModuleList()
        self.depth = depth

        self.network.append(fc_module((input_size, hidden_size[0]), activation='relu', dropout_rate=0))
        addtemp = input_size
        for i in range(depth - 1):
            self.network.append(fc_module((hidden_size[i] + addtemp, hidden_size[i+1]), activation='relu', dropout_rate=0))
            addtemp += hidden_size[i]
        self.network.append(fc_module((hidden_size[-1] + addtemp, 1), activation='tanh', dropout_rate=0))
        # self.network = nn.Sequential(*layers)
        
        #depth 是激活层的层数-1 （除去最后一层的tanh）


        
        self.feature_map = feature_map

    def forward(self, x):

        x1 = x
        output = self.network[0](x1)
        # x1 = cat(x2, x1)
        x1 = x1

        for i in range(1, self.depth):
            x2 = output
            try:
                output = self.network[i](torch.concat([output,x1],dim=1))
            except:
                import sys
                sys.exit(f"{[output.shape,x1.shape]}")
            # output = self.network[i](torch.concat([output,x1],dim=0))
            x1 = torch.concat([x2, x1],dim=1)

        output = self.network[-1](torch.concat([output,x1],dim=1))
        # print("Anon: ",output.shape)

        return output

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

# ---------------- Evaluation (val set) ----------------
@torch.no_grad()
def evaluate(model, datamodule, device="cpu"):
    model.eval().to(device)
    datamodule.setup()
    loader = datamodule.val_dataloader()

    y_true_norm, y_pred_norm = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        preds = model(xb).cpu()
        y_true_norm.append(yb.cpu())
        y_pred_norm.append(preds)

    y_true_norm = torch.cat(y_true_norm, dim=0).squeeze()
    y_pred_norm = torch.cat(y_pred_norm, dim=0).squeeze()

    # 归一化空间指标
    mae_norm = torch.mean(torch.abs(y_pred_norm - y_true_norm)).item()
    rmse_norm = torch.sqrt(torch.mean((y_pred_norm - y_true_norm) ** 2)).item()

    # 反归一化到 °C
    sc = datamodule.scalers[datamodule.target_column]
    y_true = y_true_norm * sc['range'] + sc['min']
    y_pred = y_pred_norm * sc['range'] + sc['min']
    mae_c = torch.mean(torch.abs(y_pred - y_true)).item()
    rmse_c = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()

    print(f"\n=== Evaluation on Validation Set ===")
    print(f"MAE (norm):  {mae_norm:.6f}")
    print(f"RMSE (norm): {rmse_norm:.6f}")
    print(f"MAE (°C):    {mae_c:.6f}")
    print(f"RMSE (°C):   {rmse_c:.6f}")
    return mae_c, rmse_c

# ---------------- Train/Test runner ----------------
def main():
    ap = argparse.ArgumentParser(description="PhysZeroMLP 训练/测试（同一脚本）")
    # 模式选择
    ap.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="train 或 test")
    # 数据与特征
    ap.add_argument("--data_dir", type=str, default="data/pd", help="处理后CSV所在目录")
    ap.add_argument("--history_depth", type=int, default=12, help="历史步长（需与数据处理一致）")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    # 模型与优化
    ap.add_argument("--hidden_size", type=int, default=64)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--lambda_constraint", type=float, default=1.5)
    ap.add_argument("--max_epochs", type=int, default=75)   # 你当前代码默认 75
    ap.add_argument("--seed", type=int, default=42)
    # 日志与存储
    ap.add_argument("--log_dir", type=str, default="logs")
    ap.add_argument("--log_name", type=str, default="PhysZeroMLP_final")
    ap.add_argument("--model_out", type=str, default="models/PhysZeroMLP_final.pth")
    ap.add_argument("--model_in", type=str, default="models/PhysZeroMLP_final.pth", help="test 模式加载的模型路径")
    args = ap.parse_args()

    pl.seed_everything(args.seed)

    # 列定义（与处理脚本一致）
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
        # hidden_size=args.hidden_size,
        hidden_size=[8]*8,
        depth=args.depth,
        lr=args.learning_rate,
        lambda_constraint=args.lambda_constraint
    )

    if args.mode == "train":
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

        # 训练结束顺带评估一次（可选）
        device = "cuda" if torch.cuda.is_available() else "cpu"
        evaluate(model, data_module, device=device)

    else:  # args.mode == "test"
        # 加载已训练权重并在验证集评估
        if not os.path.exists(args.model_in):
            raise FileNotFoundError(f"模型权重不存在: {args.model_in}")
        ckpt = torch.load(args.model_in, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        evaluate(model, data_module, device=device)

if __name__ == "__main__":
    main()
