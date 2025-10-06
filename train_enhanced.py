# -*- coding: utf-8 -*-
"""
最终版集成训练脚本 (完整版)

功能:
- 阶段一: 自动处理原始Excel数据，生成包含完整物理特征的训练/测试集。
- 阶段二: 动态适配模型输入，加载处理好的数据。
- 阶段三: 使用带有物理和业务逻辑约束的 PhysZeroMLP 模型进行训练。

如何使用:
1. 将此文件保存为 `run_training_final.py`。
2. 将您的数据文件 (例如 'train.xlsx') 放在同一个目录下。
3. 在下面的 `main` 函数中，确认 `INPUT_FILE_PATH` 与您的文件名一致。
4. 运行命令: `python run_training_final.py`
"""

import pandas as pd
import numpy as np
import os
import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import warnings

# 忽略不必要的警告信息
warnings.filterwarnings('ignore')

# ======================================================================================
# 阶段一: 数据处理部分
# ======================================================================================

@dataclass
class EnhancedConfig:
    INPUT_FILE_PATH: str = "train.xlsx"
    OUTPUT_DIR: str = "data/processed_data_enhanced"
    RESAMPLE_RULE: str = "5T"
    HISTORY_DEPTH: int = 12
    TEST_SPLIT_RATIO: float = 0.2
    RATED_VOLTAGE: float = 380.0
    POWER_FACTOR: float = 0.85
    COLUMN_PATTERNS: Dict[str, List[str]] = field(default_factory=lambda: {
        "time": ["recordTime"], "temp_room": [r"0100\[?库温\]?"],
        "current_avg": [r"0105\[?三相平均电流\]?"], "temp_defrost": [r"0101\[?化霜温度\]?"],
        "temp_start_setting": [r"0107\[?开机温度\]?"], "temp_stop_setting": [r"0108\[?停机温度\]?"],
        "state_cooling": ["制冷状态"], "state_defrost": ["化霜状态"],
        "state_controller_on": [r"0406\[?控制器开关机\]?"]
    })

def _find_column(columns: List[str], patterns: List[str]) -> Optional[str]:
    for pattern in patterns:
        for col in columns:
            if re.search(pattern, col, re.IGNORECASE): return col
    return None

def _convert_current_to_power(current_amps: pd.Series, voltage: float, power_factor: float) -> pd.Series:
    return 1.732 * voltage * current_amps.abs() * power_factor

def process_data_stage_one(config: EnhancedConfig):
    print(f"--- [第一阶段] 开始处理数据: {config.INPUT_FILE_PATH} ---")
    input_path = Path(config.INPUT_FILE_PATH)
    if not input_path.exists():
        print(f"错误：找不到输入文件 {input_path}。")
        return False

    print("步骤 1/7: 加载数据...")
    df_raw = pd.read_excel(input_path) if input_path.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(input_path)

    print("步骤 2/7: 定位核心列...")
    cols = {key: _find_column(df_raw.columns, patterns) for key, patterns in config.COLUMN_PATTERNS.items()}
    if not all([cols["time"], cols["temp_room"]]):
        print("错误：缺少核心列 'recordTime' 或 '库温'。")
        return False
    
    found_cols = {k: v for k, v in cols.items() if v is not None}
    df = df_raw[list(found_cols.values())].rename(columns={v: k for k, v in found_cols.items()})

    print(f"步骤 3/7: 清洗与重采样 (频率: {config.RESAMPLE_RULE})...")
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time']).set_index('time').sort_index()
    agg_rules = {col: 'mean' for col in df.columns}
    for state_col in ["state_cooling", "state_defrost", "state_controller_on"]:
        if state_col in df.columns: agg_rules[state_col] = 'max'
    df_resampled = df.resample(config.RESAMPLE_RULE).agg(agg_rules).ffill()

    print("步骤 4/7: 构建特征集...")
    df_fmt = pd.DataFrame(index=df_resampled.index)
    df_fmt['Current_State'] = df_resampled['temp_room']
    df_fmt['Next_State'] = df_fmt['Current_State'].shift(-1)
    if 'current_avg' in df_resampled.columns and df_resampled['current_avg'].notna().any():
        df_fmt['Current_Action_Power'] = _convert_current_to_power(df_resampled['current_avg'], config.RATED_VOLTAGE, config.POWER_FACTOR)
    else:
        df_fmt['Current_Action_Power'] = df_resampled.get('state_cooling', 0).fillna(0) * 3000
    df_fmt['Defrost_Temp'] = df_resampled.get('temp_defrost', df_resampled['temp_room'])
    df_fmt['Is_Cooling'] = df_resampled.get('state_cooling', 0).fillna(0).astype(int)
    df_fmt['Is_Defrosting'] = df_resampled.get('state_defrost', 0).fillna(0).astype(int)
    df_fmt['Controller_On'] = df_resampled.get('state_controller_on', 1).fillna(1).astype(int)
    df_fmt['Start_Temp_Setting'] = df_resampled.get('temp_start_setting', -10.0)
    df_fmt['Stop_Temp_Setting'] = df_resampled.get('temp_stop_setting', -20.0)
    df_fmt['Outside_Air_Temperature'] = -5.0

    print(f"步骤 5/7: 创建 {config.HISTORY_DEPTH} 步历史特征...")
    for i in range(1, config.HISTORY_DEPTH + 1):
        df_fmt[f'Previous_State_t-{i}'] = df_fmt['Current_State'].shift(i)
        df_fmt[f'Previous_Action_Power_t-{i}'] = df_fmt['Current_Action_Power'].shift(i)

    print("步骤 6/7: 最终格式化与划分...")
    df_final = df_fmt.dropna().copy()
    if len(df_final) == 0:
        print("错误：处理后无有效数据。")
        return False
    df_final['Time'] = range(len(df_final))
    test_size = int(len(df_final) * config.TEST_SPLIT_RATIO)
    train_df = df_final.iloc[:-test_size]
    test_df = df_final.iloc[-test_size:]

    print(f"步骤 7/7: 保存数据至 '{config.OUTPUT_DIR}'...")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(config.OUTPUT_DIR, "My_Training_data_enhanced.csv")
    test_path = os.path.join(config.OUTPUT_DIR, "My_Test_data_enhanced.csv")
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')
    
    print("--- ✅ [第一阶段] 数据处理成功 ---")
    return True

# ======================================================================================
# 阶段二: PyTorch Lightning 数据模块与支持函数
# ======================================================================================

class EnhancedDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, feature_columns, target_column, batch_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.batch_size = batch_size
        self.scalers = {}

    def setup(self, stage=None):
        train_path = os.path.join(self.data_dir, "My_Training_data_enhanced.csv")
        val_path = os.path.join(self.data_dir, "My_Test_data_enhanced.csv")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        # 在训练集上计算标尺
        for col in self.feature_columns + [self.target_column]:
            min_val, max_val = train_df[col].min(), train_df[col].max()
            range_val = max_val - min_val if max_val > min_val else 1.0
            self.scalers[col] = {'min': min_val, 'max': max_val, 'range': range_val}

        # 归一化数据
        X_train_norm, y_train_norm = self._normalize(train_df)
        X_val_norm, y_val_norm = self._normalize(val_df)

        self.train_dataset = TensorDataset(X_train_norm, y_train_norm)
        self.val_dataset = TensorDataset(X_val_norm, y_val_norm)

    def _normalize(self, df):
        X_norm = df[self.feature_columns].copy()
        for col in self.feature_columns:
            X_norm[col] = (df[col] - self.scalers[col]['min']) / self.scalers[col]['range']
        
        y_norm = (df[self.target_column] - self.scalers[self.target_column]['min']) / self.scalers[self.target_column]['range']
        
        return torch.tensor(X_norm.values, dtype=torch.float32), torch.tensor(y_norm.values, dtype=torch.float32).view(-1, 1)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

# ======================================================================================
# 阶段三: 带物理约束的 PyTorch Lightning 模型
# ======================================================================================

class PhysZeroMLP(pl.LightningModule):
    def __init__(self, input_size, feature_map, hidden_size, depth, lr, lambda_constraint=1.0):
        super(PhysZeroMLP, self).__init__()
        self.save_hyperparameters()
        
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)
        
        # 创建从特征名到输入张量索引的映射
        self.feature_map = feature_map

    def forward(self, x):
        return self.network(x)

    def _calculate_constrained_loss(self, x, y_hat, y):
        prediction_loss = nn.MSELoss()(y_hat, y)

        # 从输入张量x中安全地获取特征列
        current_state_norm = x[:, self.feature_map['Current_State']]
        is_cooling = x[:, self.feature_map['Is_Cooling']]
        is_defrosting = x[:, self.feature_map['Is_Defrosting']]
        
        delta_t_predicted = y_hat.squeeze() - current_state_norm
        
        # 约束1: 制冷时温度下降
        cooling_mask = is_cooling > 0.5
        cooling_violation = torch.relu(delta_t_predicted[cooling_mask]).mean() if cooling_mask.any() else 0.0

        # 约束2: 化霜时温度上升
        defrost_mask = is_defrosting > 0.5
        defrost_violation = torch.relu(-delta_t_predicted[defrost_mask]).mean() if defrost_mask.any() else 0.0
        
        # 约束3: 自然状态下温度不下降
        natural_mask = (is_cooling < 0.5) & (is_defrosting < 0.5)
        natural_violation = torch.relu(-delta_t_predicted[natural_mask]).mean() if natural_mask.any() else 0.0
        
        constraint_loss = cooling_violation + defrost_violation + natural_violation
        total_loss = prediction_loss + self.hparams.lambda_constraint * constraint_loss

        return total_loss, prediction_loss, constraint_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss, pred_loss, cons_loss = self._calculate_constrained_loss(x, y_hat, y)
        self.log_dict({
            'train/total_loss': loss, 'train/prediction_loss': pred_loss, 
            'train/constraint_loss': cons_loss
        }, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss, pred_loss, cons_loss = self._calculate_constrained_loss(x, y_hat, y)
        self.log_dict({
            'val/total_loss': loss, 'val/prediction_loss': pred_loss, 
            'val/constraint_loss': cons_loss
        }, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# ======================================================================================
# 主训练流程
# ======================================================================================

def run_training(config: EnhancedConfig, model_params: dict):
    # --- 阶段一 ---
    if not process_data_stage_one(config):
        print("数据处理失败，训练终止。")
        return

    # --- 阶段二 & 三 ---
    print("\n--- [第二/三阶段] 开始模型训练 ---")
    pl.seed_everything(model_params.get("seed", 42))

    # 动态定义特征列
    # 这是最关键的一步：确保列的顺序是固定的
    base_features = ['Current_State', 'Current_Action_Power', 'Defrost_Temp', 'Is_Cooling', 'Is_Defrosting',
                     'Controller_On', 'Start_Temp_Setting', 'Stop_Temp_Setting', 'Outside_Air_Temperature']
    history_features = [f'Previous_State_t-{i}' for i in range(1, config.HISTORY_DEPTH + 1)] + \
                       [f'Previous_Action_Power_t-{i}' for i in range(1, config.HISTORY_DEPTH + 1)]
    feature_columns = base_features + history_features
    target_column = 'Next_State'
    
    # 为模型创建从特征名到索引的映射
    feature_map = {name: i for i, name in enumerate(feature_columns)}

    # 初始化数据模块
    data_module = EnhancedDataModule(
        data_dir=config.OUTPUT_DIR,
        feature_columns=feature_columns,
        target_column=target_column,
        batch_size=model_params.get("batch_size", 256)
    )

    # 初始化模型
    model = PhysZeroMLP(
        input_size=len(feature_columns),
        feature_map=feature_map,
        hidden_size=model_params.get("hidden_size", 64),
        depth=model_params.get("depth", 8),
        lr=model_params.get("learning_rate", 1e-4),
        lambda_constraint=model_params.get("lambda_constraint", 1.0)
    )

    # 设置提前停止
    early_stop_callback = EarlyStopping(monitor="val/total_loss", min_delta=1e-5, patience=20, verbose=True, mode="min")
    
    # 初始化训练器
    trainer = pl.Trainer(
        max_epochs=model_params.get("max_epochs", 500),
        accelerator='auto',
        callbacks=[early_stop_callback],
        logger=pl.loggers.TensorBoardLogger("logs/", name="PhysZeroMLP_final")
    )
    
    # 开始训练
    trainer.fit(model, data_module)
    
    # 保存最终模型
    save_path = f"models/PhysZeroMLP_final.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    
    print(f"\n--- ✅ 训练完成 ---")
    print(f"模型已保存至: {save_path}")


if __name__ == "__main__":
    # --- 1. 配置数据处理参数 ---
    data_config = EnhancedConfig(
        INPUT_FILE_PATH="realdata.xlsx",  # <--- 请确认这是您的数据文件名
        RESAMPLE_RULE="5T",
        HISTORY_DEPTH=12
    )

    # --- 2. 配置模型与训练参数 ---
    model_parameters = {
        "hidden_size": 64,
        "depth": 8,
        "learning_rate": 1e-4,
        "lambda_constraint": 1.5, # 物理约束的权重，可以调参
        "batch_size": 256,
        "max_epochs": 500,
        "seed": 42
    }

    # --- 3. 运行完整流程 ---
    run_training(data_config, model_parameters)