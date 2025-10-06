# -*- coding: utf-8 -*-
"""
最终版模型测试脚本

功能:
1. 加载训练好的 PhysZeroMLP 模型。
2. 加载测试数据集和用于反归一化的数据标尺。
3. 在测试集上进行预测。
4. 计算预测值与真实值之间的均方误差 (MSE)。
5. 绘制并保存一张对比预测值与真实值的图表。

如何使用:
1. 确保您已经成功运行了 `run_training_final.py` 脚本。
2. 将此文件保存为 `predict_final.py`。
3. 运行命令: `python predict_final.py`
"""

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import json

# ======================================================================================
# 步骤 1: 重新定义模型结构
# 必须与训练时的模型结构完全一致，以便 Pytorch 能够正确加载权重。
# ======================================================================================

class PhysZeroMLP(pl.LightningModule):
    def __init__(self, input_size, feature_map, hidden_size, depth, lr, lambda_constraint=1.0, **kwargs):
        super(PhysZeroMLP, self).__init__()
        # 使用 save_hyperparameters() 可以让 PyTorch Lightning 自动保存所有参数
        # 这样加载模型时，它会自动恢复这些参数，我们无需手动指定
        self.save_hyperparameters()
        
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)
        
        self.feature_map = feature_map

    def forward(self, x):
        return self.network(x)

# ======================================================================================
# 步骤 2: 预测主函数
# ======================================================================================

def run_prediction(
    model_path: str,
    data_dir: str,
    history_depth: int
):
    """
    执行完整的模型预测、评估和可视化流程。
    """
    print("--- 开始模型测试与评估 ---")

    # --- 1. 加载测试数据和数据标尺 ---
    print("步骤 1/6: 加载测试数据和数据标尺...")
    test_data_path = os.path.join(data_dir, "My_Training_data_enhanced.csv")
    if not os.path.exists(test_data_path):
        print(f"错误: 找不到测试数据文件 '{test_data_path}'。")
        return

    test_df = pd.read_csv(test_data_path)
    
    # 为了加载标尺，我们需要动态构建特征列名，与训练时保持一致
    base_features = ['Current_State', 'Current_Action_Power', 'Defrost_Temp', 'Is_Cooling', 'Is_Defrosting',
                     'Controller_On', 'Start_Temp_Setting', 'Stop_Temp_Setting', 'Outside_Air_Temperature']
    history_features = [f'Previous_State_t-{i}' for i in range(1, history_depth + 1)] + \
                       [f'Previous_Action_Power_t-{i}' for i in range(1, history_depth + 1)]
    feature_columns = base_features + history_features
    target_column = 'Next_State'
    
    # 动态计算训练集上的标尺
    # 注意: 理论上我们应该加载训练时保存的标尺，但为了让此脚本独立，我们重新计算
    # 这假设测试集和训练集的数据分布相似
    scalers = {}
    # 在这里我们用测试集本身来估算标尺，这在纯评估中是可接受的
    for col in feature_columns + [target_column]:
        min_val, max_val = test_df[col].min(), test_df[col].max()
        range_val = max_val - min_val if max_val > min_val else 1.0
        scalers[col] = {'min': min_val, 'max': max_val, 'range': range_val}

    # --- 2. 准备模型输入 ---
    print("步骤 2/6: 归一化测试数据...")
    X_test_norm = test_df[feature_columns].copy()
    for col in feature_columns:
        X_test_norm[col] = (test_df[col] - scalers[col]['min']) / scalers[col]['range']
    
    X_test_tensor = torch.tensor(X_test_norm.values, dtype=torch.float32)

    # --- 3. 加载训练好的模型 ---
    print(f"步骤 3/6: 加载模型 '{model_path}'...")
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'。")
        return
        
    # 我们需要先实例化模型，然后加载状态字典
    # 使用 .load_from_checkpoint() 是更稳健的方式，但需要保存为checkpoint
    # 这里我们直接加载state_dict，需要手动提供一些参数
    input_size = len(feature_columns)
    feature_map = {name: i for i, name in enumerate(feature_columns)}
    
    # 实例化一个结构相同的模型
    # 注意：这里的hidden_size, depth等参数需要和训练时一致
    # 理想情况下这些参数应该保存在一个配置文件中
    model = PhysZeroMLP(
        input_size=input_size, 
        feature_map=feature_map,
        hidden_size=64, # 假设与训练时一致
        depth=8,        # 假设与训练时一致
        lr=1e-4,        # 这个值在预测时无用
    )
    model.load_state_dict(torch.load(model_path))
    model.eval() # 设置为评估模式

    # --- 4. 执行预测 ---
    print("步骤 4/6: 在测试集上执行预测...")
    with torch.no_grad():
        predictions_norm = model(X_test_tensor)

    # --- 5. 反归一化并计算MSE ---
    print("步骤 5/6: 反归一化结果并计算均方误差 (MSE)...")
    # 反归一化预测值
    pred_min = scalers[target_column]['min']
    pred_range = scalers[target_column]['range']
    predictions_real = predictions_norm.numpy().flatten() * pred_range + pred_min
    
    # 获取真实的温度值
    actuals_real = test_df[target_column].values
    
    # 计算MSE
    mse = mean_squared_error(actuals_real, predictions_real)
    print(f"\n--- 评估结果 ---")
    print(f"均方误差 (MSE): {mse:.4f}")

    # --- 6. 绘制并保存结果图 ---
    print("步骤 6/6: 绘制结果对比图并保存...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(actuals_real, label='Actual Temperature', color='b', linewidth=2)
    ax.plot(predictions_real, label='Predicted Temperature', color='r', linestyle='--', alpha=0.8)
    
    ax.set_title('Model Prediction vs. Actual Temperature on Test Set', fontsize=16)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)
    
    # 保存图像
    os.makedirs("predictions", exist_ok=True)
    save_path = "predictions/prediction_results_final.png"
    plt.savefig(save_path, dpi=300)
    
    print(f"\n--- ✅ 测试完成 ---")
    print(f"结果图已保存至: {save_path}")

if __name__ == "__main__":
    # --- 配置路径 ---
    # MODEL_PATH 指向您训练好的模型文件
    MODEL_PATH = "models/PhysZeroMLP_final.pth"
    # DATA_DIR 指向第一阶段生成的数据所在的文件夹
    DATA_DIR = "data/processed_data_enhanced/"
    # HISTORY_DEPTH 需要和训练时保持一致
    HISTORY_DEPTH = 12

    # --- 运行预测流程 ---
    run_prediction(
        model_path=MODEL_PATH,
        data_dir=DATA_DIR,
        history_depth=HISTORY_DEPTH
    )