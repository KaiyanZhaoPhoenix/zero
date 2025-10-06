# -*- coding: utf-8 -*-
"""
第一阶段: 数据与特征工程 (修复版)

功能:
1. 读取 'train.xlsx' 文件中的原始冷库运行数据。
2. 提取所有核心物理量、状态量和控制设定参数。
3. 对数据进行清洗、按指定频率重采样，并填充缺失值。
4. 构建一个包含当前状态、历史趋势和控制逻辑的完整特征集。
5. 将处理好的数据按时间顺序划分为训练集和测试集。
6. 输出可直接用于模型训练的CSV文件。

修复说明:
- 增加了 .fillna(0) 来处理因重采样导致数据开头可能存在的NaN值，
  避免在 .astype(int) 类型转换时出现 IntCastingNaNError。
"""

import pandas as pd
import numpy as np
import os
import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class EnhancedConfig:
    """
    配置类，用于管理所有数据处理的参数。
    """
    # [需要配置] 根据您的文件名修改，这里假设是 realdata.xlsx
    INPUT_FILE_PATH: str = "realdata.xlsx"
    
    # 输出文件夹名称
    OUTPUT_DIR: str = "data/processed_data_enhanced"
    
    # 数据重采样频率。"5T"代表5分钟, "30S"代表30秒, "1H"代表1小时
    RESAMPLE_RULE: str = "5T"
    
    # 创建历史数据的深度（步长）
    HISTORY_DEPTH: int = 12
    
    # 测试集在总数据中所占的比例
    TEST_SPLIT_RATIO: float = 0.2
    
    # --- 电流转换为功率所需参数 ---
    RATED_VOLTAGE: float = 380.0  # 额定电压 (V)
    POWER_FACTOR: float = 0.85   # 功率因数

    # --- 列名匹配规则 ---
    COLUMN_PATTERNS: Dict[str, List[str]] = field(default_factory=lambda: {
        "time": ["recordTime"],
        "temp_room": [r"0100\[?库温\]?"],
        "current_avg": [r"0105\[?三相平均电流\]?"],
        "temp_defrost": [r"0101\[?化霜温度\]?"],
        "temp_start_setting": [r"0107\[?开机温度\]?"],
        "temp_stop_setting": [r"0108\[?停机温度\]?"],
        "state_cooling": ["制冷状态"],
        "state_defrost": ["化霜状态"],
        "state_controller_on": [r"0406\[?控制器开关机\]?"]
    })

def _find_column(columns: List[str], patterns: List[str]) -> Optional[str]:
    """
    根据正则表达式在DataFrame的列名中查找匹配的列。
    """
    for pattern in patterns:
        for col in columns:
            if re.search(pattern, col, re.IGNORECASE):
                return col
    return None

def _convert_current_to_power(current_amps: pd.Series, voltage: float, power_factor: float) -> pd.Series:
    """
    将三相电流(A)转换为有功功率(W)。
    """
    return 1.732 * voltage * current_amps.abs() * power_factor

def process_data_stage_one(config: EnhancedConfig):
    """
    执行第一阶段数据处理的完整流程。
    """
    print(f"--- [第一阶段] 开始处理增强版数据: {config.INPUT_FILE_PATH} ---")
    input_path = Path(config.INPUT_FILE_PATH)
    if not input_path.exists():
        print(f"错误：找不到输入文件 {input_path}。")
        return

    # 1. 加载数据
    print("步骤 1/7: 加载原始数据...")
    try:
        df_raw = pd.read_excel(input_path, sheet_name=0) if input_path.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(input_path)
    except Exception as e:
        print(f"错误：加载文件失败。错误信息: {e}")
        return

    # 2. 定位所有核心列
    print("步骤 2/7: 根据规则自动定位所有核心列...")
    cols = {key: _find_column(df_raw.columns, patterns) for key, patterns in config.COLUMN_PATTERNS.items()}
    
    if not all([cols["time"], cols["temp_room"]]):
        print("错误：数据中缺少最核心的列（'recordTime'或'库温'），处理中止。")
        return
    
    found_cols = {k: v for k, v in cols.items() if v is not None}
    print(f"成功定位到 {len(found_cols)} 个核心列: {list(found_cols.keys())}")
    df = df_raw[list(found_cols.values())].rename(columns={v: k for k, v in found_cols.items()})

    # 3. 清洗、重采样与填充
    print(f"步骤 3/7: 清洗数据，并按 '{config.RESAMPLE_RULE}' 频率重采样...")
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time']).set_index('time').sort_index()

    agg_rules = {col: 'mean' for col in df.columns}
    for state_col in ["state_cooling", "state_defrost", "state_controller_on"]:
        if state_col in df.columns:
            agg_rules[state_col] = 'max' 
            
    df_resampled = df.resample(config.RESAMPLE_RULE).agg(agg_rules)
    df_resampled = df_resampled.ffill()

    # 4. 构建完整的特征集
    print("步骤 4/7: 构建包含物理信息和控制逻辑的完整特征集...")
    df_fmt = pd.DataFrame(index=df_resampled.index)
    
    df_fmt['Current_State'] = df_resampled['temp_room']
    df_fmt['Next_State'] = df_fmt['Current_State'].shift(-1)
    
    if 'current_avg' in df_resampled.columns and df_resampled['current_avg'].notna().any():
        df_fmt['Current_Action_Power'] = _convert_current_to_power(df_resampled['current_avg'], config.RATED_VOLTAGE, config.POWER_FACTOR)
    else:
        print("警告: 未找到有效的'三相平均电流'数据，将根据'制冷状态'估算功率。")
        df_fmt['Current_Action_Power'] = df_resampled.get('state_cooling', 0).fillna(0) * 3000 
        
    df_fmt['Defrost_Temp'] = df_resampled.get('temp_defrost', df_resampled['temp_room'])
    
    # --- 修复区域 ---
    # 在 astype(int) 之前，增加 .fillna(0) 来填充可能存在的NaN值
    df_fmt['Is_Cooling'] = df_resampled.get('state_cooling', 0).fillna(0).astype(int)
    df_fmt['Is_Defrosting'] = df_resampled.get('state_defrost', 0).fillna(0).astype(int)
    # 对于控制器状态，默认填充为1（开启）更合理
    df_fmt['Controller_On'] = df_resampled.get('state_controller_on', 1).fillna(1).astype(int) 
    # --- 修复结束 ---
    
    df_fmt['Start_Temp_Setting'] = df_resampled.get('temp_start_setting', -10.0) 
    df_fmt['Stop_Temp_Setting'] = df_resampled.get('temp_stop_setting', -20.0)
    df_fmt['Outside_Air_Temperature'] = -5.0

    # 5. 创建历史特征
    print(f"步骤 5/7: 为核心动态变量创建 {config.HISTORY_DEPTH} 步历史特征...")
    for i in range(1, config.HISTORY_DEPTH + 1):
        df_fmt[f'Previous_State_t-{i}'] = df_fmt['Current_State'].shift(i)
        df_fmt[f'Previous_Action_Power_t-{i}'] = df_fmt['Current_Action_Power'].shift(i)
    
    # 6. 最终格式化与数据集划分
    print("步骤 6/7: 移除存在缺失值的行，并按时间顺序划分训练/测试集...")
    df_final = df_fmt.dropna().copy()
    if len(df_final) == 0:
        print("错误：处理后没有剩下任何有效数据行，请检查原始数据和采样频率设置。")
        return
        
    df_final['Time'] = range(len(df_final))

    test_size = int(len(df_final) * config.TEST_SPLIT_RATIO)
    train_df = df_final.iloc[:-test_size]
    test_df = df_final.iloc[-test_size:]

    # 7. 保存数据
    print(f"步骤 7/7: 保存处理好的数据到 '{config.OUTPUT_DIR}' 文件夹...")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    train_path = os.path.join(config.OUTPUT_DIR, "My_Training_data_enhanced.csv")
    test_path = os.path.join(config.OUTPUT_DIR, "My_Test_data_enhanced.csv")
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')
    
    print("\n--- ✅ [第一阶段] 数据处理成功 ---")
    print(f"共生成 {len(df_final)} 行有效数据。")
    print(f"训练数据 ({len(train_df)}行) 已保存至: {train_path}")
    print(f"测试数据 ({len(test_df)}行) 已保存至: {test_path}")

if __name__ == "__main__":
    config = EnhancedConfig()
    # 根据您执行的命令，文件名是 'process_data.py'，所以您当时用的是 'realdata.xlsx'
    # 我在这里将配置改回 'realdata.xlsx' 以匹配您的运行环境
    config.INPUT_FILE_PATH = "realdata.xlsx" 
    process_data_stage_one(config)