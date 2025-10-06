# process_data.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import argparse

def _find_column(columns: List[str], patterns: List[str]) -> Optional[str]:
    for pattern in patterns:
        for col in columns:
            if re.search(pattern, col, re.IGNORECASE):
                return col
    return None

def _convert_current_to_power(current_amps: pd.Series, voltage: float, power_factor: float) -> pd.Series:
    # 三相有功功率近似 (W)
    return 1.732 * voltage * current_amps.abs() * power_factor

@dataclass
class EnhancedConfig:
    INPUT_FILE_PATH: str = "realdata.xlsx"
    OUTPUT_DIR: str = "data/processed_data_enhanced"
    RESAMPLE_RULE: str = "5T"
    HISTORY_DEPTH: int = 12
    TEST_SPLIT_RATIO: float = 0.2
    RATED_VOLTAGE: float = 380.0
    POWER_FACTOR: float = 0.85
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

def process_data_stage_one(config: EnhancedConfig) -> bool:
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
        if state_col in df.columns:
            agg_rules[state_col] = 'max'
    df_resampled = df.resample(config.RESAMPLE_RULE).agg(agg_rules).ffill()

    print("步骤 4/7: 构建特征集...")
    df_fmt = pd.DataFrame(index=df_resampled.index)
    df_fmt['Current_State'] = df_resampled['temp_room']
    df_fmt['Next_State'] = df_fmt['Current_State'].shift(-1)

    if 'current_avg' in df_resampled.columns and df_resampled['current_avg'].notna().any():
        df_fmt['Current_Action_Power'] = _convert_current_to_power(
            df_resampled['current_avg'], config.RATED_VOLTAGE, config.POWER_FACTOR
        )
    else:
        # 若无电流，用制冷状态估算功率（占位）
        # 注意：当列完全不存在时，get 返回标量；此处使用存在性判断避免 fillna 作用于标量
        if 'state_cooling' in df_resampled.columns:
            df_fmt['Current_Action_Power'] = df_resampled['state_cooling'].fillna(0) * 3000
        else:
            df_fmt['Current_Action_Power'] = 0.0

    df_fmt['Defrost_Temp'] = df_resampled.get('temp_defrost', df_resampled['temp_room'])

    # 二值状态列：若不存在，用常量广播生成整列
    if 'state_cooling' in df_resampled.columns:
        df_fmt['Is_Cooling'] = df_resampled['state_cooling'].fillna(0).astype(int)
    else:
        df_fmt['Is_Cooling'] = 0

    if 'state_defrost' in df_resampled.columns:
        df_fmt['Is_Defrosting'] = df_resampled['state_defrost'].fillna(0).astype(int)
    else:
        df_fmt['Is_Defrosting'] = 0

    if 'state_controller_on' in df_resampled.columns:
        df_fmt['Controller_On'] = df_resampled['state_controller_on'].fillna(1).astype(int)
    else:
        df_fmt['Controller_On'] = 1

    df_fmt['Start_Temp_Setting'] = df_resampled.get('temp_start_setting', -10.0)
    df_fmt['Stop_Temp_Setting'] = df_resampled.get('temp_stop_setting', -20.0)
    df_fmt['Outside_Air_Temperature'] = -5.0  # 如有外温传感器可替换

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

    test_size = max(1, int(len(df_final) * config.TEST_SPLIT_RATIO))
    test_size = min(test_size, len(df_final) - 1)
    train_df = df_final.iloc[:-test_size]
    test_df = df_final.iloc[-test_size:]

    print(f"步骤 7/7: 保存数据至 '{config.OUTPUT_DIR}'...")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(config.OUTPUT_DIR, "My_Training_data_enhanced.csv")
    test_path = os.path.join(config.OUTPUT_DIR, "My_Test_data_enhanced.csv")
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')

    print("--- ✅ [第一阶段] 数据处理成功 ---")
    print(f"训练集: {len(train_df)} 行, 测试集: {len(test_df)} 行")
    return True

def parse_args():
    ap = argparse.ArgumentParser(description="冷库数据增强处理与切分")
    ap.add_argument("--input_file", type=str, default="realdata.xlsx", help="原始数据文件路径（xlsx/csv）")
    ap.add_argument("--output_dir", type=str, default="data/pd", help="输出目录")
    ap.add_argument("--resample_rule", type=str, default="5T", help="重采样频率(如 5T/30S/1H)")
    ap.add_argument("--history_depth", type=int, default=12, help="历史步长")
    ap.add_argument("--test_split_ratio", type=float, default=0.2, help="测试集占比")
    ap.add_argument("--rated_voltage", type=float, default=380.0, help="额定电压(V)")
    ap.add_argument("--power_factor", type=float, default=0.85, help="功率因数")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = EnhancedConfig(
        INPUT_FILE_PATH=args.input_file,
        OUTPUT_DIR=args.output_dir,
        RESAMPLE_RULE=args.resample_rule,
        HISTORY_DEPTH=args.history_depth,
        TEST_SPLIT_RATIO=args.test_split_ratio,
        RATED_VOLTAGE=args.rated_voltage,
        POWER_FACTOR=args.power_factor,
    )
    ok = process_data_stage_one(cfg)
    if not ok:
        raise SystemExit(1)
