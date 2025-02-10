import os
from typing import Dict
import pandas as pd
from tqdm import tqdm
import argparse
from paths import path_loader

# HYPER
SPLIT = "newly_built"
REF_SPLITS_PATH = "/data1/yfliu/solar_baseline/solar/newly_built/newly_built_stations_info.csv"


def interpolate_missing_data(df):
    """
    Interpolates missing data in the DataFrame where missing values are identified by -999.
    
    Args:
        df (pd.DataFrame): The DataFrame to interpolate.
    
    Returns:
        pd.DataFrame: DataFrame with interpolated values.
    """
    df.replace(-999, pd.NA, inplace=True)  # Replace -999 with NaN
    # Convert all columns (except the timestamp) to numeric, forcing errors to NaN
    numeric_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    # Perform linear interpolation along the columns
    numeric_df.interpolate(method='linear', axis=1, inplace=True)
    # If you need to replace the original DataFrame with the interpolated data
    df.iloc[:, 1:] = numeric_df.iloc[:, :]
    df.fillna(method='bfill', inplace=True)  # Fill remaining NaNs with backward fill
    df.fillna(method='ffill', inplace=True)  # Fill remaining NaNs with forward fill
    return df

def save_data(train_file, test_file, train_power_file, valid_power_file, test_power_file, split=0.95, months=12):
    """
    Saves training and testing data to CSV files.
    """
    # Calculate the index to split at
    split_idx = int(len(train_file) * split)
    valid_df = train_file.iloc[split_idx:]
    train_df = train_file.iloc[:split_idx]
    split_date = train_df.iloc[:, 0].min() + pd.DateOffset(months=months)
    train_df = train_df[train_df.iloc[:, 0] < split_date]
    test_df = test_file
    train_df.to_csv(train_power_file, index=False)
    valid_df.to_csv(valid_power_file, index=False)
    test_df.to_csv(test_power_file, index=False)

def load_data(train_filename, test_filename):
    """
    Loads training and testing data from CSV files if they exist.
    
    Args:
        train_file (str): Path to the file where training data is saved.
        test_file (str): Path to the file where testing data is saved.
    
    Returns:
        dict: A dictionary containing loaded training and testing data.
    """
    if os.path.exists(train_filename) and os.path.exists(test_filename):
        training_set = pd.read_csv(train_filename).iloc[:, 1:].values  # 删除第一列
        testing_set = pd.read_csv(test_filename).iloc[:, 1:].values  # 删除第一列
    return {
        "train": training_set,
        "test": testing_set,
    }

def get_data(plant_number, months, overwrite=False, fin_time='2024-04-17 09:00:00', split_info:Dict=None):
    path_loader.init(f"{args.months}m", SPLIT, plant_number)
    train_power_file = path_loader.paths['train_power_file']
    valid_power_file = path_loader.paths['valid_power_file']
    test_power_file = path_loader.paths['test_power_file']
    source_power_file = path_loader.paths['source_power_file']
    if not os.path.exists(source_power_file):
        raise RuntimeError(f'{source_power_file} does not exist')
    if overwrite or not os.path.exists(train_power_file) or not os.path.exists(test_power_file):
        # Load and preprocess data
        data = pd.read_csv(source_power_file)
        # Handle missing data (-999) with linear interpolation
        data = interpolate_missing_data(data)
        data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0])
        fin_time = pd.to_datetime(fin_time)
        if split_info:
            split_date = pd.to_datetime(split_info['测试集开始时间'])
        else:
            split_date = data.iloc[:, 0].min() + pd.DateOffset(years=1)
        training_set = data[data.iloc[:, 0] < split_date]
        testing_set = data[(data.iloc[:, 0] >= split_date) & (data.iloc[:, 0] <= fin_time)]
        # Save the processed data
        save_data(training_set, testing_set, train_power_file, valid_power_file, test_power_file, months=months)
    
    return load_data(train_power_file, test_power_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("months", type=int)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument("--sort", action='store_true', help='do it only at first time')
    args = parser.parse_args()
    path_loader.init(f'{args.months}m', SPLIT, 0)
    args.csv_dir = path_loader.config['paths']['source_power_stat']
    if args.sort:
        sorted_csv_dir = os.path.join(os.path.dirname(args.csv_dir), f'sorted_{os.path.basename(args.csv_dir)}')
        if not os.path.exists(sorted_csv_dir) or args.overwrite:
            df = pd.read_csv(f"{args.csv_dir}")
            df = df.sort_values(by='TYPE')
            df.to_csv(sorted_csv_dir)
    else:
        df = pd.read_csv(args.csv_dir)
    pbar = tqdm(range(len(df)))
    df_splits = None
    split_info = None
    if REF_SPLITS_PATH:
        df_splits = pd.read_csv(REF_SPLITS_PATH)
    for idx, row in df.iterrows():
        plant_idx = idx
        if df_splits is not None:
            plant_no = row['PLANT_NO']
            split_info = df_splits[df_splits['PLANT_NO'] == plant_no].to_dict(orient='records')[0]
        get_data(plant_idx, months=args.months, overwrite=args.overwrite, split_info=split_info)
        pbar.update()