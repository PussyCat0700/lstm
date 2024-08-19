import os
import pandas as pd
from paths import source_power_file, train_power_file, valid_power_file, test_power_file

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

def save_data(train_file, test_file, split=0.95):
    """
    Saves training and testing data to CSV files.
    """
    # Calculate the index to split at
    split_idx = int(len(train_file) * split)
    valid_df = train_file.iloc[split_idx:]
    train_df = train_file.iloc[:split_idx]
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
        training_set = pd.read_csv(train_filename).drop(columns=['Unnamed: 0']).values
        testing_set = pd.read_csv(test_filename).drop(columns=['Unnamed: 0']).values
    return {
        "train": training_set,
        "test": testing_set,
    }

def get_data(overwrite=False):
    if overwrite or not os.path.exists(train_power_file) or not os.path.exists(test_power_file):
        # Load and preprocess data
        data = pd.read_csv(source_power_file)
        # Handle missing data (-999) with linear interpolation
        data = interpolate_missing_data(data)
        data['Unnamed: 0'] = pd.to_datetime(data['Unnamed: 0'])
        split_date = data['Unnamed: 0'].min() + pd.DateOffset(years=1) + pd.DateOffset(months=1)
        training_set = data[data['Unnamed: 0'] < split_date]
        testing_set = data[data['Unnamed: 0'] >= split_date]
        # Save the processed data
        save_data(training_set, testing_set)
    
    return load_data(train_power_file, test_power_file)

if __name__ == "__main__":
    get_data(overwrite=True)