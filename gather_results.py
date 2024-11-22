import argparse
import os
import pandas as pd
from paths import PLANTS, path_loader


# Directory containing the subdirectories with metrics.csv files
parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("months")
parser.add_argument("plantset", choices=PLANTS.keys())
args = parser.parse_args()

path_loader.init(args.months, args.plantset, 0)
info_csv_file = path_loader.paths['source_power_stat']
df = pd.read_csv(info_csv_file)
# 按TYPE列进行分组
grouped = df.groupby('TYPE')
save_path = os.path.join('./outputs', args.plantset, args.model, args.months)
os.makedirs(save_path, exist_ok=True)
# 对每个TYPE组进行处理
for group_type, group_data in grouped:
    # 这里你可以对每个分组的数据进行操作
    # 例如：打印该组的数据
    print(f"Processing data for TYPE {group_type}:")
    output_file = os.path.join(save_path, f"averaged_metrics_{group_type}.csv")
    filtered_record_file = os.path.join(save_path, f"filtered_stations_{group_type}.txt")
    extreme_large_file = os.path.join(save_path, f"toolargecap_stations_{group_type}.txt")
    # Initialize an empty DataFrame to store metrics from all files
    all_metrics = []
    all_filtered = []
    all_extreme_large = []
    # Loop through all subdirectories in ckpt_dir
    for idx, row in group_data.iterrows():
        station_number = str(int(row['PLANT_NO']))
        station_path, _ = path_loader.get_run_path_status(args.model)
        station_path = '/'.join(station_path.split('/')[:-1]+[f'{args.model}_{station_number}',])
        metrics_file = os.path.join(station_path, "metrics.csv")
        # Check if metrics.csv exists in the current subdirectory
        if os.path.isfile(metrics_file):
            # Read the metrics.csv and append to the list
            df = pd.read_csv(metrics_file)
            df['station'] = station_number
            if df["rmse"][0] >= 0:
                all_metrics.append(df)
            else:
                all_extreme_large.append(station_number)
        else:
            all_filtered.append(station_number)

    with open(filtered_record_file, 'w') as f:
        f.writelines([x+'\n' for x in all_filtered])

    with open(extreme_large_file, 'w') as f:
        f.writelines([x+'\n' for x in all_extreme_large])

    # Combine all metrics into a single DataFrame
    if all_metrics:
        combined_df = pd.concat(all_metrics)
        
        # Calculate the average of each metric
        average_metrics = combined_df.mean().to_frame().T
        
        # Save the averaged metrics to a new CSV file
        average_metrics.to_csv(output_file, index=False)
        
        # Print the averaged metrics and the total number of files collected
        print("Averaged metrics:")
        print(average_metrics)
        print(f"Total CSV files collected: {len(all_metrics)}")
        print(f"Averaged metrics saved in {output_file}")
    else:
        print("No metrics.csv files found.")
