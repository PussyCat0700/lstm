import argparse
import os
import pandas as pd


# Directory containing the subdirectories with metrics.csv files
parser = argparse.ArgumentParser()
parser.add_argument("ckpt_dir")
parser.add_argument('prefix')
args = parser.parse_args()

info_csv_file = "/data1/yfliu/solar_baseline/solar/nmg_data_new/nmg_info.csv"
df = pd.read_csv(info_csv_file)
# 按TYPE列进行分组
grouped = df.groupby('TYPE')
ckpt_dir = args.ckpt_dir
subdir = os.path.basename(os.path.normpath(ckpt_dir))
# 对每个TYPE组进行处理
for group_type, group_data in grouped:
    # 这里你可以对每个分组的数据进行操作
    # 例如：打印该组的数据
    print(f"Processing data for TYPE {group_type}:")
    output_file = f"averaged_metrics_{subdir}_{group_type}.csv"
    filtered_record_file = f"filtered_stations_{subdir}_{group_type}.txt"
    extreme_large_file = f"toolargecap_stations_{subdir}_{group_type}.txt"

    # Initialize an empty DataFrame to store metrics from all files
    all_metrics = []
    all_filtered = []
    all_extreme_large = []
    # Loop through all subdirectories in ckpt_dir
    for idx, row in group_data.iterrows():
        station_number = str(int(row['PLANT_NO']))
        station_path = os.path.join(ckpt_dir, f'{args.prefix}_'+station_number)
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
