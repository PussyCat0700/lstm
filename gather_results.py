import argparse
import os
import pandas as pd
from paths import results_save_path


# Directory containing the subdirectories with metrics.csv files
parser = argparse.ArgumentParser()
parser.add_argument("model_type_number", type=int)
args = parser.parse_args()

ckpt_dir = results_save_path
if args.model_type_number == 0:
    subdir = "lstm"
elif args.model_type_number == 1:
    subdir = "ffnn"
elif args.model_type_number == 2:
    subdir = "gpnn"
elif args.model_type_number == 3:
    subdir = "xgboost"
elif args.model_type_number == 4:
    subdir = "gpcf"
ckpt_dir = os.path.join(ckpt_dir, subdir)
output_file = f"averaged_metrics_{subdir}.csv"
filtered_record_file = f"filtered_stations_{subdir}.txt"
extreme_large_file = f"toolargecap_stations_{subdir}.txt"

# Initialize an empty DataFrame to store metrics from all files
all_metrics = []
all_filtered = []
all_extreme_large = []
# Loop through all subdirectories in ckpt_dir
for subdir in os.listdir(ckpt_dir):
    subdir_path = os.path.join(ckpt_dir, subdir)
    metrics_file = os.path.join(subdir_path, "metrics.csv")
    
    station_number = subdir.split('_')[-1]
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
