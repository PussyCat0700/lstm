import argparse
import os
import pandas as pd


# Directory containing the subdirectories with metrics.csv files
parser = argparse.ArgumentParser()
parser.add_argument("model_type_number", type=int)
args = parser.parse_args()

ckpt_dir = "/data1/yfliu/windpower_baseline"
if args.model_type_number == 0:
    subdir = "lstm_hourly"
elif args.model_type_number == 1:
    subdir = "ffnn_hourly"
elif args.model_type_number == 2:
    subdir = "gpnn_hourly"
ckpt_dir = os.path.join(ckpt_dir, subdir)
output_file = f"averaged_metrics_{subdir}.csv"

# Initialize an empty DataFrame to store metrics from all files
all_metrics = []

# Loop through all subdirectories in ckpt_dir
for subdir in os.listdir(ckpt_dir):
    subdir_path = os.path.join(ckpt_dir, subdir)
    metrics_file = os.path.join(subdir_path, "metrics.csv")
    
    # Check if metrics.csv exists in the current subdirectory
    if os.path.isfile(metrics_file):
        # Read the metrics.csv and append to the list
        df = pd.read_csv(metrics_file)
        all_metrics.append(df)

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
