import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_predictions_vs_ground_truth_vanilla(all_preds, all_gts, filename, days=10, all_y_times=None):
    mae = np.mean(np.abs(all_preds - all_gts))
    mse = np.mean(np.abs(all_preds - all_gts) ** 2)
    # Plot the results
    plt.figure()
    plt.plot(all_gts[:96*days], label='Ground Truth', color='blue')
    plt.plot(all_preds[:96*days], label='Prediction', color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Power (MW)')
    plt.title('Predicted vs Ground Truth Power Output')
    plt.legend()
    plt.savefig(filename)
    dir_name = os.path.dirname(filename)
    if all_y_times is None:
        np.save(os.path.join(dir_name, 'all_preds.npy'), all_preds)
        np.save(os.path.join(dir_name, 'all_gts.npy'), all_gts)
    else:
        df = pd.DataFrame({
            'Datetime': all_y_times,
            'Predictions': all_preds,
            'Ground Truth': all_gts
        })
        df.to_csv(os.path.join(dir_name, 'output.csv'), index=False)
    return mae, mse