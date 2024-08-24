import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataloading import get_latest_checkpoint, load_checkpoint
from utils import get_model_and_loader
from lstm_model import BiLSTMNWPOnly
from constants import model_type_dict
import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions_vs_ground_truth(model, test_loader, denormalizer, filename, days=10, device='cuda'):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_gts = []
    
    with torch.no_grad():
        for X, Y, nwp_data in test_loader:
            X, Y, nwp_data = X.to(device), Y.to(device), nwp_data.to(device)
            # Generate predictions
            preds = model(nwp_data)
            # If outputs are normalized, denormalize them (assuming `scaler` was used to normalize)
            preds = denormalizer(preds).cpu().numpy()
            gt = denormalizer(Y).cpu().numpy()
            all_preds.extend(preds)
            all_gts.extend(gt)
    
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds).flatten()
    all_gts = np.array(all_gts).flatten()
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
    return mae, mse

# Example usage
# plot_predictions_vs_ground_truth(model, test_loader, scaler, device='cuda')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train BiLSTM model for power forecasting")
    parser.add_argument("model_type", type=int, choices=[0, 1])
    parser.add_argument("--plant_number", type=int, required=True, help="Power plant number to be used for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    
    args = parser.parse_args()
    args.model_type = model_type_dict[args.model_type]
    print(f'now drawing for {args.model_type}')
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data loaders and model
    model, train_loader, val_loader, test_loader, denormalizer = get_model_and_loader(args, device)

    latest_checkpoint = get_latest_checkpoint(args.checkpoint_dir)
    print(f'loading from {latest_checkpoint}')
    _, _ = load_checkpoint(latest_checkpoint, model, None)
    filename = os.path.join(args.checkpoint_dir, f"{args.plant_number}.png")
    mae, mse = plot_predictions_vs_ground_truth(model, test_loader, denormalizer, filename, device=device)
    print(mae)
    print(mse)