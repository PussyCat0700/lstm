import os
from lstm_model import BiLSTMWithFusion
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataloading import get_data_loaders_and_denormalizer, get_latest_checkpoint, load_checkpoint, save_checkpoint
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_model(device, model, train_loader, val_loader, test_loader, denormalizer, num_epochs, use_wandb=False, log_dir="runs", checkpoint_dir="checkpoints", weight_decay=1e-5, patience=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=weight_decay)
    # ReduceLROnPlateau scheduler reduces the learning rate when a metric has stopped improving
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    # Set up TensorBoard writer or Weights & Biases logging
    if use_wandb:
        wandb.init(
            project="power-forecasting-lstm",
            config={
                "epochs": num_epochs,
                "batch_size": train_loader.batch_size,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "architecture": "BiLSTMWithFusion"
            },
            name=os.path.basename(checkpoint_dir),    
        )
        wandb.watch(model, log="all")
    else:
        writer = SummaryWriter(log_dir=log_dir)

    # Create checkpoint directory if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_epoch = 0
    patience_counter = 0
    best_val_loss = float('inf')

    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        start_epoch, best_val_loss = load_checkpoint(latest_checkpoint, model, optimizer)
    else:
        print("No checkpoint found, starting from scratch.")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader)
        print(f'*****{epoch=}******')
        for batch_idx, (X, Y, nwp_data) in enumerate(pbar):
            X, Y, nwp_data = X.to(device), Y.to(device), nwp_data.to(device)
            power_data = X.view(X.size(0), X.size(1), 1)
            
            optimizer.zero_grad()
            outputs = model(nwp_data, power_data)
            # see if view helps (https://datascience.stackexchange.com/questions/87630/pytorch-lstm-for-time-series-failing-to-learn)
            loss = criterion(denormalizer(outputs.view([outputs.size(0), outputs.size(1)])), denormalizer(Y))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_description(f"train loss={loss.item()}")
            if use_wandb:
                wandb.log({"train_loss": loss.item(), "epoch": epoch})
            else:
                writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + batch_idx)
        
        train_loss /= len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (X, Y, nwp_data) in enumerate(val_loader):
                X, Y, nwp_data = X.to(device), Y.to(device), nwp_data.to(device)
                power_data = X.view(X.size(0), X.size(1), 1)
                outputs = model(nwp_data, power_data)
                loss = criterion(denormalizer(outputs.squeeze(-1)), denormalizer(Y))
                val_loss += loss.item()
                
                if use_wandb:
                    wandb.log({"val_loss": loss.item(), "epoch": epoch})
                else:
                    writer.add_scalar("Loss/val", loss.item(), epoch * len(val_loader) + batch_idx)
        
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")
        
        # Adjust the learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save checkpoint if this epoch has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        
        if use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    
    best_epoch, _ = load_checkpoint(latest_checkpoint, model, optimizer)
    # Testing loop
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, (X, Y, nwp_data) in enumerate(test_loader):
            X, Y, nwp_data = X.to(device), Y.to(device), nwp_data.to(device)
            power_data = X.view(X.size(0), X.size(1), 1)
            outputs = model(nwp_data, power_data)
            loss = criterion(denormalizer(outputs.squeeze(-1)), denormalizer(Y))
            test_loss += loss.item()
            
            if use_wandb:
                wandb.log({"test_loss": loss.item()})
            else:
                writer.add_scalar("Loss/test", loss.item(), batch_idx)
    
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    
    if use_wandb:
        wandb.log({"final_test_loss": test_loss})
        wandb.log({"best_epoch": best_epoch})
        wandb.finish()
    else:
        writer.close()
    
    return model

def main(args):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data loaders
    train_loader, val_loader, test_loader, denormalizer = get_data_loaders_and_denormalizer(args.plant_number, args.batch_size)

    # Initialize model, criterion, and optimizer
    model = BiLSTMWithFusion().to(device)

    # Train the model
    train_model(0, model, train_loader, val_loader, test_loader, denormalizer, args.num_epochs, use_wandb=args.use_wandb, checkpoint_dir=args.checkpoint_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BiLSTM model for power forecasting")
    parser.add_argument("--plant_number", type=int, required=True, help="Power plant number to be used for training")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--hidden_size", type=int, default=100, help="Hidden size of the LSTM layers")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    
    args = parser.parse_args()
    main(args)