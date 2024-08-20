import os
from lstm_model import BiLSTMWithFusion
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataloading import get_data_loaders, get_latest_checkpoint, load_checkpoint, save_checkpoint
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm


def train_model(device, model, train_loader, val_loader, test_loader, num_epochs, criterion, optimizer, use_wandb=False, log_dir="runs", checkpoint_dir="checkpoints"):
    # Set up TensorBoard writer or Weights & Biases logging
    if use_wandb:
        wandb.init(project="power-forecasting-lstm", config={
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "architecture": "BiLSTMWithFusion"
        })
        wandb.watch(model, log="all")
    else:
        writer = SummaryWriter(log_dir=log_dir)

    # Create checkpoint directory if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_epoch = 0
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
            loss = criterion(outputs, Y)
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
                loss = criterion(outputs, Y)
                val_loss += loss.item()
                
                if use_wandb:
                    wandb.log({"val_loss": loss.item(), "epoch": epoch})
                else:
                    writer.add_scalar("Loss/val", loss.item(), epoch * len(val_loader) + batch_idx)
        
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint if this epoch has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"))
        
        if use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    
    # Testing loop
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, (X, Y, nwp_data) in enumerate(test_loader):
            X, Y, nwp_data = X.to(device), Y.to(device), nwp_data.to(device)
            power_data = X.view(X.size(0), X.size(1), 1)
            outputs = model(nwp_data, power_data)
            loss = criterion(outputs, Y)
            test_loss += loss.item()
            
            if use_wandb:
                wandb.log({"test_loss": loss.item()})
            else:
                writer.add_scalar("Loss/test", loss.item(), batch_idx)
    
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    
    if use_wandb:
        wandb.log({"final_test_loss": test_loss})
        wandb.finish()
    else:
        writer.close()
    
    return model

def main(args):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(args.plant_number, args.batch_size)

    # Initialize model, criterion, and optimizer
    model = BiLSTMWithFusion(nwp_input_size=16, power_input_size=1, hidden_size=args.hidden_size, output_size=96, num_layers=args.num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    train_model(0, model, train_loader, val_loader, test_loader, args.num_epochs, criterion, optimizer, use_wandb=args.use_wandb, checkpoint_dir=args.checkpoint_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BiLSTM model for power forecasting")
    parser.add_argument("--plant_number", type=int, required=True, help="Power plant number to be used for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--hidden_size", type=int, default=100, help="Hidden size of the LSTM layers")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    
    args = parser.parse_args()
    main(args)