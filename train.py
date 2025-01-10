import csv
import os
import sys
import traceback
import numpy as np
from draw import plot_predictions_vs_ground_truth_vanilla
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataloading import get_latest_checkpoint, load_checkpoint, save_checkpoint
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from paths import KEY_CTX_COORDS, KEY_NORM_NWP, KEY_NORM_X, KEY_NORM_Y, KEY_REAL_Y, KEY_TIME_NWP_PE, KEY_TIME_X_PE, KEY_TIME_Y, KEY_TS_COORDS, PLANTS, path_loader
from utils import compute_all_metrics, get_model_and_loader, get_parameter_number
from constants import CROSS_VIVIT, model_type_dict
from pytorch_lightning import seed_everything


seed_everything(42)


def train_model(device, model, train_loader, val_loader, test_loader, denormalizer, args, log_dir="runs", weight_decay=1e-5, patience=3):
    num_epochs = args.num_epochs
    use_wandb=args.use_wandb
    checkpoint_dir=args.checkpoint_dir
    skip_model_selection = False
    crossvt = args.with_neighbor 
    # 如果真按1k epochs训练效果会更好，但是8分钟才训完一个站，太慢了。
    # if skip_model_selection:
    #     patience = 100000  # magic number: inf
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'mae':
        criterion = nn.L1Loss()
    else:
        raise NotImplementedError(f"{args.loss} not supported")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=weight_decay)
    # ReduceLROnPlateau scheduler reduces the learning rate when a metric has stopped improving
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    # Set up TensorBoard writer or Weights & Biases logging
    if use_wandb:
        wandb.init(
            project=f"power-forecasting-183",
            config={
                "epochs": num_epochs,
                "batch_size": train_loader.batch_size,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "architecture": args.model_type,
            },
            name=os.path.basename(checkpoint_dir),    
        )
        wandb.watch(model, log="all")
    else:
        writer = SummaryWriter(log_dir=log_dir)
    
    params_info = get_parameter_number(model)
    print(params_info)
    if use_wandb:
        wandb.log({"params": params_info['Trainable']})
    else:
        writer.add_scalar("params", params_info['Trainable'], 0) 

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
    def forward_model(batch, training:bool):
        REAL_Y = batch[KEY_REAL_Y].to(device)
        nwp_data = batch[KEY_NORM_NWP].to(device)
        if crossvt:
            history_x = batch[KEY_NORM_X].to(device).unsqueeze(-1)  # [B, T, C]
            coords_x = batch[KEY_TS_COORDS].to(device)  # [B, 2, 1, 1]
            coords_nwp = batch[KEY_CTX_COORDS].to(device)  # [B, 2, H, W]
            time_coords_ctx = batch[KEY_TIME_NWP_PE].to(device)  # [B, T, C, H, W]
            time_coords_ts = batch[KEY_TIME_X_PE].to(device)  # [B, T, C, H, W]
            outputs = model(
                ctx=nwp_data,
                ctx_coords=coords_nwp,
                ts=history_x,
                ts_coords=coords_x,
                time_coords_ctx=time_coords_ctx,
                time_coords_ts=time_coords_ts,
                mask=training,
            )
            outputs = outputs[0]
        else:
            outputs = model(nwp_data)
        outputs_denormalized = denormalizer(outputs)
        loss = criterion(outputs_denormalized, REAL_Y)
        return loss, outputs_denormalized
    # Training loop
    if not args.test:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            train_loss = 0.0
            pbar = tqdm(train_loader)
            print(f'*****{epoch=}******')
            for batch_idx, batch in enumerate(pbar):
                optimizer.zero_grad()
                loss = forward_model(batch, True)[0]
                if torch.isnan(loss):
                    print("NaN detected in training loss. Stopping training.")
                    with open(os.path.join(checkpoint_dir, "NAN_FOUND"), "w") as f:
                        f.write("NaN detected in training loss at batch index {}.".format(batch_idx))
                    exit()
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
                for batch_idx, batch in enumerate(val_loader):
                    loss = forward_model(batch, False)[0]
                    if torch.isnan(loss):
                        print("NaN detected in validation loss. Stopping training.")
                        with open(os.path.join(checkpoint_dir, "NAN_FOUND"), "w") as f:
                            f.write("NaN detected in validation loss at batch index {}.".format(batch_idx))
                        exit()
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
                patience_counter = 0
                print(f'{best_val_loss=}')
                for path in [f"checkpoint_epoch_{epoch+1}.pt", "checkpoint_best.pt"]:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                    }, os.path.join(checkpoint_dir, path))
            else:
                # otherwise you might get very high training loss. Our val set is too small.
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
            
            if use_wandb:
                wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        
        if skip_model_selection:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"))
            best_epoch = epoch
        else:
            latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
            best_epoch, _ = load_checkpoint(latest_checkpoint, model, optimizer)
    else:
        best_epoch = start_epoch

    print(f'testing on epoch {best_epoch}')
    # Testing loop
    model.eval()
    test_loss = 0.0
    all_outputs = []
    all_gts = []
    all_y_times = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            REAL_Y = batch[KEY_REAL_Y].to(device)
            TIME_Y = batch[KEY_TIME_Y]
            TIME_Y = np.array(TIME_Y).T.flatten()  # tackle the mysterious way torch dataloader handles list of string.
            all_y_times.append(TIME_Y)
            loss, outputs = forward_model(batch, False)
            test_loss += loss
            all_outputs.append(outputs.detach().cpu().numpy())
            all_gts.append(REAL_Y.detach().cpu().numpy())
            if use_wandb:
                wandb.log({"test_loss": loss.item()})
            else:
                writer.add_scalar("Loss/test", loss.item(), batch_idx)
    all_outputs = np.maximum(np.concatenate(all_outputs, axis=0).flatten(), 0)
    all_gts = np.maximum(np.concatenate(all_gts, axis=0).flatten(), 0)
    all_y_times = np.concatenate(all_y_times)
    filename = os.path.join(args.checkpoint_dir, f"{args.plant_number}.png")
    mae, mse = plot_predictions_vs_ground_truth_vanilla(all_outputs, all_gts, filename, all_y_times=all_y_times)
    print(mae)
    print(mse)
    all_metrics = compute_all_metrics(all_outputs, all_gts, denormalizer(1.0))
    def write_csv():
        csv_filename = os.path.join(checkpoint_dir, 'metrics.csv')
        with open(csv_filename, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            # 写入表头（字典的键）
            csv_writer.writerow(all_metrics.keys())
            # 写入内容（字典的值）
            csv_writer.writerow(all_metrics.values())
    write_csv()
    for key, metric in all_metrics.items():
        if use_wandb:
            wandb.log({f"test_{key}": metric})
        else:
            writer.add_scalar(f"{key}/test", metric, len(test_loader))
    test_loss /= len(test_loader)
    print(f"Test Loss (Batched): {test_loss:.4f}")
    if use_wandb:
        wandb.log({"test_mae": mae, "test_mse":mse})
        wandb.log({"final_test_loss": test_loss})
        wandb.log({"best_epoch": best_epoch})
        wandb.finish()
    else:
        writer.close()
    
    return model

def main(args):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, train_loader, val_loader, test_loader, denormalizer = get_model_and_loader(args, device)

    # Train the model
    train_model(0, model, train_loader, val_loader, test_loader, denormalizer, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BiLSTM model for power forecasting")
    parser.add_argument("model_type", type=int)
    parser.add_argument("--plant_number", type=int, required=True, help="Power plant number to be used for training")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--months", help="months used in training set.")
    parser.add_argument("--plant_set", choices=PLANTS.keys())
    parser.add_argument("--loss", choices=['mae', 'mse'], default='mse')
    parser.add_argument("--plant_type", type=int, choices=[0, 1], default=None, help="0 for windpower, 1 for solarpower.")
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()
    args.model_type = model_type_dict[args.model_type]
    args.with_neighbor = args.model_type == CROSS_VIVIT
    args.num_epochs = 30
    print(f'now training {args.model_type}')
    path_loader.init(args.months, args.plant_set, args.plant_number, args.plant_type)
    args.nwp_input_size = path_loader.nwp_input_size
    args.checkpoint_dir, is_done = path_loader.get_run_path_status(args.model_type)
    print(f"ckpt: {args.checkpoint_dir}")
    logger_file = os.path.join(args.checkpoint_dir, 'log.txt')
    with open(logger_file, 'w') as sys.stdout:
        if not path_loader.check_exists():
            print(f"{args.plant_number} does not have source input file")
            exit(0)
        if is_done:
            if args.test:
                files_to_check = [os.path.join(args.checkpoint_dir, x) for x in ['all_gts.npy', 'all_preds.npy']]
                files_to_check2 = os.path.join(args.checkpoint_dir, 'output.csv')
                if all([os.path.isfile(file) for file in files_to_check]) or os.path.isfile(files_to_check2):
                    print("test enabled but files are generated.")
                    exit(0)
            else:
                print(f"{args.plant_number} already has output metrics.csv at {args.checkpoint_dir}")
                exit(0)
        else:
            if args.test:
                print("Not done! No test!")
                exit(0)
        try:
            main(args)
        except Exception as e:
            with open(os.path.join(args.checkpoint_dir, 'log.err'), 'w') as error_file:
                error_file.write(f"Error occurred: {str(e)}\n")
                # Optionally, write the full traceback for debugging
                traceback.print_exc(file=error_file)