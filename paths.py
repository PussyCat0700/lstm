# Read only
import os
import yaml


use_china = True
if use_china:
    cfg_filename = './conf/china.yaml'
else:
    cfg_filename = './conf/183.yaml'
with open(cfg_filename, 'r') as file:
    config = yaml.safe_load(file)

source_power_file = config['paths']['source_power_file']
source_nwp_dir = config['paths']['source_nwp_dir']
# writable
train_power_file = config['paths']['train_power_file']
valid_power_file = config['paths']['valid_power_file']
test_power_file = config['paths']['test_power_file']
results_save_path = config['paths']['results_save_path']
nwp_min_file = config['paths']['nwp_min_file']
nwp_max_file = config['paths']['nwp_max_file']
nwp_input_size = config['params']['nwp_input_size']  # NWP data has 16 features


class BaseSavePath:
    def __str__(self) -> str:
        return self.save_path
    
    def get_model_path(self):
        return os.path.join(self.save_path, 'model.ckpt')
    
    def get_png_path(self):
        return os.path.join(self.save_path, f'{self.plant_number}.png')
    
    def get_metrics_path(self):
        return os.path.join(self.save_path, 'metrics.csv')

class XGBSavePath(BaseSavePath):
    def __init__(self, plant_number) -> None:
        XGBOOST_SAVE_PATH = os.path.join(results_save_path, "xgboost")
        self.save_path = os.path.join(XGBOOST_SAVE_PATH, f'xgboost_{plant_number}')
        self.plant_number = plant_number
        os.makedirs(self.save_path, exist_ok=True)

class GPCFSavePath(BaseSavePath):
    def __init__(self, plant_number) -> None:
        GPCF_SAVE_PATH = os.path.join(results_save_path, "gpcf")
        self.save_path = os.path.join(GPCF_SAVE_PATH, f'gpcf_{plant_number}')
        self.plant_number = plant_number
        os.makedirs(self.save_path, exist_ok=True)