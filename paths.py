# Read only
import os


source_power_file = "/data1/yfliu/windpower_baseline/nmg_wf_history.csv"
source_nwp_dir = "/data1/yfliu/windpower_baseline/weather_history"
# Writable
train_power_file = "./inputs/train_nmg_wf_history.csv"
valid_power_file = "./inputs/valid_nmg_wf_history.csv"
test_power_file = "./inputs/test_nmg_wf_history.csv"
results_save_path = "/data1/yfliu/windpower_baseline/"
nwp_min_file = './inputs/nwp_min.npy'
nwp_max_file = './inputs/nwp_max.npy'


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