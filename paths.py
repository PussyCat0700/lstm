# Read only
import os
import yaml

class LazyPathLoader:
    def __init__(self):
        self.plant_number = None
        self._paths = None
        self._load_cfg()
    
    @property
    def paths(self):
        self._load_cfg()
        self._paths = self._update_paths(self.config['paths'])
        self._create_directories(self._paths)
        return self._paths
    
    def check_exists(self):
        return os.path.exists(self.paths['source_power_file'])
    
    def _load_cfg(self):
        # 读取 YAML 配置文件
        with open('./conf/solar/nmg.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
    
    def _update_paths(self, paths):
        return {key: value.format(plant_number=self.plant_number) for key, value in paths.items()}
    
    def _create_directories(self, paths):
        # 使用 os.makedirs 确保每个路径的目录存在
        for path in paths.values():
            dir_path = os.path.dirname(path)
            os.makedirs(dir_path, exist_ok=True)

    @property
    def nwp_input_size(self):
        return self.config['params']['nwp_input_size']



# 创建懒加载路径加载器
path_loader = LazyPathLoader()
# 不受影响的变量
nwp_input_size = path_loader.nwp_input_size

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