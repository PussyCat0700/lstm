# Read only
import os
import yaml
import pandas as pd

PLANTS = {
    "china": './conf/solar/china.yaml',
    "nmg": './conf/solar/nmg.yaml',
}

class LazyPathLoader:
    def __init__(self):
        self.plant_number = None
        self.plantnumdict = {}
    
    def init(self, months, plantset, plant_number):
        self.plant_number = plant_number
        self.plantset = plantset
        cfg_filename = PLANTS[self.plantset]
        self._load_cfg(cfg_filename)
        self.ablation_name = None
        if months != '12m':
            self.ablation_name = months
        # init paths
        df = pd.read_csv(self.config['paths']['source_power_stat'])
        for idx, row in df.iterrows():
            plant_no = row["PLANT_NO"]
            self.plantnumdict[idx] = int(plant_no)
        self.plant_id = self.plantnumdict[self.plant_number]
        self.paths = self._prep_paths()
    
    def _prep_paths(self):
        _paths = self._update_paths(self.config['paths'])
        self._create_directories(_paths)
        return _paths
    
    def check_exists(self):
        return os.path.exists(self.paths['source_power_file'])
    
    def _load_cfg(self, filename):
        # 读取 YAML 配置文件
        with open(filename, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def _update_paths(self, paths):
        paths = {key: value.format(plant_number=self.plant_id) for key, value in paths.items()}
        processed_dir = paths["processed_dir"]
        rel_dir = f"{self.plantset}/{self.plant_id}"
        if self.ablation_name is not None:
            rel_dir = f"ablation_{self.plantset}/{self.ablation_name}/{self.plant_id}"
        paths["train_power_file"] = os.path.join(processed_dir, f"{rel_dir}/train_china_{self.plantset}_solar_history.csv")
        paths["valid_power_file"] = os.path.join(processed_dir, f"{rel_dir}/valid_china_{self.plantset}_solar_history.csv")
        paths["test_power_file"] = os.path.join(processed_dir, f"{rel_dir}/test_china_{self.plantset}_solar_history.csv")
        paths["nwp_min_file"] = os.path.join(processed_dir, f"{rel_dir}/nwp_min.npy")
        paths["nwp_max_file"] = os.path.join(processed_dir, f"{rel_dir}/nwp_max.npy")
        return paths
    
    def _create_directories(self, paths):
        # 使用 os.makedirs 确保每个路径的目录存在
        for path in paths.values():
            dir_path = os.path.dirname(path)
            os.makedirs(dir_path, exist_ok=True)

    def get_run_path_status(self, modelname):
        """
        Returns:
            runpath, is it over with metrics.csv.
        """
        midname = f"runs_{modelname}"
        if self.ablation_name is not None:
            midname = f"ablation_{modelname}/{self.ablation_name}"
        runpath = f"{self.paths['results_save_path']}/{midname}/{modelname}_{self.plant_id}"
        os.makedirs(runpath, exist_ok=True)
        metric_dir = os.path.join(runpath, 'metrics.csv')
        return runpath, os.path.exists(metric_dir)

    @property
    def nwp_input_size(self):
        return self.config['params']['nwp_input_size']



# 创建懒加载路径加载器
path_loader = LazyPathLoader()
# 不受影响的变量
nwp_input_size = 27

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