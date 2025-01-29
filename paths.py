# Read only
import os
import yaml
import pandas as pd

PLANTS = {
    "china": './conf/solar/china.yaml',
    "nmg": './conf/solar/nmg.yaml',
    "china_add": './conf/solar/china_add.yaml',
    "china_real": './conf/solar/china_real.yaml',
}
KEY_REAL_X = "real_x"
KEY_REAL_Y = "real_y"
KEY_NORM_X = "norm_x"
KEY_NORM_Y = "norm_y"
KEY_NORM_NWP = "norm_nwp"
KEY_TIME_NWP_PE = "time_coordinates_nwp"
KEY_TIME_X_PE = "time_coordinates_x"
KEY_CTX_COORDS = "spatial_coordinates"
KEY_TS_COORDS = "station_coords"
KEY_TIME_X = "time_x"
KEY_TIME_Y = "time_y"


def read_station_info(csv_file):
    df = pd.read_csv(csv_file, dtype={'PLANT_NO': str})
    return df


class LazyPathLoader:
    def __init__(self):
        # Subject to change considering different type_value.
        self.plant_number = None
        self.plantnumdict = {}
    
    def init(self, months, plantset, plant_number, type_value=None, period: int=None):
        self.period = period
        # TODO make this look more like code
        if period is not None and period > 24:
            self.period = None
        else:
            self.period = 24
        # End of TODO
        self.plant_number = plant_number
        self.plantset = plantset
        self.type_value = type_value
        cfg_filename = PLANTS[self.plantset]
        self._load_cfg(cfg_filename)
        self.ablation_name = None
        if months != '12m':
            self.ablation_name = months
        # init paths
        df = read_station_info(self.config['paths']['source_power_stat'])
        for idx, row in df.iterrows():
            plant_no = row["PLANT_NO"]
            plant_type = row.get("TYPE", None)
            if self.type_value is not None and plant_type != self.type_value:
                self.plantnumdict[idx] = None
                continue
            self.plantnumdict[idx] = plant_no
        self.plant_id = self.plantnumdict.get(self.plant_number, None)
        print(f'Specified plant # {self.plant_number}/{len(self.plantnumdict)} is actually {self.plant_id} officially.')
        if self.plant_number >= len(self.plantnumdict):
            print(f"Warning: {self.plant_number=} out of range for {len(self.plantnumdict)}.")
        if self.plant_id is not None:
            self.paths = self._prep_paths()
            self.meta = df.iloc[self.plant_number]
    
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
        # special case here: when only weather is different:
        chn = lambda s: s.replace('_real', '')
        rel_dir = f"{self.plantset}/{self.plant_id}"
        if self.ablation_name is not None:
            rel_dir = f"ablation_{self.plantset}/{self.ablation_name}/{self.plant_id}"
        paths["train_power_file"] = os.path.join(processed_dir, chn(f"{rel_dir}/train_china_{self.plantset}_solar_history.csv"))
        paths["valid_power_file"] = os.path.join(processed_dir, chn(f"{rel_dir}/valid_china_{self.plantset}_solar_history.csv"))
        paths["test_power_file"] = os.path.join(processed_dir, chn(f"{rel_dir}/test_china_{self.plantset}_solar_history.csv"))
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
        if self.period:
            midname += f'_{self.period}'
        if self.ablation_name is not None:
            midname = f"ablation_{modelname}/{self.ablation_name}"
            if self.period:
                midname += f'_{self.period}'
        runpath = f"{self.paths['results_save_path']}/{midname}/{modelname}_{self.plant_id}"
        os.makedirs(runpath, exist_ok=True)
        metric_filename = 'metrics.csv'
        if self.period:
            metric_filename = f'metric_{self.period}.csv'
        metric_dir = os.path.join(runpath, metric_filename)
        return runpath, os.path.exists(metric_dir)

    @property
    def nwp_input_size(self):
        return self.config['params']['nwp_input_size']
    
    @property
    def is_weather_real(self):
        return self.config['params']['real_weather']



# 创建懒加载路径加载器
path_loader = LazyPathLoader()

class BaseSavePath:
    def __str__(self) -> str:
        return self.save_path
    
    def get_model_path(self):
        return os.path.join(self.save_path, 'model.ckpt')
    
    def get_png_path(self):
        return os.path.join(self.save_path, f'{self.plant_number}.png')
    
    def get_metrics_path(self):
        return os.path.join(self.save_path, 'metrics.csv')

class GPCFSavePath(BaseSavePath):
    def __init__(self, plant_number) -> None:
        GPCF_SAVE_PATH = os.path.join(results_save_path, "gpcf")
        self.save_path = os.path.join(GPCF_SAVE_PATH, f'gpcf_{plant_number}')
        self.plant_number = plant_number
        os.makedirs(self.save_path, exist_ok=True)