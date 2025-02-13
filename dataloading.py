import math
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from paths import KEY_CTX_COORDS, KEY_NORM_NWP, KEY_NORM_X, KEY_NORM_Y, KEY_REAL_X, KEY_REAL_Y, KEY_TIME_NWP_PE, KEY_TIME_X, KEY_TIME_X_PE, KEY_TIME_Y, KEY_TS_COORDS, path_loader


H, W = 8, 8


def get_time_pe(start_time:pd.Timestamp, periods, freq):
    # see CrossViVit/tscontext_dataset/TSContextDataset for details.
    # We are just borrowing their code here.
    time_utc = pd.date_range(start=start_time.strftime('%Y-%m-%d %H:%M:%S'), periods=periods, freq=freq, tz="UTC")
    months = torch.from_numpy(time_utc.month.values)[(...,) + (None,) * 3].repeat(1, 1, H, W)
    days = torch.from_numpy(time_utc.day.values)[(...,) + (None,) * 3].repeat(1, 1, H, W)
    hours = torch.from_numpy(time_utc.hour.values)[(...,) + (None,) * 3].repeat(1, 1, H, W)
    minutes = torch.from_numpy(time_utc.minute.values)[(...,) + (None,) * 3].repeat(1, 1, H, W)

    time_coords = torch.cat([months, days, hours, minutes], dim=1)  # [T, 4, 1, 1]
    return time_coords


class PowerPlantDataset(Dataset):
    def __init__(self, split, plant_number, power_minmax=None):
        """
        Args:
            csv_file (string): Path to the CSV file with power generation data.
            nwp_dir (string): Directory with NWP future 49-hour forecast .npy files.
            plant_number (int): Index of the power plant to be used for the dataset.
            power_minmax ([float, float]) Power min and power max for the given station only.
            If None, will be determined with current file.
        """
        if split == "train":
            csv_file = path_loader.paths['train_power_file']
        elif split == "valid":
            csv_file = path_loader.paths['valid_power_file']
        elif split == "test":
            csv_file = path_loader.paths['test_power_file']
        self.split = split
        self.data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        self.nwp_dir = path_loader.paths['source_nwp_dir']
        self.nwp_max_file = path_loader.paths['nwp_max_file']
        self.nwp_min_file = path_loader.paths['nwp_min_file']
        self.plant_number = plant_number
        if power_minmax is None:
            self.power_minmax = [
                max(0.0, self.data.min()[0]),
                self.data.max()[0]]
        else:
            self.power_minmax = power_minmax
        self.power_min = self.power_minmax[0]
        self.power_max = self.power_minmax[1]
        # Fit the weather scaler based on all weather data from all farms
        self.init_weather_minmax()
        print('done doing weather.')

    def normalize_power_data(self, data):
        return (data - self.power_min) / (self.power_max - self.power_min)
    
    def denormalize_power_data(self, data):
        return data * (self.power_max - self.power_min) + self.power_min

    def __len__(self):
        return len(self.data) - 96*2  # Each sample requires data from two consecutive days. 96 points for each day

    def _get_start_time(self, idx):
        start_time = self.data.index[idx].replace(minute=0, second=0, microsecond=0)
        return start_time
    
    def _get_nwp(self, nwp_time):
        if path_loader.is_weather_real:
            nwp_data = []
            for hour in range(0, 48):
                fixed_time = pd.to_datetime(nwp_time + pd.Timedelta(hours=hour))
                nwp_file = os.path.join(self.nwp_dir, f"{fixed_time.strftime('%Y-%m-%d_%H:%M:%S')}_{path_loader.plantnumdict[self.plant_number]}.npy")
                nwp_data.append(np.load(nwp_file))
            nwp_data_trunc = np.concatenate(nwp_data, axis=0).reshape(48, -1)
        else:
            fixed_times = pd.to_datetime([
                f"{nwp_time.strftime('%Y-%m-%d')} 00:00:00",
                f"{nwp_time.strftime('%Y-%m-%d')} 06:00:00",
                f"{nwp_time.strftime('%Y-%m-%d')} 12:00:00",
                f"{nwp_time.strftime('%Y-%m-%d')} 18:00:00"
            ])
            valid_times = [t for t in fixed_times if t <= nwp_time]
            closest_time = min(valid_times, key=lambda t: abs(t - nwp_time))
            nwp_file = os.path.join(self.nwp_dir, f"{closest_time.strftime('%Y-%m-%d_%H:%M:%S')}_{path_loader.plantnumdict[self.plant_number]}.npy")
            nwp_data = np.load(nwp_file)
            hours_diff = abs((closest_time - nwp_time).total_seconds()) // 3600
            nwp_data_trunc = nwp_data[int(hours_diff):int(hours_diff)+48]
        return nwp_data_trunc
    
    def _get_global_min_max_weather(self):
        weather_data_dir = self.nwp_dir
        if (not os.path.exists(self.nwp_max_file)) or (not os.path.join(self.nwp_min_file)): 
            print('doing nwp minmax')
            nan_count = 0
            valid_count = 0
            global_min = global_max = global_sum = None
            for x in os.listdir(weather_data_dir):
                # 加载当前.npy文件
                file_path = os.path.join(weather_data_dir, x)
                data = np.load(file_path)
                if global_max is None:
                    global_max = np.full((data.shape[-1]), -np.inf)
                if global_min is None:
                    global_min = np.full((data.shape[-1]), np.inf)
                if global_sum is None:
                    global_sum = np.full((data.shape[-1]), .0)
                # 计算每个变量的最小值和最大值
                local_max = np.max(data, axis=tuple(range(data.ndim - 1)))
                local_min = np.min(data, axis=tuple(range(data.ndim - 1)))
                local_avg = np.average(data, axis=tuple(range(data.ndim - 1)))
                if np.isnan(local_max).any() or np.isnan(local_min).any():
                    nan_count += 1
                    fill_value = global_sum / (valid_count+valid_count)
                    data[np.isnan(data)] = fill_value[np.isnan(data)]
                    np.save(file_path, data)
                    continue
                else:
                    global_sum += local_avg
                    valid_count += 1
                # 更新全局最大值和最小值
                global_max = np.maximum(global_max, local_max)
                global_min = np.minimum(global_min, local_min)

            # 保存最终结果
            np.save(self.nwp_max_file, global_max)
            np.save(self.nwp_min_file, global_min)
        else:
            print('loading nwp minmax')
            global_max = np.load(self.nwp_max_file)
            global_min = np.load(self.nwp_min_file)
        return global_min, global_max
    
    # Function to fit the MinMaxScaler on the weather data
    def init_weather_minmax(self):
        # 提取该场站的最大值和最小值
        global_min, global_max = self._get_global_min_max_weather()
        self.station_nwp_max = global_max  # (nwp_input_size,)
        self.station_nwp_min = global_min  # (nwp_input_size,)

    def __getitem__(self, idx):
        """
        1. csv不需要倒时差
        - 历史数据起始 终止
        day0					day1
        08:15:00【含】-> +96 08:00:00【含】
        - 预测数据起始 终止
        day1					day1			day2[00:00:00-23:45:00]
        +96 08:15:00【含】-> +15h45m(dropped)+24h(要的)时刻 【含】【要不就192个】
        评估：
        2. nwp的时差
        08:00:00->00:00:00 -8h
        """
        # Current day data
        start_time = self._get_start_time(idx)  # day0
        end_time = start_time + pd.DateOffset(hours=23, minutes=45)  # day1
        X = self.data.loc[start_time:end_time].iloc[:, 0].values
        X_norm = self.normalize_power_data(X)
        
        # Next day data
        # total span: 96
        next_start_time = start_time + pd.DateOffset(days=1) + pd.DateOffset(hours=15, minutes=45)  # day2 start 00:00:00
        next_end_time = next_start_time + pd.DateOffset(hours=23, minutes=45)  # day2 end 23:45:00
        Y = self.data.loc[next_start_time:next_end_time].iloc[:, 0].values
        Y_norm = self.normalize_power_data(Y)

        # Load the corresponding NWP data
        nwp_time = end_time - pd.DateOffset(hours=8)
        nwp_data = self._get_nwp(nwp_time)
        range_values = self.station_nwp_max - self.station_nwp_min
        nwp_data_scaled = np.zeros_like(nwp_data)
        epsilon = 1e-10
        for i in range(nwp_data.shape[-1]):
            if abs(range_values[i]) < epsilon:  # 判断是否接近于0
                nwp_data_scaled[..., i] = 1  # 归一化为常数1
            else:
                nwp_data_scaled[..., i] = (nwp_data[..., i] - self.station_nwp_min[i]) / range_values[i]
        time_nwp_pe = get_time_pe(end_time, 48, "1H")  # in 2 days into the future
        time_x_pe = get_time_pe(start_time, 48, "30T")  # in 1 day of the past
        time_x = self.data.loc[start_time:end_time].index.strftime('%Y-%m-%d %H:%M:%S').tolist()
        time_y = self.data.loc[next_start_time:next_end_time].index.strftime('%Y-%m-%d %H:%M:%S').tolist()
        return {
            KEY_REAL_X: torch.tensor(X, dtype=torch.float32),
            KEY_REAL_Y: torch.tensor(Y, dtype=torch.float32),
            KEY_NORM_X: torch.tensor(X_norm, dtype=torch.float32),
            KEY_NORM_Y: torch.tensor(Y_norm, dtype=torch.float32),
            KEY_NORM_NWP: torch.tensor(nwp_data_scaled, dtype=torch.float32),
            KEY_TIME_NWP_PE: time_nwp_pe,
            KEY_TIME_X_PE: time_x_pe,
            KEY_TIME_X: time_x,
            KEY_TIME_Y: time_y,
        }


class PowerPlantDatasetWithNeighbors(PowerPlantDataset):
    grid = torch.Tensor(np.array([[(x, y) for y in np.arange(54, 2.75, -0.25)] for x in np.arange(73, 136.25, 0.25)]))
    
    
    def __init__(self, split, plant_number, power_minmax=None, **kwargs):
        super().__init__(split, plant_number, power_minmax, **kwargs)
        self.coords = np.load(path_loader.paths['source_coords_file'])
    
    def get_coords_neighbors(self):
        
        weather_coords = PowerPlantDatasetWithNeighbors.grid[
            self.coords[self.plant_number, :, 0],
            self.coords[self.plant_number, :, 1],
            :
        ]  # [64, 2]
        weather_coords = weather_coords.reshape(H, W, 2).permute(2, 0, 1)  # [2, H, W]
        return weather_coords

    def get_coords_station(self):
        meta = path_loader.meta
        station_coords = torch.Tensor((meta['LONGITUDE'], meta['LATITUDE']))
        return station_coords.unsqueeze(-1).unsqueeze(-1)  # [2, 1, 1]
    
    def normalize_coords(self, nb_coords, st_coords):
        global_min = torch.min(nb_coords.min(), st_coords.min())
        global_max = torch.max(nb_coords.max(), st_coords.max())
        nb_coords_normalized = (nb_coords - global_min) / (global_max - global_min)
        st_coords_normalized = (st_coords - global_min) / (global_max - global_min)
        # Scale both tensors to [-1, 1]
        nb_coords_normalized = 2 * nb_coords_normalized - 1
        st_coords_normalized = 2 * st_coords_normalized - 1
        return nb_coords_normalized, st_coords_normalized
    
    def __getitem__(self, idx):
        ret = super().__getitem__(idx)
        ts_coords = self.get_coords_station()
        ctx_coords = self.get_coords_neighbors()
        ctx_coords_scaled, ts_coords_scaled = self.normalize_coords(ctx_coords, ts_coords)
        nwp_data = ret[KEY_NORM_NWP]
        nwp_data_withcoords = nwp_data.permute(0, 2, 1)  # [T, H*W, C] -> [T, C, H*W]
        nwp_data_withcoords = nwp_data_withcoords.reshape(*nwp_data_withcoords.shape[:2], H, W)
        ret[KEY_NORM_NWP] = nwp_data_withcoords  # [T, C, H, W]
        ret[KEY_CTX_COORDS] = ctx_coords_scaled
        ret[KEY_TS_COORDS] = ts_coords_scaled
        return ret
        

class PowerPlantDailyDataset(PowerPlantDatasetWithNeighbors):
    def __len__(self):
        return len(self.data) // 96 -  2  # Daily

    def _get_start_time(self, idx):
        offset = 1 + 4*8  # 08:15:00
        start_time = self.data.index[idx*96+offset].replace(second=0, microsecond=0)
        return start_time


class PowerPlantHourlyDataset(PowerPlantDatasetWithNeighbors):
    def __len__(self):
        return len(self.data) // 4 - (24+16+24)  # Hourly
    
    def _get_start_time(self, idx):
        offset = 1  # hh:15:00
        start_time = self.data.index[idx*4+offset].replace(second=0, microsecond=0)
        return start_time


class PowerPlantShortTermDataset(PowerPlantDataset):
    def __init__(self, split, plant_number, pred_span, power_minmax=None):
        super().__init__(split, plant_number, power_minmax)
        self.pred_span = pred_span
        self.outlen = pred_span*4
    
    def _could_pad_len(self, idx, x, pad_value=0):
        if len(x) < self.outlen:
            pad_len = self.outlen - len(x)
            print(f'padding {idx}th sample in {self.split}. length is {pad_len} matching length {self.outlen}')
            x = np.pad(x, (0, pad_len), mode='constant', constant_values=pad_value)
        return x

    def __getitem__(self, idx):
        """
        1. csv不需要倒时差
        - 历史数据起始 终止
        day0					day1
        08:15:00【含】-> +96 08:00:00【含】
        - 预测数据起始 终止
        day1					day1
        +96 08:15:00【含】-> +self.pred_span小时，如1小时就是09:00:00【含】
        评估：
        2. nwp的时差
        08:00:00->00:00:00 -8h
        """
        # Current day data
        start_time = self._get_start_time(idx)  # day0
        end_time = start_time + pd.DateOffset(hours=23, minutes=45)  # day1
        X = self.data.loc[start_time:end_time].iloc[:, 0].values
        X_norm = self.normalize_power_data(X)
        
        # Next day data
        # total span: 96
        next_start_time = end_time + pd.DateOffset(minutes=15)  # starting from 08:15:00
        next_end_time = end_time + pd.DateOffset(hours=self.pred_span)  # ending
        Y = self.data.loc[next_start_time:next_end_time].iloc[:, 0].values
        Y = self._could_pad_len(idx, Y)
        Y_norm = self.normalize_power_data(Y)

        # Load the corresponding NWP data
        nwp_time = end_time - pd.DateOffset(hours=8)
        nwp_data = self._get_nwp(nwp_time)
        range_values = self.station_nwp_max - self.station_nwp_min
        nwp_data_scaled = np.zeros_like(nwp_data)
        epsilon = 1e-10
        for i in range(nwp_data.shape[-1]):
            if abs(range_values[i]) < epsilon:  # 判断是否接近于0
                nwp_data_scaled[..., i] = 1  # 归一化为常数1
            else:
                nwp_data_scaled[..., i] = (nwp_data[..., i] - self.station_nwp_min[i]) / range_values[i]
        time_nwp_pe = get_time_pe(end_time, 48, "1H")  # in 2 days into the future
        time_x_pe = get_time_pe(start_time, 48, "30T")  # in 1 day of the past
        time_x = self.data.loc[start_time:end_time].index.strftime('%Y-%m-%d %H:%M:%S').tolist()
        time_y = self.data.loc[next_start_time:next_end_time].index.strftime('%Y-%m-%d %H:%M:%S')
        time_y = self._could_pad_len(idx, time_y, '').tolist()
        return {
            KEY_REAL_X: torch.tensor(X, dtype=torch.float32),
            KEY_REAL_Y: torch.tensor(Y, dtype=torch.float32),
            KEY_NORM_X: torch.tensor(X_norm, dtype=torch.float32),
            KEY_NORM_Y: torch.tensor(Y_norm, dtype=torch.float32),
            KEY_NORM_NWP: torch.tensor(nwp_data_scaled, dtype=torch.float32),
            KEY_TIME_NWP_PE: time_nwp_pe,
            KEY_TIME_X_PE: time_x_pe,
            KEY_TIME_X: time_x,
            KEY_TIME_Y: time_y,
        }


class PowerPlantShortTermDatasetWithNeighbors(PowerPlantShortTermDataset):
    grid = torch.Tensor(np.array([[(x, y) for y in np.arange(54, 2.75, -0.25)] for x in np.arange(73, 136.25, 0.25)]))
    
    
    def __init__(self, split, plant_number, pred_span:int, power_minmax=None):
        super().__init__(split, plant_number, pred_span, power_minmax)
        self.coords = np.load(path_loader.paths['source_coords_file'])
    
    def get_coords_neighbors(self):
        
        weather_coords = PowerPlantDatasetWithNeighbors.grid[
            self.coords[self.plant_number, :, 0],
            self.coords[self.plant_number, :, 1],
            :
        ]  # [64, 2]
        weather_coords = weather_coords.reshape(H, W, 2).permute(2, 0, 1)  # [2, H, W]
        return weather_coords

    def get_coords_station(self):
        meta = path_loader.meta
        station_coords = torch.Tensor((meta['LONGITUDE'], meta['LATITUDE']))
        return station_coords.unsqueeze(-1).unsqueeze(-1)  # [2, 1, 1]
    
    def normalize_coords(self, nb_coords, st_coords):
        global_min = torch.min(nb_coords.min(), st_coords.min())
        global_max = torch.max(nb_coords.max(), st_coords.max())
        nb_coords_normalized = (nb_coords - global_min) / (global_max - global_min)
        st_coords_normalized = (st_coords - global_min) / (global_max - global_min)
        # Scale both tensors to [-1, 1]
        nb_coords_normalized = 2 * nb_coords_normalized - 1
        st_coords_normalized = 2 * st_coords_normalized - 1
        return nb_coords_normalized, st_coords_normalized
    
    def __getitem__(self, idx):
        ret = super().__getitem__(idx)
        ts_coords = self.get_coords_station()
        ctx_coords = self.get_coords_neighbors()
        ctx_coords_scaled, ts_coords_scaled = self.normalize_coords(ctx_coords, ts_coords)
        nwp_data = ret[KEY_NORM_NWP]
        nwp_data_withcoords = nwp_data.permute(0, 2, 1)  # [T, H*W, C] -> [T, C, H*W]
        nwp_data_withcoords = nwp_data_withcoords.reshape(*nwp_data_withcoords.shape[:2], H, W)
        ret[KEY_NORM_NWP] = nwp_data_withcoords  # [T, C, H, W]
        ret[KEY_CTX_COORDS] = ctx_coords_scaled
        ret[KEY_TS_COORDS] = ts_coords_scaled
        return ret


class PowerPlantShortTermPeriodlyDataset(PowerPlantShortTermDatasetWithNeighbors):
    def __len__(self):
        return math.floor((len(self.data) // 4 - 24) / self.pred_span)
    
    def _get_start_time(self, idx):
        offset = 1  # hh:15:00
        start_time = self.data.index[idx*4*self.pred_span+offset].replace(second=0, microsecond=0)
        return start_time


class PowerPlantShortTermHourlyDataset(PowerPlantShortTermDatasetWithNeighbors):
    def __len__(self):
        return len(self.data) // 4 - 24 - 24  # extra 24 evading 1800 offset
    
    def _get_start_time(self, idx):
        offset = 1  # hh:15:00
        start_time = self.data.index[idx*4+offset].replace(second=0, microsecond=0)
        return start_time


class PowerPlantSklearnHourlyDataset(PowerPlantHourlyDataset):
    """小时对小时的数据集
    当前整点数据对应40小时后的整点数据
    """
    def __init__(self, split, plant_number, pred_span=40, power_minmax=None):
        super().__init__(split, plant_number, power_minmax)
        self.pred_span = pred_span
        self.offset = 8+6 if self.split == 'train' else 0
    
    def __len__(self):
        if self.pred_span > 24:
            length = len(self.data) // 4 - (self.offset+16+24)  # Hourly
        else:
            length = len(self.data) // 4 - (self.offset+self.pred_span)  # Hourly
        return length
    
    def _get_start_time(self, idx):
        start_time = self.data.index[4*(idx+self.offset)].replace(second=0, microsecond=0)
        return start_time
    
    def __getitem__(self, idx):
        """
        1. csv不需要倒时差
        - 当前时间day0 08:00:00-08:45:00
        - 预测目标day2 00:00:00-00:45:00 (in 40h scenario)
        评估：
        2. nwp的时差
        08:00:00->00:00:00 -8h
        3. nwp到csv预测值需要偏移：40h (by default)
        
        returns:
        X_norm/Y_norm: a single digit
        nwp_data_scaled: shaped (nwp_input_size,)
        """
        # Current day data
        x_start_time = self._get_start_time(idx)  # start
        x_end_time = x_start_time + pd.DateOffset(minutes=45)
        X = self.data.loc[x_start_time:x_end_time].iloc[:, 0].values
        X_norm = self.normalize_power_data(X)
        
        # Next day data
        y_start_time = x_start_time + pd.DateOffset(hours=self.pred_span)
        y_end_time = y_start_time + pd.DateOffset(minutes=45)
        Y = self.data.loc[y_start_time:y_end_time].iloc[:, 0].values
        Y_norm = self.normalize_power_data(Y)

        # Load the corresponding NWP data
        nwp_time = x_end_time - pd.DateOffset(hours=8)
        nwp_data = self._get_nwp(nwp_time)
        range_values = self.station_nwp_max - self.station_nwp_min
        nwp_data_scaled = np.zeros_like(nwp_data)
        epsilon = 1e-10
        for i in range(nwp_data.shape[-1]):
            if abs(range_values[i]) < epsilon:  # 判断是否接近于0
                nwp_data_scaled[..., i] = 1  # 归一化为常数1
            else:
                nwp_data_scaled[..., i] = (nwp_data[..., i] - self.station_nwp_min[i]) / range_values[i]
        nwp_data_scaled = nwp_data_scaled[self.pred_span-1]  # only nwp_input_dim is left
        time_x = self.data.loc[x_start_time:x_end_time].index.strftime('%Y-%m-%d %H:%M:%S')
        time_y = self.data.loc[y_start_time:y_end_time].index.strftime('%Y-%m-%d %H:%M:%S')
        return {
            KEY_REAL_X: torch.tensor(X, dtype=torch.float32),
            KEY_REAL_Y: torch.tensor(Y, dtype=torch.float32),
            KEY_NORM_X: torch.tensor(X_norm, dtype=torch.float32),
            KEY_NORM_Y: torch.tensor(Y_norm, dtype=torch.float32),
            KEY_NORM_NWP: torch.tensor(nwp_data_scaled, dtype=torch.float32),
            KEY_TIME_X: time_x,
            KEY_TIME_Y: time_y,
        }


import csv

def convert_torch_dataset_to_csv(dataset, folder_path, postfix="", force_data=False):
    # 创建文件夹（如果不存在）
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 定义保存的文件路径
    X_file = os.path.join(folder_path, f"X_norm{postfix}.csv")
    Y_file = os.path.join(folder_path, f"Y_norm{postfix}.csv")
    X_file_real = os.path.join(folder_path, f"X_real{postfix}.csv")
    Y_file_real = os.path.join(folder_path, f"Y_real{postfix}.csv")
    time_csv = os.path.join(folder_path, f"time{postfix}.csv")
    nwp_file = os.path.join(folder_path, f"nwp_data_scaled{postfix}.csv")
    ready_sign_path = os.path.join(folder_path, f"READY{postfix}")
    ready_time_sign_path = os.path.join(folder_path, f"READY(TIME){postfix}")
    is_ready = os.path.exists(ready_time_sign_path) and os.path.exists(ready_sign_path) and (not force_data)
    if not is_ready:
        print(f"Saving dataset to CSV in {folder_path}...")

        # 打开文件，以写入模式逐步保存数据
        with open(X_file, 'w', newline='') as f_X, \
            open(Y_file, 'w', newline='') as f_Y, \
            open(X_file_real, 'w', newline='') as f_X_real, \
            open(Y_file_real, 'w', newline='') as f_Y_real, \
            open(nwp_file, 'w', newline='') as f_nwp, \
            open(time_csv, 'w', newline='') as f_time:

            # 创建csv writer对象
            writer_X = csv.writer(f_X)
            writer_Y = csv.writer(f_Y)
            writer_X_real = csv.writer(f_X_real)
            writer_Y_real = csv.writer(f_Y_real)
            writer_nwp = csv.writer(f_nwp)
            writer_time = csv.writer(f_time)
            pbar = tqdm(range(len(dataset)))
            for i in pbar:
                item = dataset[i]
                # 将每个样本写入csv文件
                writer_X.writerow(item[KEY_NORM_X].tolist())        # 保存 X_norm
                writer_Y.writerow(item[KEY_NORM_Y].tolist())        # 保存 Y_norm
                writer_X_real.writerow(item[KEY_REAL_X].tolist())        # 保存 X_norm
                writer_Y_real.writerow(item[KEY_REAL_Y].tolist())        # 保存 Y_norm
                writer_nwp.writerow(item[KEY_NORM_NWP].tolist())  # 保存nwp_data_scaled展平为一行
                writer_time.writerow(np.array(item[KEY_TIME_Y]).T.flatten())
        with open(ready_sign_path, "w") as f:
            f.write("")
        with open(ready_time_sign_path, 'w') as f:
            f.write("")
                
    return load_csv_data(X_file, Y_file, X_file_real, Y_file_real, nwp_file, time_csv)


def load_csv_data(X_file, Y_file, X_file_real, Y_file_real, nwp_file, time_csv):
    # 加载并转换为numpy数组
    X = np.loadtxt(X_file, delimiter=',')
    Y = np.loadtxt(Y_file, delimiter=',')
    X_real = np.loadtxt(X_file_real, delimiter=',')
    Y_real = np.loadtxt(Y_file_real, delimiter=',')
    nwp = np.loadtxt(nwp_file, delimiter=',').reshape(-1, path_loader.nwp_input_size)  # 恢复原来的形状
    times = []
    with open(time_csv, 'r') as fi:
        for line in fi.readlines():
            times.extend([x.strip() for x in line.split(',')])
    return {
        KEY_REAL_X: X_real,
        KEY_REAL_Y: Y_real,
        KEY_NORM_X: X,
        KEY_NORM_Y: Y,
        KEY_NORM_NWP: nwp,
        KEY_TIME_Y: times,
    }

def get_dataset_and_denormalizer_sklearn(plant_number, split, folder_path, period:int, force_data=False):
    dataset = PowerPlantSklearnHourlyDataset(split, plant_number, period)
    postfix = ""
    if period <= 24:
        postfix = f"_{period}h"
    data = convert_torch_dataset_to_csv(dataset, os.path.join(folder_path, split), postfix, force_data)
    return data, dataset.denormalize_power_data

def get_data_loaders_and_denormalizer(plant_number, batch_size, period:int):
    test_loaders = {}
    if period > 24:
        train_dataset = PowerPlantHourlyDataset("train", plant_number)
        power_minmax = train_dataset.power_minmax
        valid_dataset = PowerPlantHourlyDataset("valid", plant_number, power_minmax)
        test_dataset = PowerPlantDailyDataset("test", plant_number, power_minmax)
        test_loaders = {40: DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=False)}
    else:
        # We only consider when period=24 for the moment
        train_dataset = PowerPlantShortTermHourlyDataset("train", plant_number, period)
        power_minmax = train_dataset.power_minmax
        valid_dataset = PowerPlantShortTermHourlyDataset("valid", plant_number, period, power_minmax)
        for period in [1, 4, 24]:
            test_dataset = PowerPlantShortTermPeriodlyDataset("test", plant_number, period, power_minmax)
            test_loaders.update({
                period: DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=False),
            })

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=1, shuffle=False)

    return train_loader, val_loader, test_loaders, train_dataset.denormalize_power_data


def save_checkpoint(state, filename):
    torch.save(state, filename)


def get_latest_checkpoint(checkpoint_dir, load_best=True):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not checkpoints:
        return None
    best_ckpt = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
    load_best = load_best and os.path.isfile(best_ckpt)
    if load_best:
        return best_ckpt
    else:
        checkpoints.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]), reverse=True)
        return os.path.join(checkpoint_dir, checkpoints[0])


def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint.get('loss', None)
    return start_epoch, loss


if __name__ == '__main__':
    plant_number = 298
    bs = 2
    path_loader.init('12m', 'china', plant_number)
    for period in [1, 4, 24, 40]:
        print(f'{period=}')
        data, _ = get_dataset_and_denormalizer_sklearn(plant_number, "valid", "here", period)
        assert len(data[KEY_TIME_Y]) == len(data[KEY_NORM_Y].flatten())
    train_loader, val_loader, test_loaders, denormalizer = get_data_loaders_and_denormalizer(plant_number, bs, 24)
    for period, test_loader in test_loaders.items(): 
        outlen = period*4
        print(f"testing {period=}")
        for i, batch in enumerate(test_loader):
            assert len(batch[KEY_NORM_X][bs-1]) == 96 and len(batch[KEY_NORM_Y][bs-1]) == outlen, i
        print(batch[KEY_TIME_X][-1], batch[KEY_TIME_Y][-1])