import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from openpan.registry import DATASETS

@DATASETS.register_module()
class PansharpeningH5Dataset(Dataset):
    def __init__(self, data_path, max_value=2047.0, normalize_mode='0_1'):
        super().__init__()
        self.data_path = data_path
        self.max_value = float(max_value)
        self.normalize_mode = normalize_mode
        
        # 预先读取文件长度
        with h5py.File(self.data_path, 'r') as f:
            self.length = len(f['ms'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.data_path, 'r') as f:
            ms = f['ms'][idx]
            pan = f['pan'][idx]
            # 你的旧数据集可能有 gt，由于现在是无监督，我们可选加载
            gt = f['gt'][idx] if 'gt' in f else ms 

        # 归一化逻辑
        ms = ms.astype(np.float32) / self.max_value
        pan = pan.astype(np.float32) / self.max_value
        gt = gt.astype(np.float32) / self.max_value

        if self.normalize_mode == '-1_1':
            ms = ms * 2.0 - 1.0
            pan = pan * 2.0 - 1.0
            gt = gt * 2.0 - 1.0
            
        return {
            'ms': torch.from_numpy(ms).float(),
            'pan': torch.from_numpy(pan).float(),
            'gt': torch.from_numpy(gt).float()
        }