import os
import yaml
import argparse
from torch.utils.data import DataLoader

# ==========================================
# 🌟 极其重要的一步：强制导入所有模块以激活 Registry 装饰器！
# 如果不导入，Registry 字典里就是空的，会报错 "Module not found"
# ==========================================
import openpan.datasets.h5_dataset
import openpan.models.backbones.dynamic_conv
import openpan.models.necks.sft_neck
import openpan.models.heads.dynamic_head
import openpan.models.losses.unsupervised_loss
import openpan.models.framework

from openpan.registry import MODELS, LOSSES, DATASETS
from openpan.engine.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train OpenPan Framework')
    parser.add_argument('--config', default='configs/models/unsupervised_dynamic_pan.yaml', help='train config file path')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 加载 YAML 配置
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    print(f"🚀 Loaded config from: {args.config}")
        
    # 2. 积木拼装：瞬间实例化模型、损失和数据集
    model = MODELS.build(cfg['model'])
    loss_fn = LOSSES.build(cfg['loss'])
    
    train_dataset = DATASETS.build(cfg['dataset_train'])
    val_dataset = DATASETS.build(cfg['dataset_val'])
    
    # 3. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['train']['batch_size'], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=2
    )
    
    print(f"📦 Training Dataset Size: {len(train_dataset)}")
    print(f"📦 Validation Dataset Size: {len(val_dataset)}")
    print(f"🧠 Model Architecture:\n{model}")
    
    # 4. 启动炼丹引擎
    trainer = Trainer(model, loss_fn, train_loader, val_loader, cfg)
    trainer.train()

if __name__ == '__main__':
    # 确保当前工作目录是项目根目录
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()