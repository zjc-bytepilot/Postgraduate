# tools/train.py
import yaml
from torch.utils.data import DataLoader
from openpan.registry import MODELS, LOSSES, DATASETS
from openpan.engine.trainer import Trainer

def main():
    # 1. 加载配置
    with open("configs/models/unsupervised_adaptive_pan.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
        
    # 2. 构建模型与损失函数
    model = MODELS.build(cfg['model'])
    loss_fn = LOSSES.build(cfg['loss'])
    
    # 3. 构建数据集 (训练集和验证集)
    train_dataset = DATASETS.build(cfg['dataset_train'])
    val_dataset = DATASETS.build(cfg['dataset_val'])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # 4. 启动训练引擎
    trainer = Trainer(model, loss_fn, train_loader, val_loader, cfg)
    trainer.train()

if __name__ == '__main__':
    main()