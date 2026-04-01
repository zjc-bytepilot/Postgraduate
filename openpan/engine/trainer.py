import os
import time
import logging
import torch
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, loss_fn, train_loader, val_loader, cfg):
        self.cfg = cfg['train']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 核心组件
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 2. 优化器设置
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.cfg['lr'], 
            weight_decay=self.cfg.get('weight_decay', 1e-4)
        )
        # 可以根据需要加入学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg['epochs'])

        # 3. 状态变量
        self.start_epoch = 0
        self.best_metric = float('inf') # 无监督损失越小越好；如果是有监督PSNR则是越大越好(float('-inf'))
        self.work_dir = self.cfg['work_dir']
        os.makedirs(self.work_dir, exist_ok=True)

        # 4. 初始化日志系统 (Logging & TensorBoard)
        self.logger = self._init_logger()
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.work_dir, 'tf_logs'))
        
        self.logger.info(f"🚀 Initialize Trainer on {self.device}")
        self.logger.info(f"📂 Work directory: {self.work_dir}")

        # 5. 断电恢复 (Resume)
        if self.cfg.get('resume_from') is not None:
            self.resume_checkpoint(self.cfg['resume_from'])

    def _init_logger(self):
        """初始化日志，同时输出到终端和本地文件"""
        logger = logging.getLogger('OpenPan')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 文件处理器
        file_handler = logging.FileHandler(os.path.join(self.work_dir, 'train.log'))
        file_handler.setFormatter(formatter)
        # 终端处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        return logger

    def save_checkpoint(self, epoch, is_best=False, filename=None):
        """保存 Checkpoint (包含模型、优化器、epoch等所有断电所需状态)"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric
        }
        
        if filename is None:
            filename = os.path.join(self.work_dir, 'latest.pth')
            
        torch.save(state, filename)
        
        if is_best:
            best_filename = os.path.join(self.work_dir, 'best.pth')
            torch.save(state, best_filename)
            self.logger.info(f"🌟 New best model saved at epoch {epoch}!")

    def resume_checkpoint(self, resume_path):
        """断电恢复功能"""
        if not os.path.isfile(resume_path):
            self.logger.error(f"❌ Checkpoint not found at: {resume_path}")
            return
            
        self.logger.info(f"🔄 Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        
        self.logger.info(f"✅ Successfully resumed. Starting from epoch {self.start_epoch}")

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for step, batch in enumerate(self.train_loader):
            ms = batch['ms'].to(self.device)
            pan = batch['pan'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播 (以无监督为例)
            pred_hrms = self.model(ms, pan)
            
            # 计算损失 (假设是之前的无监督Loss，需要ms_lr和pan_hr参与)
            # 在有监督网络中，这里就是 loss_fn(pred_hrms, gt)
            loss, loss_dict = self.loss_fn(pred_hrms, ms, pan)
            
            # 反向传播与优化
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # 定期刊印日志
            if (step + 1) % self.cfg['log_interval'] == 0:
                elapsed = time.time() - start_time
                self.logger.info(
                    f"Epoch [{epoch}/{self.cfg['epochs']}] "
                    f"Step [{step+1}/{len(self.train_loader)}] | "
                    f"Loss: {loss.item():.4f} | "
                    f"Time: {elapsed:.2f}s"
                )
                
                # 写入 TensorBoard
                global_step = epoch * len(self.train_loader) + step
                self.tb_writer.add_scalar('Train/Total_Loss', loss.item(), global_step)
                for k, v in loss_dict.items():
                    self.tb_writer.add_scalar(f'Train/{k}', v, global_step)
                
                start_time = time.time()

        self.scheduler.step()
        return epoch_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch):
        """验证循环"""
        self.model.eval()
        val_loss = 0.0
        
        for batch in self.val_loader:
            ms = batch['ms'].to(self.device)
            pan = batch['pan'].to(self.device)
            
            pred_hrms = self.model(ms, pan)
            loss, _ = self.loss_fn(pred_hrms, ms, pan)
            val_loss += loss.item()
            
        avg_val_loss = val_loss / len(self.val_loader)
        
        self.logger.info(f"📊 Validation Epoch {epoch} | Avg Loss: {avg_val_loss:.4f}")
        self.tb_writer.add_scalar('Val/Total_Loss', avg_val_loss, epoch)
        
        # 判断是否为最优模型
        is_best = False
        if avg_val_loss < self.best_metric: # 这里假设验证指标是 Loss (越小越好)
            self.best_metric = avg_val_loss
            is_best = True
            
        return is_best

    def train(self):
        """主训练循环"""
        self.logger.info("🔥 Start Training...")
        for epoch in range(self.start_epoch, self.cfg['epochs']):
            # 1. 训练一个 Epoch
            train_loss = self.train_epoch(epoch)
            
            # 2. 强制保存最新状态 (latest.pth，用于断电恢复)
            self.save_checkpoint(epoch, filename=os.path.join(self.work_dir, 'latest.pth'))
            
            # 3. 定期保存历史版本 (如 epoch_50.pth, epoch_100.pth)
            if (epoch + 1) % self.cfg['save_interval'] == 0:
                self.save_checkpoint(epoch, filename=os.path.join(self.work_dir, f'epoch_{epoch+1}.pth'))
                
            # 4. 验证并保存 Best 模型
            if (epoch + 1) % self.cfg['val_interval'] == 0:
                is_best = self.validate(epoch)
                if is_best:
                    self.save_checkpoint(epoch, is_best=True)
                    
        self.logger.info("🎉 Training Finished!")
        self.tb_writer.close()