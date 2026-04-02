import os
import time
import logging
import torch
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, loss_fn, train_loader, val_loader, cfg):
        self.cfg = cfg['train']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg.get('weight_decay', 1e-4)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg['epochs'])

        self.start_epoch = 0
        self.best_metric = float('inf') # 对应 Loss, 越小越好
        self.work_dir = self.cfg['work_dir']
        os.makedirs(self.work_dir, exist_ok=True)

        self.logger = self._init_logger()
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.work_dir, 'tf_logs'))
        
        self.logger.info(f"🚀 Initialize Trainer on {self.device}")
        
        if self.cfg.get('resume_from') is not None:
            self.resume_checkpoint(self.cfg['resume_from'])

    def _init_logger(self):
        logger = logging.getLogger('OpenPan')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(os.path.join(self.work_dir, 'train.log'))
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        return logger

    def save_checkpoint(self, epoch, is_best=False, filename=None):
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
        if not os.path.isfile(resume_path):
            self.logger.error(f"❌ Checkpoint not found: {resume_path}")
            return
            
        self.logger.info(f"🔄 Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint.get('best_metric', float('inf'))

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for step, batch in enumerate(self.train_loader):
            ms = batch['ms'].to(self.device)
            pan = batch['pan'].to(self.device)
            
            self.optimizer.zero_grad()
            
            pred_hrms = self.model(ms, pan)
            loss, loss_dict = self.loss_fn(pred_hrms, ms, pan)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # 🌟 改进点 1：将总 Loss 和所有各项子 Loss 拼接到一起打印
            if (step + 1) % self.cfg['log_interval'] == 0:
                elapsed = time.time() - start_time
                
                # 动态生成 Loss 打印字符串
                loss_str = f"Total: {loss.item():.4f}"
                for k, v in loss_dict.items():
                    loss_str += f" | {k}: {v:.4f}"
                    
                self.logger.info(
                    f"Epoch [{epoch}/{self.cfg['epochs']}] "
                    f"Step [{step+1}/{len(self.train_loader)}] | {loss_str} | Time: {elapsed:.2f}s"
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
        self.model.eval()
        val_loss = 0.0
        val_loss_dict = {} # 🌟 用于累加各项子 Loss
        
        for batch in self.val_loader:
            ms = batch['ms'].to(self.device)
            pan = batch['pan'].to(self.device)
            
            pred_hrms = self.model(ms, pan)
            loss, loss_dict = self.loss_fn(pred_hrms, ms, pan)
            
            val_loss += loss.item()
            # 累加各项子 Loss
            for k, v in loss_dict.items():
                val_loss_dict[k] = val_loss_dict.get(k, 0.0) + v
                
        # 计算平均值
        avg_val_loss = val_loss / len(self.val_loader)
        avg_loss_dict = {k: v / len(self.val_loader) for k, v in val_loss_dict.items()}
        
        # 🌟 改进点 2：在验证集打印时也显示各项子 Loss
        loss_str = f"Total: {avg_val_loss:.4f}"
        for k, v in avg_loss_dict.items():
            loss_str += f" | {k}: {v:.4f}"
            
        self.logger.info(f"📊 Validation Epoch {epoch} | {loss_str}")
        
        # 🌟 改进点 3：将验证集的各项子 Loss 也写入 TensorBoard
        self.tb_writer.add_scalar('Val/Total_Loss', avg_val_loss, epoch)
        for k, v in avg_loss_dict.items():
            self.tb_writer.add_scalar(f'Val/{k}', v, epoch)
        
        is_best = False
        if avg_val_loss < self.best_metric:
            self.best_metric = avg_val_loss
            is_best = True
        return is_best

    def train(self):
        self.logger.info("🔥 Start Training...")
        for epoch in range(self.start_epoch, self.cfg['epochs']):
            self.train_epoch(epoch)
            self.save_checkpoint(epoch, filename=os.path.join(self.work_dir, 'latest.pth'))
            
            if (epoch + 1) % self.cfg['save_interval'] == 0:
                self.save_checkpoint(epoch, filename=os.path.join(self.work_dir, f'epoch_{epoch+1}.pth'))
                
            if (epoch + 1) % self.cfg['val_interval'] == 0:
                is_best = self.validate(epoch)
                if is_best:
                    self.save_checkpoint(epoch, is_best=True)
                    
        self.logger.info("🎉 Training Finished!")
        self.tb_writer.close()