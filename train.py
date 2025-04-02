import os
import datetime
from contextlib import nullcontext
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
if tuple(map(int, torch.__version__.split('+')[0].split(".")[:3])) >= (2, 5, 0):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from config import Config
from loss import PixLoss, ClsLoss
from dataset import MyData
from models.birefnet import BiRefNet, BiRefNetC2F
from utils import Logger, AverageMeter, set_seed, check_state_dict, WandbLogger
from evaluation.metrics import SMeasure, MAEMeasure
import numpy as np
import cv2
from PIL import Image
import tempfile

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


parser = argparse.ArgumentParser(description='')
parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--ckpt_dir', default='ckpt/tmp', help='Temporary folder')
parser.add_argument('--dist', default=False, type=lambda x: x == 'True')
parser.add_argument('--use_accelerate', action='store_true', help='`accelerate launch --multi_gpu train.py --use_accelerate`. Use accelerate for training, good for FP16/BF16/...')
parser.add_argument('--wandb', default=True, type=lambda x: x == 'True', help='Enable Weights & Biases logging')
parser.add_argument('--wandb_project', default=None, type=str, help='W&B project name')
parser.add_argument('--wandb_entity', default=None, type=str, help='W&B entity (team) name')
args = parser.parse_args()

config = Config()

if args.use_accelerate:
    from accelerate import Accelerator, utils
    mixed_precision = ['no', 'fp16', 'bf16', 'fp8'][1]
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=1,
        kwargs_handlers=[
            utils.InitProcessGroupKwargs(backend="nccl", timeout=datetime.timedelta(seconds=3600*10)),
            utils.DistributedDataParallelKwargs(find_unused_parameters=True),
            utils.GradScalerKwargs(backoff_factor=0.5)],
    )
    args.dist = False

# DDP
to_be_distributed = args.dist
if to_be_distributed:
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*10))
    device = int(os.environ["LOCAL_RANK"])
else:
    if args.use_accelerate:
        device = accelerator.device
    else:
        device = config.device

if config.rand_seed:
    set_seed(config.rand_seed + device.index if isinstance(device, torch.device) else device)

epoch_st = 1
# make dir for ckpt
os.makedirs(args.ckpt_dir, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.ckpt_dir, "log.txt"))
logger_loss_idx = 1

# Initialize W&B logger
# Override config settings with command-line arguments if provided
if args.wandb is not None:
    config.use_wandb = args.wandb
if args.wandb_project is not None:
    config.wandb_project = args.wandb_project
if args.wandb_entity is not None:
    config.wandb_entity = args.wandb_entity

wandb_logger = WandbLogger(config, args)

# log model and optimizer params
# logger.info("Model details:"); logger.info(model)
# if args.use_accelerate and accelerator.mixed_precision != 'no':
#     config.compile = False
logger.info("datasets: load_all={}, compile={}.".format(config.load_all, config.compile))
logger.info("Other hyperparameters:"); logger.info(args)
print('batch size:', config.batch_size)

from dataset import custom_collate_fn

def prepare_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, to_be_distributed=False, is_train=True):
    # Prepare dataloaders
    if to_be_distributed:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(config.num_workers, batch_size), pin_memory=True,
            shuffle=False, sampler=DistributedSampler(dataset), drop_last=True, collate_fn=custom_collate_fn if is_train and config.dynamic_size else None
        )
    else:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(config.num_workers, batch_size), pin_memory=True,
            shuffle=is_train, sampler=None, drop_last=True, collate_fn=custom_collate_fn if is_train and config.dynamic_size else None
        )


def init_data_loaders(to_be_distributed):
    # Prepare datasets
    train_loader = prepare_dataloader(
        MyData(datasets=config.training_set, data_size=None if config.dynamic_size else config.size, is_train=True),
        config.batch_size, to_be_distributed=to_be_distributed, is_train=True
    )
    print(len(train_loader), "batches of train dataloader {} have been created.".format(config.training_set))
    
    # Create validation loader if validation is enabled
    val_loader = None
    if config.validate_during_training:
        val_loader = prepare_dataloader(
            MyData(datasets=config.validation_set, data_size=config.size, is_train=False),
            config.batch_size_valid, to_be_distributed=to_be_distributed, is_train=False
        )
        print(len(val_loader), "batches of validation dataloader {} have been created.".format(config.validation_set))
    
    return train_loader, val_loader


def init_models_optimizers(epochs, to_be_distributed):
    # Init models
    if config.model == 'BiRefNet':
        model = BiRefNet(bb_pretrained=True and not os.path.isfile(str(args.resume)))
    elif config.model == 'BiRefNetC2F':
        model = BiRefNetC2F(bb_pretrained=True and not os.path.isfile(str(args.resume)))
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            state_dict = torch.load(args.resume, map_location='cpu', weights_only=True)
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
            global epoch_st
            epoch_st = int(args.resume.rstrip('.pth').split('epoch_')[-1]) + 1
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    if not args.use_accelerate:
        if to_be_distributed:
            model = model.to(device)
            model = DDP(model, device_ids=[device])
        else:
            model = model.to(device)
    if config.compile:
        model = torch.compile(model, mode=['default', 'reduce-overhead', 'max-autotune'][0])
    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')

    # Setting optimizer
    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=1e-2)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[lde if lde > 0 else epochs + lde + 1 for lde in config.lr_decay_epochs],
        gamma=config.lr_decay_rate
    )
    # logger.info("Optimizer details:"); logger.info(optimizer)

    return model, optimizer, lr_scheduler


class Trainer:
    def __init__(
        self, data_loaders, model_opt_lrsch,
    ):
        self.model, self.optimizer, self.lr_scheduler = model_opt_lrsch
        self.train_loader, self.val_loader = data_loaders
        if args.use_accelerate:
            self.train_loader, self.model, self.optimizer = accelerator.prepare(self.train_loader, self.model, self.optimizer)
            if self.val_loader is not None:
                self.val_loader = accelerator.prepare(self.val_loader)
        if config.out_ref:
            self.criterion_gdt = nn.BCELoss()

        # Setting Losses
        self.pix_loss = PixLoss()
        self.cls_loss = ClsLoss()
        
        # Others
        self.loss_log = AverageMeter()
        
        # Validation metrics
        self.best_val_scores = {
            'S-measure': 0.0,  # Higher is better
            'MAE': float('inf'),  # Lower is better
        }
        self.best_epoch = {
            'S-measure': 0,
            'MAE': 0,
        }

    def _train_batch(self, batch):
        if args.use_accelerate:
            inputs = batch[0]#.to(device)
            gts = batch[1]#.to(device)
            class_labels = batch[2]#.to(device)
        else:
            inputs = batch[0].to(device)
            gts = batch[1].to(device)
            class_labels = batch[2].to(device)
        self.optimizer.zero_grad()
        scaled_preds, class_preds_lst = self.model(inputs)
        if config.out_ref:
            (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
            for _idx, (_gdt_pred, _gdt_label) in enumerate(zip(outs_gdt_pred, outs_gdt_label)):
                _gdt_pred = nn.functional.interpolate(_gdt_pred, size=_gdt_label.shape[2:], mode='bilinear', align_corners=True).sigmoid()
                _gdt_label = _gdt_label.sigmoid()
                loss_gdt = self.criterion_gdt(_gdt_pred, _gdt_label) if _idx == 0 else self.criterion_gdt(_gdt_pred, _gdt_label) + loss_gdt
            # self.loss_dict['loss_gdt'] = loss_gdt.item()
        if None in class_preds_lst:
            loss_cls = 0.
        else:
            loss_cls = self.cls_loss(class_preds_lst, class_labels) * 1.0
            self.loss_dict['loss_cls'] = loss_cls.item()

        # Loss
        loss_pix = self.pix_loss(scaled_preds, torch.clamp(gts, 0, 1)) * 1.0
        self.loss_dict['loss_pix'] = loss_pix.item()
        # since there may be several losses for sal, the lambdas for them (lambdas_pix) are inside the loss.py
        loss = loss_pix + loss_cls
        if config.out_ref:
            loss = loss + loss_gdt * 1.0

        self.loss_log.update(loss.item(), inputs.size(0))
        if args.use_accelerate:
            loss = loss / accelerator.gradient_accumulation_steps
            accelerator.backward(loss)
        else:
            loss.backward()
        self.optimizer.step()
        
        # We'll add image logging in a future update

    def train_epoch(self, epoch):
        global logger_loss_idx, wandb_logger
        self.model.train()
        self.loss_dict = {}
        if epoch > args.epochs + config.finetune_last_epochs:
            if config.task == 'Matting':
                self.pix_loss.lambdas_pix_last['mae'] *= 1
                self.pix_loss.lambdas_pix_last['mse'] *= 0.9
                self.pix_loss.lambdas_pix_last['ssim'] *= 0.9
            else:
                self.pix_loss.lambdas_pix_last['bce'] *= 0
                self.pix_loss.lambdas_pix_last['ssim'] *= 1
                self.pix_loss.lambdas_pix_last['iou'] *= 0.5
                self.pix_loss.lambdas_pix_last['mae'] *= 0.9

        for batch_idx, batch in enumerate(self.train_loader):
            # with nullcontext if not args.use_accelerate or accelerator.gradient_accumulation_steps <= 1 else accelerator.accumulate(self.model):
            self._train_batch(batch)
            
            # Calculate global step for logging
            global_step = (epoch - 1) * len(self.train_loader) + batch_idx
            
            # Log to W&B
            if batch_idx % 20 == 0:
                # Log metrics to W&B
                log_dict = {
                    'epoch': epoch,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/loss': self.loss_log.val,
                }
                # Add individual loss components
                for loss_name, loss_value in self.loss_dict.items():
                    log_dict[f'train/{loss_name}'] = loss_value
                
                wandb_logger.log(log_dict, step=global_step)
                
                # Console logging
                info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}].'.format(epoch, args.epochs, batch_idx, len(self.train_loader))
                info_loss = 'Training Losses'
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += ', {}: {:.3f}'.format(loss_name, loss_value)
                logger.info(' '.join((info_progress, info_loss)))
                
        # Log epoch summary
        wandb_logger.log({
            'epoch': epoch,
            'train/epoch_loss': self.loss_log.avg,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        })
        
        info_loss = '@==Final== Epoch[{0}/{1}]  Training Loss: {loss.avg:.3f}  '.format(epoch, args.epochs, loss=self.loss_log)
        logger.info(info_loss)

        self.lr_scheduler.step()
        return self.loss_log.avg
    
    def validate_epoch(self, epoch):
        """Run validation on the validation dataset"""
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        
        # Initialize metrics
        metrics = {}
        if 'S' in config.validation_metrics:
            metrics['S'] = SMeasure()
        if 'MAE' in config.validation_metrics:
            metrics['MAE'] = MAEMeasure()
        
        # Process validation data
        with torch.no_grad():
            for batch in self.val_loader:
                if args.use_accelerate:
                    inputs = batch[0]
                    gts = batch[1]
                else:
                    inputs = batch[0].to(device)
                    gts = batch[1].to(device)
                
                # Get predictions
                scaled_preds, _ = self.model(inputs)
                if isinstance(scaled_preds, tuple) and config.out_ref:
                    scaled_preds = scaled_preds[1]  # Extract actual predictions if output includes references
                
                # Get the final prediction
                preds = scaled_preds[-1].sigmoid()  # Final prediction
                
                # Calculate metrics for each image in the batch
                for i in range(preds.shape[0]):
                    # For multi-channel predictions, we'll evaluate each channel separately
                    # and use the first channel as the primary one for metrics
                    num_channels = preds.shape[1]
                    
                    if num_channels > 1:
                        # When we have multiple output channels, use the first channel for metrics
                        # This can be customized based on how you want to evaluate multi-channel predictions
                        pred = preds[i, 0].cpu().numpy() * 255  # Convert to numpy and scale to 0-255
                    else:
                        pred = preds[i, 0].cpu().numpy() * 255  # Convert to numpy and scale to 0-255
                    
                    # Use the appropriate ground truth channel
                    if gts.shape[1] > 1 and gts.shape[1] >= num_channels:
                        # If ground truth has multiple channels, use the corresponding one
                        gt = gts[i, 0].cpu().numpy() * 255
                    else:
                        gt = gts[i, 0].cpu().numpy() * 255
                    
                    # Apply metrics
                    for metric_name, metric in metrics.items():
                        metric.step(pred=pred, gt=gt)
        
        # Calculate final metrics
        results = {}
        for metric_name, metric in metrics.items():
            if metric_name == 'S':
                results['S-measure'] = metric.get_results()['sm']
            elif metric_name == 'MAE':
                results['MAE'] = metric.get_results()['mae']
        
        self.model.train()
        return results


def main():

    trainer = Trainer(
        data_loaders=init_data_loaders(to_be_distributed),
        model_opt_lrsch=init_models_optimizers(args.epochs, to_be_distributed)
    )

    for epoch in range(epoch_st, args.epochs+1):
        train_loss = trainer.train_epoch(epoch)
        
        # Run validation if enabled and it's time to validate
        if config.validate_during_training and trainer.val_loader is not None and epoch % config.validation_interval == 0:
            val_metrics = trainer.validate_epoch(epoch)
            
            # Log validation metrics to W&B
            val_log_dict = {'epoch': epoch}
            for metric_name, metric_value in val_metrics.items():
                val_log_dict[f'val/{metric_name}'] = metric_value
            wandb_logger.log(val_log_dict)
            
            # Console logging
            info_val = '@==Validation== Epoch[{0}/{1}] '.format(epoch, args.epochs)
            for metric_name, metric_value in val_metrics.items():
                info_val += '{}: {:.4f}  '.format(metric_name, metric_value)
            logger.info(info_val)
            
            # Save best model for each metric
            for metric_name, metric_value in val_metrics.items():
                if metric_name in trainer.best_val_scores:
                    is_better = False
                    if metric_name == 'MAE':  # Lower is better
                        if metric_value < trainer.best_val_scores[metric_name]:
                            is_better = True
                    else:  # Higher is better (S-measure)
                        if metric_value > trainer.best_val_scores[metric_name]:
                            is_better = True
                            
                    if is_better:
                        trainer.best_val_scores[metric_name] = metric_value
                        trainer.best_epoch[metric_name] = epoch
                        
                        # Save the best model
                        if args.use_accelerate:
                            if mixed_precision == 'fp16':
                                state_dict = {k: v.half() for k, v in trainer.model.state_dict().items()}
                        else:
                            state_dict = trainer.model.module.state_dict() if to_be_distributed else trainer.model.state_dict()
                        torch.save(state_dict, os.path.join(args.ckpt_dir, 'best_{}.pth'.format(metric_name)))
        
        # Save checkpoint
        # DDP
        if epoch >= args.epochs - config.save_last and epoch % config.save_step == 0:
            if args.use_accelerate:
                if mixed_precision == 'fp16':
                    state_dict = {k: v.half() for k, v in trainer.model.state_dict().items()}
            else:
                state_dict = trainer.model.module.state_dict() if to_be_distributed else trainer.model.state_dict()
            torch.save(state_dict, os.path.join(args.ckpt_dir, 'epoch_{}.pth'.format(epoch)))
    
    # Log best validation results
    if config.validate_during_training and trainer.val_loader is not None:
        logger.info('Best validation results:')
        for metric_name, best_score in trainer.best_val_scores.items():
            logger.info(f'Best {metric_name}: {best_score:.4f} (Epoch {trainer.best_epoch[metric_name]})')
            
        # Log best results to W&B
        best_results = {}
        for metric_name, best_score in trainer.best_val_scores.items():
            best_results[f'val/best_{metric_name}'] = best_score
            best_results[f'val/best_{metric_name}_epoch'] = trainer.best_epoch[metric_name]
        wandb_logger.log(best_results)
    
    # Close W&B logging
    wandb_logger.finish()
    
    if to_be_distributed:
        destroy_process_group()


if __name__ == '__main__':
    main()
