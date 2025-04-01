import logging
import os
import torch
from torchvision import transforms
import numpy as np
import random
import cv2
from PIL import Image
import wandb


def path_to_image(path, size=(1024, 1024), color_type=['rgb', 'gray'][0]):
    if color_type.lower() == 'rgb':
        image = cv2.imread(path)
    elif color_type.lower() == 'gray':
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        print('Select the color_type to return, either to RGB or gray image.')
        return
    if size:
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    if color_type.lower() == 'rgb':
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB')
    else:
        image = Image.fromarray(image).convert('L')
    return image



def check_state_dict(state_dict, unwanted_prefixes=['module.', '_orig_mod.']):
    for k, v in list(state_dict.items()):
        prefix_length = 0
        for unwanted_prefix in unwanted_prefixes:
            if k[prefix_length:].startswith(unwanted_prefix):
                prefix_length += len(unwanted_prefix)
        state_dict[k[prefix_length:]] = state_dict.pop(k)
    return state_dict


def generate_smoothed_gt(gts):
    epsilon = 0.001
    new_gts = (1-epsilon)*gts+epsilon/2
    return new_gts


class Logger():
    def __init__(self, path="log.txt"):
        self.logger = logging.getLogger('BiRefNet')
        self.file_handler = logging.FileHandler(path, "w")
        self.stdout_handler = logging.StreamHandler()
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
    
    def info(self, txt):
        self.logger.info(txt)
    
    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()


class WandbLogger:
    """Logger for Weights & Biases"""
    def __init__(self, config, args):
        self.enabled = config.use_wandb
        if not self.enabled:
            return
        
        # Initialize W&B
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config={
                # Training hyperparameters
                "learning_rate": config.lr,
                "epochs": args.epochs,
                "batch_size": config.batch_size,
                "optimizer": config.optimizer,
                
                # Model hyperparameters
                "model": config.model,
                "backbone": config.bb,
                "task": config.task,
                "image_size": config.size if not config.dynamic_size else "dynamic",
                "grayscale_input": config.grayscale_input,
                
                # Other parameters
                "training_set": config.training_set,
                "compile": config.compile,
                "precision_high": config.precisionHigh,
                "distributed": args.dist,
                "accelerate": args.use_accelerate,
            }
        )
    
    def log(self, data, step=None):
        """Log metrics to W&B"""
        if not self.enabled:
            return
        wandb.log(data, step=step)
    
    def log_image(self, image_name, image, step=None):
        """Log an image to W&B"""
        if not self.enabled:
            return
        wandb.log({image_name: wandb.Image(image)}, step=step)
    
    def finish(self):
        """End the W&B run"""
        if not self.enabled:
            return
        wandb.finish()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, path, filename="latest.pth"):
    torch.save(state, os.path.join(path, filename))


def save_tensor_img(tenor_im, path):
    im = tenor_im.cpu().clone()
    im = im.squeeze(0)
    tensor2pil = transforms.ToPILImage()
    im = tensor2pil(im)
    im.save(path)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
