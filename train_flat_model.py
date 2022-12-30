import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision, torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os, sys
import time
import types 
import argparse 

import numpy as np
from nyuv2_data.nyuv2_dataloader_adashare import NYU_v2
from nyuv2_data.pixel2pixel_loss import NYUCriterions
from nyuv2_data.pixel2pixel_metrics import NYUMetrics

from models.resnet import deeplab_resnet34
from prune.local_prune import *
# from models.pruned_resnet import deeplab_pruned_resnet34
from models.flat_resnet import deeplab_pruned_flatresnet34

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--jobid', type=int, default=0)
parser.add_argument('--keep_ratio', type=float, default=0.1)
parser.add_argument('--branching', type=str, default='flat')
parser.add_argument('--logs_dir', type=str, default='assets/pth/flat_pruned_models/keep_0.1')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--testing', action='store_true')
args = parser.parse_args() 

def createdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created!')

def logprint(log, logs_fp=None):
    if not isinstance(logs_fp, type(None)):
        with open(logs_fp, 'a') as f:
            f.write(log + '\n')
    print(log)

def torch_save(obj, name, path):
    torch.save({name:obj}, path)
    print(f'{path} saved!')

def torch_load(name, path):
    ckpt = torch.load(path)
    return ckpt[name]

def load_weights_from_unbranched(model, unbranched_ckpt):
    new_ckpt = {}
    for k,v in unbranched_ckpt.items():
        if k in model.state_dict().keys():
            new_ckpt[k] = v 
        else:
            for i in range(28, 32):
                if f'{i}' in k:
                    new_k = k.replace(f'{i}', f'{i-28}')
            b1_key = new_k.replace('layers', 'branch_1')
            b2_key = new_k.replace('layers', 'branch_2')

            new_ckpt[b1_key] = v 
            new_ckpt[b2_key] = v 
    for k, v in model.state_dict().items():
        if 'branch_downsample' in k:
            new_ckpt[k] = v
    return new_ckpt

def hook_forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)

def train_mtl(model, train_dataloader, optimizer, scheduler, device, prune=False):
    model.train()  
    loss_list = {task: [] for task in tasks}
    for i, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        x = data['input'].to(device)
        output = model(x, prune)

        loss = 0
        for task in tasks:
            y = data[task].to(device)
            if task + '_mask' in data:
                tloss = criterion_dict[task](output[task], y, data[task + '_mask'].to(device))
            else:
                tloss = criterion_dict[task](output[task], y)
            loss_list[task].append(tloss.item())
            loss += tloss
        
        loss.backward()
        optimizer.step()

        if args.testing:
            break

    scheduler.step()
    for task in tasks:
        logprint('Task {} Train Loss: {:.4f}'.format(task[:4], np.mean(loss_list[task])), logs_fp)

def validate_mtl(model, val_dataloader, device, prune=False):
    model.eval()  
    loss_list = {task: [] for task in tasks}
    for i, data in enumerate(val_dataloader):
        x = data['input'].to(device)
        output = model(x, prune)
        for task in tasks:
            y = data[task].to(device)
            if task + '_mask' in data:
                loss = criterion_dict[task](output[task], y, data[task + '_mask'].to(device))
                metric_dict[task](output[task], y, data[task + '_mask'].to(device))
            else:
                loss = criterion_dict[task](output[task], y)
                metric_dict[task](output[task], y)
            loss_list[task].append(loss.item())
        
        if args.testing:
            break
    
    ret_results = {}
    for task in tasks:
        val_results = metric_dict[task].val_metrics()
        ret_results[task] = val_results.copy()
        logprint('Task {} Val Loss: {:.4f}'.format(task[:4], np.mean(loss_list[task])), logs_fp)
        logprint('{}'.format(val_results), logs_fp)
    return np.mean([np.mean(loss_list[task]) for task in tasks]), ret_results
    # return ret_results

keep_ratio = args.keep_ratio
# logs_dir = f'assets/ckpt/flat_resnet34_gated/keep_{keep_ratio}' 
logs_dir = f'assets/ckpt/models/resnet34/flat/local_pruning/training/run_1/keep_{keep_ratio}' 
logs_fp = os.path.join(logs_dir, 'logs.txt')
createdirs(logs_dir)

tasks = ['segment_semantic','normal','depth_zbuffer']
T = len(tasks)
cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}

batch_size = 16

dataroot = '/work/siddhantgarg_umass_edu/datasets/nyu_v2/nyu_v2' # change to your data root
dataset = NYU_v2(dataroot, 'train', crop_h=321, crop_w=321)
train_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=args.num_workers)

dataset = NYU_v2(dataroot, 'test', crop_h=321, crop_w=321)
test_loader = DataLoader(dataset, 8, shuffle=True, num_workers=args.num_workers)

criterion_dict = {}
metric_dict = {}
for task in tasks:
    criterion_dict[task] = NYUCriterions(task)
    metric_dict[task] = NYUMetrics(task)

config_dir = f'assets/ckpt/models/resnet34/flat/local_pruning/initializations'

base_config = [64] + [64]*3 + [128]*4 + [256]*6 + [512]*3
model = deeplab_pruned_flatresnet34(tasks, cls_num, base_config).to(device)

init_weights_path = os.path.join(config_dir, 'init_weights_base_config.pth')
init_weights = torch_load('init_weights', init_weights_path)
model.load_state_dict(init_weights)
print(f'init wts loaded from {init_weights_path}!')

mask_path = os.path.join(config_dir, f'keep_{keep_ratio}/total_model_mask.pth')
total_model_mask = torch_load('total_model_mask', mask_path)

for name, layer in model.named_modules():
    if name in total_model_mask.keys():
        layer.weight = torch.nn.Parameter(total_model_mask[name].clone())
        layer.weight.requires_grad = False 

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

save_name = os.path.join(logs_dir, 'pruned_model.pth')

T = 200
best_val_loss = 100.
for t in range(T):
    logprint('Epoch {}/{}'.format(t+1, T), logs_fp)
    train_mtl(model, train_loader, optimizer, scheduler, device)
    if (t+1) % 10 == 0 or args.testing:
        val, results = validate_mtl(model, test_loader, device)
        if val < best_val_loss:
            best_val_loss = val 
            torch.save(model.state_dict(), save_name)
        logprint('-'*100, logs_fp)
    if args.testing:
        break   

logprint('---- Validation ----', logs_fp)
model.load_state_dict(torch.load(save_name))
loss, results = validate_mtl(model, test_loader, device)
torch_save(results, 'results', os.path.join(logs_dir, 'results.pth'))

'''
python train_flat_model.py --keep_ratio=0.1 --testing
'''