import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision, torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import os, sys
import time
import types 

import numpy as np
from ptflops import get_model_complexity_info

from nyuv2_data.nyuv2_dataloader_adashare import NYU_v2
from nyuv2_data.pixel2pixel_loss import NYUCriterions
from nyuv2_data.pixel2pixel_metrics import NYUMetrics

from models.resnet import deeplab_resnet34

# from models.pruned_resnet import deeplab_pruned_resnet34
from models.flat_resnet import deeplab_pruned_flatresnet34

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def createdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created!')

def torch_save(obj, name, path):
    torch.save({name:obj}, path)
    print(f'{path} saved!')

def torch_load(name, path):
    ckpt = torch.load(path)
    return ckpt[name]

def filter_name(name):
    # if 'resnet' in name:
    #     if 'conv1' in name or 'conv2' in name:
    #         return True
    # else:
    #     return False

    if 'gate1' in name or 'gate2' in name:
        return True
    else:
        return False

def get_pruned_config(total_model_mask): # correct for Local Pruning 
    config = []
    i = 1
    for name, mask in total_model_mask.items():
        if filter_name(name):
            # print(i, name, 'mask:', int(torch.sum(mask).item()))
            i+=1
            config.append(int(torch.sum(mask).item()))
    # print(config)
    # sys.exit()
    return config

parser = argparse.ArgumentParser()

parser.add_argument('--save', action='store_true')
parser.add_argument('--pruning', type=str, default='local_pruning')
args = parser.parse_args() 

pruning = args.pruning 
if pruning == 'local_pruning':
    from prune.local_prune import *
else:
    from prune.global_pruning import * 

tasks = ['segment_semantic','normal','depth_zbuffer']
T = len(tasks)
cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}

batch_size = 16

dataroot = '/work/siddhantgarg_umass_edu/datasets/nyu_v2/nyu_v2' # change to your data root
dataset = NYU_v2(dataroot, 'train', crop_h=321, crop_w=321)
train_loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)

dataset = NYU_v2(dataroot, 'test', crop_h=321, crop_w=321)
test_loader = DataLoader(dataset, 8, shuffle=False, num_workers=0)

criterion_dict = {}
metric_dict = {}
for task in tasks:
    criterion_dict[task] = NYUCriterions(task)
    metric_dict[task] = NYUMetrics(task)

base_config = [64] + [64]*3 + [128]*4 + [256]*6 + [512]*3
model = deeplab_pruned_flatresnet34(tasks, cls_num, base_config).to(device)

# config_dir = f'assets/pth/flat_resnet_gated'
config_dir = f'assets/ckpt/models/resnet34/flat/{pruning}/initializations'
if not os.path.exists(config_dir):
    os.makedirs(config_dir)
    print(f'config_dir: {config_dir} created!')

'''
save init weights for the first time
'''
init_weights_path = os.path.join(config_dir, 'init_weights_base_config.pth')

if args.save:
    torch_save(model.state_dict(), 'init_weights', init_weights_path)
    print(f'init weights saved to {init_weights_path}')
'''
need to load init weights from the unbranched version and then prune
'''
init_weights = torch_load('init_weights', init_weights_path)
model.load_state_dict(init_weights)
print(f'init wts loaded from {init_weights_path}!')

total_path = os.path.join(config_dir, 'total_importance_scores.pth')
tasks_path = os.path.join(config_dir, 'tasks_importance_scores.pth')
if args.save:
    num_batches = 50
    total_importance_scores, tasks_importance_scores = get_task_total_importance_scores(model, tasks, train_loader, num_batches, criterion_dict)

    torch_save(total_importance_scores, 'total_importance_scores', total_path)
    torch_save(tasks_importance_scores, 'tasks_importance_scores', tasks_path)

total_importance_scores = torch_load('total_importance_scores', total_path)
tasks_importance_scores = torch_load('tasks_importance_scores', tasks_path)

if pruning == 'local_pruning':
    '''get sroccs between tasks'''
    tasks_masks, tasks_filter_ranks = get_all_task_masks(model, 1., tasks, tasks_importance_scores) # keep_ratio = 1.
    sroccs = get_2wise_srocc(tasks, tasks_filter_ranks)
    print(sroccs)
    if args.save:
        torch_save(sroccs, 'sroccs', os.path.join(config_dir, 'sroccs_flatresnet34_nyu.pth'))

if args.pruning == 'global_pruning':
    min_keep = 0.1
else:
    min_keep = 0.05
keep_ratios = np.arange(min_keep, 1, 0.05)

for keep_ratio in keep_ratios:
    keep_ratio = round(keep_ratio, 2)

    total_model_mask  = get_total_mask(model, keep_ratio, total_importance_scores)
    
    pruned_config = get_pruned_config(total_model_mask)
    print('pruned_config:', pruned_config, len(pruned_config))

    # pruned_model = deeplab_pruned_flatresnet34(tasks, cls_num, pruned_config).to(device)
    # # print(pruned_model.resnet)
    
    # macs, params = get_model_complexity_info(pruned_model, (3, 321, 321), print_per_layer_stat=False, verbose=False)
    # macs_params = {'macs' : macs, 'params' : params}
    # print('keep_ratio:', keep_ratio, macs_params)

    # pathdir = os.path.join(config_dir, f'keep_{keep_ratio}')
    # createdirs(pathdir)
    # torch_save(total_model_mask, 'total_model_mask', os.path.join(pathdir, 'total_model_mask.pth'))
    # torch_save(pruned_config, 'pruned_config', os.path.join(pathdir, 'pruned_config.pth'))
    # torch_save(macs_params, 'macs_params', os.path.join(pathdir, 'macs_params.pth'))
    # sys.exit()
