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
from prune.branched_local_prune import *
from models.branched_resnet_segmentation import deeplab_pruned_branchedresnet34

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

def get_pruned_config(total_model_mask):
    config = []
    for name, mask in total_model_mask.items():
        if 'gate2' in name:
            config.append(int(torch.sum(mask).item()))
    return config

def get_branched_init_wts_from_flat_weights(model, flat_ckpt, branch_point):

    branched_ckpt = model.state_dict()
    for bkey in branched_ckpt.keys():

        if bkey in flat_ckpt.keys():
            branched_ckpt[bkey] = flat_ckpt[bkey]

        if 'branch' in bkey:
            fkey = bkey.split('.')
            pblock = int(fkey[2]) + (branch_point // 2)
            fkey[2] = str(pblock)
            fkey[1] = 'blocks'
            fkey = '.'.join(fkey)
            branched_ckpt[bkey] = flat_ckpt[fkey]
    
    return branched_ckpt

parser = argparse.ArgumentParser()

parser.add_argument('--branch_point', type=int, default=20)
parser.add_argument('--save', action='store_true')
args = parser.parse_args() 

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
branch_point = args.branch_point
model = deeplab_pruned_branchedresnet34(tasks, cls_num, base_config, branch_point).to(device)

# config_dir = f'assets/pth/branched_resnet_gated_semantic_bp_{branch_point}/init_flat_to_branched'
config_dir = f'assets/ckpt/models/resnet34/branched/segmentation/branch_point_{branch_point}/local_prune_uniform/initializations'
if not os.path.exists(config_dir):
    os.makedirs(config_dir)
    print(f'config_dir: {config_dir} created!')

flat_init_wts_path = 'assets/ckpt/models/resnet34/flat/local_pruning/initializations/init_weights_base_config.pth'
flat_ckpt = torch_load('init_weights', flat_init_wts_path)

branched_init_weights = get_branched_init_wts_from_flat_weights(model, flat_ckpt, branch_point)

del flat_ckpt

'''save branched init weights for the first time'''
init_weights_path = os.path.join(config_dir, f'init_weights_branched_{args.branch_point}_config.pth')

if args.save:
    torch_save(model.state_dict(), 'init_branched_weights', init_weights_path)
    print(f'init weights saved to {init_weights_path}')

branched_init_weights = torch_load('init_branched_weights', init_weights_path)
model.load_state_dict(branched_init_weights)
print(f'init wts loaded from flat weights {init_weights_path}!')

# sys.exit()

total_path = os.path.join(config_dir, 'total_importance_scores.pth')
tasks_path = os.path.join(config_dir, 'tasks_importance_scores.pth')
if args.save:
    num_batches = 50
    # num_batches = 1
    total_importance_scores, tasks_importance_scores = get_task_total_importance_scores(model, tasks, train_loader, num_batches, criterion_dict)

    torch_save(total_importance_scores, 'total_importance_scores', total_path)
    torch_save(tasks_importance_scores, 'tasks_importance_scores', tasks_path)

total_importance_scores = torch_load('total_importance_scores', total_path)
tasks_importance_scores = torch_load('tasks_importance_scores', tasks_path)

# for k, v in total_importance_scores.items():
#     print(k)

'''get sroccs between tasks'''
tasks_masks, tasks_filter_ranks = get_all_task_masks(model, 1., tasks, tasks_importance_scores) # keep_ratio = 1.
sroccs = get_2wise_srocc(tasks, tasks_filter_ranks)
print(sroccs)
if args.save:
    torch_save(sroccs, 'sroccs', os.path.join(config_dir, 'sroccs_branched_resnet_nyu.pth'))

flat_config_dir = 'assets/ckpt/models/resnet34/flat/local_pruning/initializations'

flat_keep_ratios = np.arange(0.05, 1, 0.05)
flat_keep_ratios = [round(x, 2) for x in flat_keep_ratios]

baseline_macs = []
for keep_ratio in flat_keep_ratios:
    macs_path = os.path.join(flat_config_dir, f'keep_{keep_ratio}', 'macs_params.pth')
    macs_dict = torch_load('macs_params', macs_path)
    macs = macs_dict['macs']
    macs = float(macs.strip(' GMac'))
    baseline_macs.append(macs)

keep_ratios = np.arange(0.02, 1, 0.01)

for keep_ratio in keep_ratios:
    keep_ratio = round(keep_ratio, 2)
    # total_model_mask  = get_total_mask(model, keep_ratio, total_importance_scores)
    total_model_mask  = get_total_mask_uniform(model, keep_ratio, total_importance_scores)
    
    pruned_config = get_pruned_config(total_model_mask)
    # print('pruned_config:',pruned_config, len(pruned_config))

    pruned_model = deeplab_pruned_branchedresnet34(tasks, cls_num, pruned_config, branch_point)
    macs, params = get_model_complexity_info(pruned_model, (3, 321, 321), print_per_layer_stat=False, verbose=False)
    macs_params = {'macs' : macs, 'params' : params}

    macs = float(macs.strip(' GMac'))

    for fmac in baseline_macs:
        if macs - 0.05 <= fmac and fmac <= macs+0.05:
            print('keep_ratio:', keep_ratio, macs_params, 'baseline:', fmac)

            pathdir = os.path.join(config_dir, f'keep_{keep_ratio}')
            createdirs(pathdir)
            torch_save(total_model_mask, 'total_model_mask', os.path.join(pathdir, 'total_model_mask.pth'))
            torch_save(pruned_config, 'pruned_config', os.path.join(pathdir, 'pruned_config.pth'))
            torch_save(macs_params, 'macs_params', os.path.join(pathdir, 'macs_params.pth'))

            break
            
    # sys.exit()

    # masked_model = deeplab_pruned_branchedresnet34(tasks, cls_num, config, branch_point).to(device)
    # macs, params = get_model_complexity_info(masked_model, (3, 321, 321), print_per_layer_stat=False, verbose=False)
    # macs_params = {'macs' : macs, 'params' : params}
    # print(macs_params)
