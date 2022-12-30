import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision, torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import os, sys
import time

import numpy as np
from scipy import stats
from nyuv2_data.nyuv2_dataloader_adashare import NYU_v2
from nyuv2_data.pixel2pixel_loss import NYUCriterions
from nyuv2_data.pixel2pixel_metrics import NYUMetrics

from models.resnet import deeplab_resnet34

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def torch_save(obj, name, path):
    torch.save({name:obj}, path)
    print(f'{path} saved!')

def torch_load(name, path):
    ckpt = torch.load(path)
    return ckpt[name]

def get_pruning_gates_list(model):

    pruning_gates_list = list()
    backbone_conv_params_count = 0

    for midx, m in enumerate(model.modules()):
        if hasattr(m, 'do_not_update'):
            print('do_not_update', midx, m)
            pruning_gates_list.append(m.weight)
            backbone_conv_params_count += torch.numel(m.weight)

    return pruning_gates_list, backbone_conv_params_count

def get_flat_to_unit_idx_map(pruning_gates_list):
    filter_to_gate_layer = {}
    filter_to_gate_unit_index = {}
    nfilters = 0
    for layer in range(len(pruning_gates_list)):
        for unit in range(len(pruning_gates_list[layer])):
            filter_to_gate_layer[nfilters] = layer 
            filter_to_gate_unit_index[nfilters] = unit 
            nfilters += 1
    return filter_to_gate_layer, filter_to_gate_unit_index

# '''local pruning'''
def get_task_importance_scores_per_layer(model):
    task_importance_score = {}
    for name, layer in model.named_modules():
        if 'gate' in name:
            task_importance_score[name] = (layer.weight * layer.weight.grad).data.pow(2)
    return task_importance_score

# def get_importance_scores(model, tasks, train_loader, num_batches, criterion_dict):

#     importance_scores = {}
#     for task in tasks:
#         importance_scores[task] = {}
#     for name, layer in model.named_modules():
#         if isinstance(layer, nn.Conv2d) and 'resnet' in name and 'downsample' not in name:
#             for task in tasks:
#                 importance_scores[task][name] = torch.zeros(layer.weight.shape[0]).to(device)

#     train_iter = iter(train_loader)
#     for i in range(num_batches):
#         print(f'{i}/{num_batches}')
#         data = next(train_iter)
#         x = data['input'].to(device)
#         output = model(x)

#         objectives = {}
#         for task in tasks:
#             y = data[task].to(device)
#             task_loss = criterion_dict[task](output[task], y)
#             objectives[task] = task_loss
    
#         for task in tasks:
#             task_loss = objectives[task]
#             task_loss.backward(retain_graph=True)
#             task_imp_scores = get_task_importance_scores_per_layer(model)
#             for name, layer in model.named_modules():
#                 if isinstance(layer, nn.Conv2d) and 'resnet' in name and 'downsample' not in name:
#                     importance_scores[task][name] += task_imp_scores[name]

#     return importance_scores

# def get_total_importance_scores(model, tasks, train_loader, num_batches, criterion_dict):

#     importance_scores = {}
#     for name, layer in model.named_modules():
#         if isinstance(layer, nn.Conv2d) and 'resnet' in name and 'downsample' not in name:
#             importance_scores[name] = torch.zeros(layer.weight.shape[0]).to(device)

#     train_iter = iter(train_loader)
#     for i in range(num_batches):
#         print(f'{i}/{num_batches}')
#         data = next(train_iter)
#         x = data['input'].to(device)
#         output = model(x)

#         objectives = {}
#         total_obj = 0
#         for task in tasks:
#             y = data[task].to(device)
#             task_loss = criterion_dict[task](output[task], y)
#             objectives[task] = task_loss
#             total_obj += task_loss
    
#         total_obj.backward(retain_graph=True)
#         for name, layer in model.named_modules():
#             if isinstance(layer, nn.Conv2d) and 'resnet' in name and 'downsample' not in name:
#                 importance_scores[name] += (layer.weight * layer.weight.grad).data.pow(2).sum(dim=-1).sum(dim=-1).sum(dim=-1)
#     return importance_scores

def get_task_total_importance_scores(model, tasks, train_loader, num_batches, criterion_dict):
    total_importance_scores = {}
    task_importance_scores = {}
    
    for task in tasks:
        task_importance_scores[task] = {} 

    for name, layer in model.named_modules():
        if 'gate' in name:
            # print('name:', name, layer.weight.requires_grad)
            total_importance_scores[name] = torch.zeros(layer.weight.shape[0]).to(device)
            
            for task in tasks:
                task_importance_scores[task][name] = torch.zeros(layer.weight.shape[0]).to(device)

    train_iter = iter(train_loader)
    for i in range(num_batches):
        print(f'{i}/{num_batches}')
        data = next(train_iter)
        x = data['input'].to(device)
        output = model(x)

        objectives = {}
        total_obj = 0
        for task in tasks:
            y = data[task].to(device)
            task_loss = criterion_dict[task](output[task], y)
            objectives[task] = task_loss
            total_obj += task_loss

        total_obj.backward(retain_graph=True)
        for name, layer in model.named_modules():
            if 'gate' in name:
                total_importance_scores[name] += (layer.weight * layer.weight.grad).data.pow(2)

        for task in tasks:
            task_loss = objectives[task]
            task_loss.backward(retain_graph=True)
            task_imp_scores = get_task_importance_scores_per_layer(model)
            for name, layer in model.named_modules():
                if 'gate' in name: 
                    task_importance_scores[task][name] += task_imp_scores[name]
    
    return total_importance_scores, task_importance_scores

def get_task_masks(model, keep_ratio, task_importance_score):
    task_gate_masks = {}
    task_filter_rankings = {}
    for name, score in task_importance_score.items():
        to_prune = int((1-keep_ratio) * len(score))
        sorted_scores, layer_filter_ranks = torch.sort(score)
        threshold_score = sorted_scores[to_prune].item()
        task_gate_masks[name] = (score > threshold_score).float()
        task_filter_rankings[name] = layer_filter_ranks

    return task_gate_masks, task_filter_rankings

def get_all_task_masks(model, keep_ratio, tasks, importance_scores):
    tasks_masks = {}
    tasks_filter_ranks = {}
    for task in tasks:
        tasks_masks[task], tasks_filter_ranks[task] = get_task_masks(model, keep_ratio, importance_scores[task])
    return tasks_masks, tasks_filter_ranks

def get_total_mask(model, keep_ratio, importance_scores):
    model_masks = {}
    for name, score in importance_scores.items():
        to_prune = int((1-keep_ratio) * len(score))
        sorted_scores, layer_filter_ranks = torch.sort(score)
        threshold_score = sorted_scores[to_prune].item()
        model_masks[name] = (score > threshold_score).float()
    return model_masks

def get_two_task_srocc(task1_rank, task2_rank):
    srocc = []
    for k in task1_rank.keys():
        if 'branch' in k:
            continue
        corr, pvalue = stats.spearmanr(task1_rank[k].detach().cpu().numpy(), task2_rank[k].detach().cpu().numpy())
        srocc.append(corr)
    return srocc

def get_2wise_srocc(tasks, tasks_filter_ranks):
    sroccs = {}
    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            t1 = tasks[i]
            t2 = tasks[j]
            # print(t1, t2)
            sroccs[f'{t1}/{t2}'] = get_two_task_srocc(tasks_filter_ranks[t1], tasks_filter_ranks[t2])
    return sroccs

