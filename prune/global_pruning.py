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

def taylor_pruning_criteria(weight, weight_grad):
    criteria = (weight * weight_grad).detach()**2 # gated layers
    # criteria = (weight * weight_grad).detach().abs() $ gated layers 
    # criteria = (weight * weight_grad).sum(dim=-1).sum(dim=-1).sum(dim=-1).detach()**2 # convolutional layers
    return criteria

def filter_name(name):
    if 'gate' in name:
        return True 
    else:
        return False

    # if 'resnet' in name:
    #     if 'conv1' in name or 'conv2' in name:
    #         return True
    # else:
    #     return False

def get_task_importance_scores_per_layer(model):
    task_importance_score = {}
    for name, layer in model.named_modules():
        if filter_name(name):
            task_importance_score[name] = taylor_pruning_criteria(layer.weight, layer.weight.grad)
    return task_importance_score

def get_task_total_importance_scores(model, tasks, train_loader, num_batches, criterion_dict):
    total_importance_scores = {}
    task_importance_scores = {}
    
    for task in tasks:
        task_importance_scores[task] = {} 

    for name, layer in model.named_modules():
        if filter_name(name):
            total_importance_scores[name] = torch.zeros(layer.weight.shape[0]).to(device)
            
            for task in tasks:
                task_importance_scores[task][name] = torch.zeros(layer.weight.shape[0]).to(device)
    
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, betas=(0.5, 0.999), weight_decay=0.0001)

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

        for task in tasks:
            task_loss = objectives[task]
            task_loss.backward(retain_graph=True)
            task_imp_scores = get_task_importance_scores_per_layer(model)
            for name, layer in model.named_modules():
                if filter_name(name): 
                    task_importance_scores[task][name] += task_imp_scores[name]
        
        # optimizer.zero_grad()
        total_obj.backward()
        for name, layer in model.named_modules():
            if filter_name(name):
                total_importance_scores[name] += taylor_pruning_criteria(layer.weight, layer.weight.grad)
        # optimizer.step()
    #     for name, layer in model.named_modules():
    #         if 'resnet' in name:
    #             if 'conv1' in name or 'conv2' in name:
    #                 score = taylor_pruning_criteria(layer.weight, layer.weight.grad)
    #                 print('name:', name, 'score:', score)
    #     sys.exit()
    
    # for name, layer in model.named_modules():
    #     if filter_name(name):
    #         print(name, total_importance_scores[name])

    return total_importance_scores, task_importance_scores

'''
global pruning
'''
def get_global_pruning_mask(model, keep_ratio, importance_scores):
    model_masks = {}
    all_scores = None 
    for name, layer_scores in importance_scores.items():
        if isinstance(all_scores, type(None)):
            all_scores = layer_scores.clone()
        else:
            all_scores = torch.cat((all_scores, layer_scores), dim=-1)
        # print('all_scores:',all_scores.shape)

    to_prune = int((1-keep_ratio) * len(all_scores))
    sorted_scores, filter_ranks = torch.sort(all_scores)
    threshold_score = sorted_scores[to_prune].item()
    print('threshold_score:', threshold_score)
    
    for name, layer_scores in importance_scores.items():
        model_masks[name] = (layer_scores > threshold_score).float()
    
    return model_masks, filter_ranks

def get_total_mask(model, keep_ratio, importance_scores):
    model_masks, _ = get_global_pruning_mask(model, keep_ratio, importance_scores)
    return model_masks

def get_all_task_masks(model, keep_ratio, tasks, importance_scores):
    tasks_masks = {}
    tasks_filter_ranks = {}
    for task in tasks:
        tasks_masks[task], tasks_filter_ranks[task] = get_global_pruning_mask(model, keep_ratio, importance_scores[task])
    return tasks_masks, tasks_filter_ranks