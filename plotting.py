import os, sys
import time

import pickle

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import torch.nn as nn
import torch

def torch_load(name, path):
    ckpt = torch.load(path)
    return ckpt[name]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tasks = ['segment_semantic','normal','depth_zbuffer']
T = len(tasks)
cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}

# def two_task_iou_trends(task1_masks, task2_masks):
    
#     density_dict = {}
#     total_elem = 0
#     total_agr = 0

#     IoUs = []
#     for gidx in range(len(task1_masks)):
#         m1 = task1_masks[gidx]
#         m2 = task2_masks[gidx]
#         intersection = torch.sum((m1 + m2) == 2.)
#         union = torch.sum(torch.clamp(m1 + m2, min=0., max=1.)) 
#         IoUs.append((intersection / union).item())
#         total_agr += intersection 
#         total_elem += union 
    
#     return IoUs, (total_agr / total_elem).item()

# def all_tasks_iou_trends(tasks, keep_masks):
    
#     density_dict = {}
#     total_elem = 0
#     total_agr = 0

#     masks = []  
#     task_masks = keep_masks[tasks[0]]
#     for idx, mask in enumerate(task_masks):
#         masks.append(torch.zeros_like(mask))

#     IoUs = []
#     for idx in range(len(masks)):
#         m = masks[idx]
#         for task in tasks:
#             m += keep_masks[task][idx]
            
#         intersection = torch.sum((m) == 3.)
#         union = torch.sum(torch.clamp(m, min=0., max=1.)) 
#         IoUs.append((intersection / union).item())
#         total_agr += intersection 
#         total_elem += union 
    
#     return IoUs, (total_agr / total_elem).item()


# def plot_iou_trends(tasks, keep_masks, keep_ratio, globalprune=True, model='resnet'):
#     colors = {
#         f'{tasks[0]}/{tasks[1]}' : 'sunsetdark',
#         f'{tasks[0]}/{tasks[2]}' : 'sunsetdark',
#         f'{tasks[1]}/{tasks[2]}' : 'sunsetdark',
#     }
#     rows = {
#         f'{tasks[0]}/{tasks[1]}' : 1,
#         f'{tasks[0]}/{tasks[2]}' : 2,
#         f'{tasks[1]}/{tasks[2]}' : 3,
#     }

#     nyu_vals = {}
#     nyu_text = {}

#     fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, x_title='Backbone Layer Index', y_title="IoU")

#     for i in range(len(tasks)):
#         for j in range(i+1, len(tasks)):
#             t1 = tasks[i]
#             t2 = tasks[j]

#             task1_masks = keep_masks[t1]
#             task2_masks = keep_masks[t2]

#             task_wise_iou_dict, total_iou = two_task_iou_trends(task1_masks, task2_masks)

#             nyu_vals[f'{t1}/{t2}'] = [val for key, val in task_wise_iou_dict.items()].copy()
#             nyu_text[f'{t1}/{t2}'] = [f"{val:.3f}" for key, val in task_wise_iou_dict.items()].copy()

#             # if model_name == 'resnet':
#             #     nyu_vals[f'{t1}/{t2}'] = nyu_vals[f'{t1}/{t2}'][:33]
#             #     nyu_vals[f'{t1}/{t2}'] = nyu_vals[f'{t1}/{t2}'][:33]

#             key = f'{t1}/{t2}'
#             fig.append_trace(go.Bar(
#                         # name = f'{t1}/{t2}',
#                         y = nyu_vals[key],
#                         text = nyu_text[key],
#                         marker=dict(color = nyu_vals[key], colorscale=colors[key])),
#                         row=rows[key], col=1)

#             fig.add_annotation(xref='x domain',
#                                 yref='y domain',
#                                 x=1,
#                                 y=1.06,
#                                 text=f"{t1, t2} Total IoU: {total_iou:.2f}", 
#                                 showarrow=False,
#                                 row=rows[key], col=1)
    
#     iou_dict, total_iou, _ = all_task_iou_trends(keep_masks)
    
#     nyu_vals = [val for key, val in iou_dict.items()].copy()
#     nyu_text = [f"{val:.3f}" for key, val in iou_dict.items()].copy()


#     if model_name == 'resnet':
#         nyu_vals = nyu_vals[:33]
#         nyu_text = nyu_text[:33]

#     fig.append_trace(go.Bar(
#                         # name = f'all tasks',
#                         y = nyu_vals,
#                         text = nyu_text,
#                         marker=dict(color = nyu_vals, colorscale='sunsetdark')),
#                         row=4, col=1)

#     fig.add_annotation(xref='x domain',
#                         yref='y domain',
#                         x=1,
#                         y=1.05,
#                         text=f"All Tasks Total IoU: {total_iou:.2f}", 
#                         showarrow=False,
#                         row=4, col=1)

#     if globalprune:
#         figure_title = f'Layerwise IoUs, Global Pruning, Keep ratio:{keep_ratio:.1f}'
#     else:
#         figure_title = f'Layerwise IoUs, Local Pruning, Keep ratio:{keep_ratio:.1f}'

#     fig.update_layout(
#         # legend=dict(
#         # orientation="h",
#         # yanchor="bottom",
#         # y=1.05,
#         # xanchor="right",
#         # x=1),
#         height=800,
#         title_text=figure_title)

#     if globalprune:
#         dirpath = os.path.join('Figs', model, 'iou_trends_global_prune')
#         create_dirs(dirpath)
#         path = os.path.join(dirpath, f'keep_{keep_ratio:.1f}.png')
#         fig.write_image(path)
#         print(f'saved to', path)
#     else:
#         dirpath = os.path.join('Figs', model, 'iou_trends_local_pruning')
#         create_dirs(dirpath)
#         path = os.path.join(dirpath, f'keep_{keep_ratio:.1f}.png')
#         fig.write_image(path)
#         print(f'saved to', path)

'''
SROCC plots
'''

def plot_srocc_trends(tasks, sroccs, plotsdir, srocc_file_name):
    colors = {}
    rows = {}

    k = 1
    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            t1 = tasks[i]
            t2 = tasks[j]
            colors[f'{t1}/{t2}'] = 'sunsetdark'
            rows[f'{t1}/{t2}'] = k 
            k+=1 

    nyu_vals = {}
    nyu_text = {}

    fig = make_subplots(rows=k-1, cols=1, shared_xaxes=True, vertical_spacing=0.02, x_title='Backbone Layer Index', y_title="SROCC")

    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            t1 = tasks[i]
            t2 = tasks[j]

            two_task_srocc = sroccs[f'{t1}/{t2}']

            nyu_vals[f'{t1}/{t2}'] = [val for val in two_task_srocc]
            nyu_text[f'{t1}/{t2}'] = [f"{val:.3f}" for val in two_task_srocc]

            key = f'{t1}/{t2}'
            fig.append_trace(go.Bar(
                        name = f'{t1}/{t2}',
                        y = nyu_vals[key],
                        text = nyu_text[key],
                        marker=dict(color = nyu_vals[key], colorscale=colors[key])),
                        row=rows[key], col=1)

            fig.add_annotation(xref='x domain',
                                yref='y domain',
                                x=1,
                                y=1.06,
                                text=f"{t1, t2}", 
                                showarrow=False,
                                row=rows[key], col=1)
    
    figure_title = 'Local Pruning, Layerwise SROCC'
    fig.update_layout(
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="right",
        x=1),
        height=800,
        title_text=figure_title)

    path = os.path.join(plotsdir, srocc_file_name)
    fig.write_image(path)
    print(f'saved to', path)


'''
Accuracy plots
'''

def accuracy_plots():
    keep_ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    keep_ratios.reverse()
    macs_list = []
    params_list = []
    metrics = {}

    config_dir = 'assets/pth/flat_resnet_basicblock/keep_{}'
    ckpt_dir = 'assets/ckpt/flat_resnet34_basicblock/run_1/keep_{}' 

    for keep_ratio in keep_ratios:

        macs_dict = torch_load('macs_params', os.path.join(config_dir.format(keep_ratio), 'macs_params.pth') )
        macs = macs_dict['macs']
        params = macs_dict['params']
        macs_list.append(macs)
        params_list.append(params)
        results = torch_load('results', os.path.join(ckpt_dir.format(keep_ratio), 'results.pth'))

        for task in tasks:
            for m, v in results[task].items():
                if m not in metrics.keys():
                    metrics[m] = []
                metrics[m].append(v) 

    plotsdir = 'assets/plots/flat_resnet_basicblock'
    x = np.arange(len(macs_list))
    mac_ticks = [f'{macs.replace(" GMac", "")}' for macs in macs_list]
    for m in metrics.keys():
        vals = metrics[m]
    
        plt.figure()
        plt.plot(x, vals)
        plt.xticks(x, mac_ticks, rotation=50)
        plt.xlabel('GMac')
        plt.ylabel(f'{m}')
        plt.title(f'{m}')
        plt.tight_layout()

        plot_path = os.path.join(plotsdir, f'{m}.png')
        plt.savefig(plot_path)
        print('saved to', plot_path)
        plt.close()

def get_macs_results(branching):
    keep_ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    keep_ratios.reverse()
    macs_list = []
    params_list = []
    metrics = {}

    if branching == 'branch':
        mac_branching = 'branched'
        logs_dir = 'assets/ckpt/branched_resnet34/normal_layer_27/keep_{}'
        config_dir = 'assets/pth/branched_normal_layer_27/keep_{}'
    else:
        mac_branching = 'flat'
        logs_dir = 'assets/ckpt/flat_resnet34/run_1/keep_{}' 
        config_dir = 'assets/models/flatresnet34/pruned_configs/keep_{}'

    for keep_ratio in keep_ratios:
        macs_dict = torch_load('macs_params', os.path.join(config_dir.format(keep_ratio), 'macs_params.pth'))
        macs = macs_dict['macs']
        params = macs_dict['params']
        macs_list.append(macs)
        params_list.append(params)
        results = torch_load('results', os.path.join(logs_dir.format(keep_ratio), 'results.pth'))
        for task in tasks:
            for m, v in results[task].items():
                if m not in metrics.keys():
                    metrics[m] = []
                metrics[m].append(v) 
    
    return macs_list, metrics.copy()

def accuracy_plots_flat_branched():

    flat_macs, flat_metrics = get_macs_results('flat')
    branch_macs, branch_metrics = get_macs_results('branch')

    flat_macs = [float(x.strip(' GMac')) for x in flat_macs]
    branch_macs = [float(x.strip(' GMac')) for x in branch_macs]
    print('flat_macs:', flat_macs)
    print('branch_macs:', branch_macs)
    all_macs = []
    flat_x = []
    branch_x = []
    i = 0 
    j = 0
    k = 0
    while i<len(flat_macs) and j<len(branch_macs):
        if flat_macs[i] > branch_macs[j]:
            all_macs.append(flat_macs[i])
            flat_x.append(k)
            i += 1
            k += 1
        else:
            all_macs.append(branch_macs[j])
            branch_x.append(k)
            k += 1
            j += 1
    while i<len(flat_macs):
        all_macs.append(flat_macs[i])
        flat_x.append(k)
        i += 1
        k += 1
    while j<len(branch_macs):
        all_macs.append(branch_macs[j])
        branch_x.append(k)
        k += 1
        j += 1
    # sys.exit()
    
    plotsdir = 'assets/plots/flat_branch_normal_L26'
    x = np.arange(len(all_macs))
    mac_ticks = all_macs

    for m in flat_metrics.keys():
        fvals = flat_metrics[m]
        bvals = branch_metrics[m]
    
        plt.figure()
        plt.plot(flat_x, fvals, label='flat', marker='o')
        plt.plot(branch_x, bvals, label='branch', marker='x')

        plt.xticks(x, mac_ticks, rotation=50)
        plt.xlabel('GMac')
        plt.ylabel(f'{m}')
        plt.title(f'{m}')
        plt.tight_layout()
        plt.legend(loc='best')

        plot_path = os.path.join(plotsdir, f'{m}.png')
        plt.savefig(plot_path)
        print('saved to', plot_path)
        plt.close()


def get_macs_results_from_path(config_dir, logs_dir):
    # print('config:', config_dir)
    # print('logs_dir:', logs_dir)

    keep_ratios = np.arange(0.02, 1, 0.01)
    keep_ratios = [round(x, 2) for x in keep_ratios]
    keep_ratios.reverse()
    macs_list = []
    params_list = []
    metrics = {}

    config_dir = os.path.join(config_dir, 'keep_{}')
    logs_dir = os.path.join(logs_dir, 'keep_{}')

    for keep_ratio in keep_ratios:
        if not os.path.exists(logs_dir.format(keep_ratio)):
            continue
        # print('keep_ratio:', keep_ratio)
        macs_path = os.path.join(config_dir.format(keep_ratio), 'macs_params.pth')

        macs_dict = torch_load('macs_params', macs_path)
        macs = macs_dict['macs']
        params = macs_dict['params']
        macs_list.append(macs)
        params_list.append(params)
        results = torch_load('results', os.path.join(logs_dir.format(keep_ratio), 'results.pth'))
        for task in tasks:
            for m, v in results[task].items():
                if m not in metrics.keys():
                    metrics[m] = []
                metrics[m].append(v) 
    
    return macs_list, metrics.copy()

def get_macs_to_x_map(macs_all):
    macs_flat = []
    for ml in macs_all.values():
        for mac in ml:
            macs_flat.append(mac)
    
    macs_flat.sort(reverse=True)
    mac_to_x = {}
    for i, m in enumerate(macs_flat):
        mac_to_x[m] = i

    return mac_to_x, macs_flat 

def accuracy_plots_multiple(config_paths, ckpt_paths, plotsdir):

    macs_all = {}
    metrics_all = {}

    for k in config_paths.keys():
        macs_list, metrics_list = get_macs_results_from_path(config_paths[k], ckpt_paths[k])
        macs_list = [float(x.strip(' GMac')) for x in macs_list]
        macs_all[k] = macs_list
        metrics_all[k] = metrics_list 

    mac_to_x, macs_flat = get_macs_to_x_map(macs_all)

    x = [i for i in range(0, len(macs_flat), 5)]
    mac_ticks = [macs_flat[i] for i in x]

    for m in metrics_list.keys():
        print('metric:', m)
        
        plt.figure()
        for k in config_paths.keys():
            val = metrics_all[k][m]
            kmacs = macs_all[k]
            kx = [mac_to_x[mac] for mac in kmacs]
            plt.plot(kx, val, label=k, marker='x')

            print('macs:', kmacs)
            print('val:', val)

        plt.xticks(x, mac_ticks, rotation=90)
        plt.xlabel('GMac')
        plt.ylabel(f'{m}')
        plt.title(f'{m}')
        plt.tight_layout()
        plt.legend(loc='best')

        plot_path = os.path.join(plotsdir, f'{m}.png')
        plt.savefig(plot_path)
        print('saved to', plot_path)
        plt.close()

# config_paths = {
#     'flat' : 'assets/pth/flat_resnet_gated',
#     # 'branched@6' : 'assets/pth/branched_resnet_gated_semantic_bp_6',
#     # 'branched@10' : 'assets/pth/branched_resnet_gated_semantic_bp_10',
#     # 'branched@16' : 'assets/pth/branched_resnet_gated_semantic_bp_16',
#     'branched@20' : 'assets/pth/branched_resnet_gated_semantic_bp_20/init_flat_to_branched',
#     'branched@26' : 'assets/pth/branched_resnet_gated_semantic_bp_26/init_flat_to_branched',
#     'branched@28' : 'assets/pth/branched_resnet_gated_semantic_bp_28/init_flat_to_branched',
# }

# ckpt_paths = {
#     'flat' : 'assets/ckpt/flat_resnet34_gated',
#     # 'branched@6' : 'assets/ckpt/branched_resnet_gated_segmentation/branch_point_6',
#     # 'branched@10' : 'assets/ckpt/branched_resnet_gated_segmentation/branch_point_10',
#     # 'branched@16' : 'assets/ckpt/branched_resnet_gated_segmentation/branch_point_16',
#     'branched@20' : 'assets/ckpt/branched_resnet_gated_segmentation/init_wts_flat_to_branch/branch_point_20',
#     'branched@26' : 'assets/ckpt/branched_resnet_gated_segmentation/init_wts_flat_to_branch/branch_point_26',
#     'branched@28' : 'assets/ckpt/branched_resnet_gated_segmentation/init_wts_flat_to_branch/branch_point_28',
# }

config_paths = {
    'flat' : 'assets/ckpt/models/resnet34/flat/local_pruning/initializations',
    # 'branched@6' : 'assets/pth/branched_resnet_gated_semantic_bp_6',
    'branched@10' : 'assets/ckpt/models/resnet34/branched/segmentation/branch_point_10/local_prune_uniform/initializations',
    'branched@16' : 'assets/ckpt/models/resnet34/branched/segmentation/branch_point_16/local_prune_uniform/initializations',
    'branched@20' : 'assets/ckpt/models/resnet34/branched/segmentation/branch_point_20/local_prune_uniform/initializations',
    'branched@26' : 'assets/ckpt/models/resnet34/branched/segmentation/branch_point_26/local_prune_uniform/initializations',
    'branched@28' : 'assets/ckpt/models/resnet34/branched/segmentation/branch_point_28/local_prune_uniform/initializations',
}

ckpt_paths = {
    'flat' : 'assets/ckpt/models/resnet34/flat/local_pruning/training/run_1',
    # 'branched@6' : 'assets/ckpt/branched_resnet_gated_segmentation/branch_point_6',
    'branched@10' : 'assets/ckpt/models/resnet34/branched/segmentation/branch_point_10/local_prune_uniform/training/run_2',
    'branched@16' : 'assets/ckpt/models/resnet34/branched/segmentation/branch_point_16/local_prune_uniform/training/run_2',
    'branched@20' : 'assets/ckpt/models/resnet34/branched/segmentation/branch_point_20/local_prune_uniform/training/run_2',
    'branched@26' : 'assets/ckpt/models/resnet34/branched/segmentation/branch_point_26/local_prune_uniform/training/run_2',
    'branched@28' : 'assets/ckpt/models/resnet34/branched/segmentation/branch_point_28/local_prune_uniform/training/run_2',
}

plotsdir = 'assets/plots/flat_resnet/local_pruning/training/run_1'
# accuracy_plots_multiple(config_paths, ckpt_paths, plotsdir)

# srocc_arr_path = 'assets/ckpt/models/resnet34/flat/local_pruning/initializations/sroccs_flatresnet34_nyu.pth'
# sroccs = torch_load('sroccs', srocc_arr_path)
# plotsdir = 'assets/ckpt/models/resnet34/flat/local_pruning/initializations'
# plot_srocc_trends(tasks, sroccs, plotsdir, f'local_pruning_flat_resnet_srocc.pdf')

# bp = 26
# sroccs = torch_load('sroccs', f'assets/pth/branched_resnet_gated_semantic_bp_{bp}/sroccs_branched_resnet_nyu.pth')
# print(sroccs)

def print_results(config_path, ckpt_path):
    macs_list, metrics_list = get_macs_results_from_path(config_path, ckpt_path)
    print('macs:',macs_list)
    for m in metrics_list.keys():
        print(m, metrics_list[m])

import pandas as pd

display_metrics = ['Pixel Acc', 'mIoU', 'abs_err', 'rel_err', 'Angle Median', 'Angle Mean']

dfs = {}
columns = ['GMacs', 'flat']
macs_list, metrics_list = get_macs_results_from_path(config_paths['flat'], ckpt_paths['flat'])
# baseline_macs = macs_list.copy()

for m in metrics_list.keys():
    flat_metric_results = []
    for macs, val in zip(macs_list, metrics_list[m]):
        gms = float(macs.strip(' GMacs'))
        r = f'{val:.4f} ({gms} GMacs)'
        flat_metric_results.append(r)

    dfs[m] = pd.DataFrame(list(zip(macs_list, flat_metric_results)), columns=columns)
    if m in display_metrics:
        print(m)
        print(dfs[m])

# sys.exit()

for m in metrics_list.keys():
    # print(m)

    baseline_macs = dfs[m]['GMacs'].tolist()
    baseline_macs = [float(gms.strip(' GMacs')) for gms in baseline_macs]
    
    for k in config_paths.keys():
        if k == 'flat':
            continue

        macs_list, metrics_list = get_macs_results_from_path(config_paths[k], ckpt_paths[k])
        macs_list = [float(gms.strip(' GMacs')) for gms in macs_list]
        
        config_metric_results = []    
        for fmac in baseline_macs:
            flag = True
            for i, macs in enumerate(macs_list):
                if macs - 0.05 <= fmac and fmac <= macs+0.05:
                    val = round(metrics_list[m][i], 4)
                    r = f'{val:.4f} ({macs} GMacs)'
                    config_metric_results.append(r)
                    flag = False
                    break 
            if flag:
                config_metric_results.append('-')
        
        # print(config_metric_results)

        dfs[m][k] = config_metric_results

    print(m)
    if m in display_metrics:
        print(dfs[m])
    # sys.exit()

df_dir = 'assets/plots/local_pruning_uniform/branched/segmentation/dataframes'

for m, df in dfs.items():
    path = os.path.join(df_dir, f'results_{m}.csv')
    df.to_csv(path)
    print('saved to', path)


# for k in config_paths.keys():
#     columns.append(k)



