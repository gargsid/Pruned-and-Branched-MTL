import sys, os 
sys.path.append('..')
import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
# from layers.gate_layer import GateLayer
from nyuv2_data.pixel2pixel import ASPPHeadNode, SmallASPPHeadNode
from .gate_layer import GateLayer

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

def get_filters(x):
    return len(x[0].sum(dim=-1).sum(dim=-1).nonzero().squeeze())

def get_conv_sum(x):
    return len(x.sum(dim=-1).sum(dim=-1).sum(dim=-1).nonzero().squeeze())

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Basic1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Basic1, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out 
    
class Basic2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Basic2, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        return out 

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Downsample, self).__init__()
        self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.downsample_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.downsample_conv(x)
        x = self.downsample_bn(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.gate1 = GateLayer(in_channels, out_channels, [1,-1,1,1]) 

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.gate2 = GateLayer(in_channels, out_channels, [1,-1,1,1]) 

        self.shortcut = Downsample(in_channels, out_channels, stride=stride)
    
    def forward(self, x, prune=False):

        res = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gate1(x)

        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gate2(x)

        x += res

        x = x * self.gate2.weight.detach().view(*self.gate2.size_mask)

        return x 

class BranchedResNet(nn.Module):
    def __init__(self, config, tasks, branch_point, masks=None):
        super(BranchedResNet, self).__init__()
        
        # possible-branch-points: [0, 2, 4, 8, 10, 12, 16, 18, 20, 22, 24, 28, 30] + [6, 14, 26]
        self.branch_point = branch_point #
        self.config = config
        self.tasks = tasks 
        self.config = config

        self.strides = [1] * 17
        self.strides[0] = 2
        self.strides[4] = 2
        self.strides[8] = 2
        self.strides[14] = 2

        self.conv1 = nn.Conv2d(3, self.config[0], kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.config[0])
        self.conv_gate2 = GateLayer(config[0], config[0], [1,-1,1,1])

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = []
        self.num_shared = branch_point // 2 + 1

        for i in range(1, self.num_shared):
            self.blocks += [BasicBlock(self.config[i-1], self.config[i], stride=self.strides[i])]

        self.blocks = nn.ModuleList(self.blocks)
    
        self.left_branch = self.create_branch()
        self.right_branch = self.create_branch()
    
    def create_branch(self):
        blocks = []
        for i in range(self.num_shared, 17):
            blocks += [BasicBlock(self.config[i-1], self.config[i], stride=self.strides[i])]
        blocks = nn.ModuleList(blocks)
        return blocks

    def branch_forward(self, blocks, x, prune=False):
        for i, block in enumerate(blocks):
            x = block(x, prune)
        return x

    def forward(self, x, prune=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv_gate2(x)

        x = self.relu(x)
        x = self.maxpool(x)

        for i, block in enumerate(self.blocks):
            x = block(x, prune)
        
        left_x = self.branch_forward(self.left_branch, x, prune)
        right_x = self.branch_forward(self.right_branch, x, prune)

        out = {}
        out['segment_semantic'] = left_x

        out['normal'] = right_x
        out['depth_zbuffer'] = right_x

        return out

class DeepLabBranchedResNet(nn.Module):
    def __init__(self, config, tasks, cls_dict, branch_point, head_channels=512):
        super(DeepLabBranchedResNet, self).__init__()
        
        self.config = config
        self.tasks = tasks 
        self.resnet = BranchedResNet(config, tasks, branch_point)

        self.heads = nn.ModuleDict()
        for task in self.tasks:
            self.heads[task] = SmallASPPHeadNode(head_channels, cls_dict[task])
        
    def forward(self, x, prune=False):
        x = self.resnet(x, prune)
        out = {task: self.heads[task](x[task]) for task in self.tasks}
        return out

def deeplab_pruned_branchedresnet34(tasks, cls_dict, config, branch_point):
    head_channels = config[-1]
    model = DeepLabBranchedResNet(config, tasks, cls_dict, branch_point, head_channels=head_channels)
    return model 

# tasks = ['segment_semantic','normal','depth_zbuffer']
# T = len(tasks)
# cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}

# config = [64] + [64]*3 + [128]*4 + [256]*6 + [512]*3
# # print(config)
# for i, c in enumerate(config):
#     print(i, c)

# branch_point = 20
# model = deeplab_pruned_branchedresnet34(tasks, cls_num, config, branch_point)
# # print(model)

# import torch
# x = 10*torch.ones(1,3,321,321)
# y = model(x)

# model = BranchedResNet()
# print(model)
# x = torch.rand(1, 3, 321, 321)
# y = model(x)

# '''
# Branch out segmentation from provided layer (idx) in the config. 
# '''
# class BranchedResNet(nn.Module):
#     def __init__(self, config, tasks, branch_point, masks=None):
#         super(BranchedResNet, self).__init__()
        
#         self.branch_point = branch_point # branching starts at this layer index. NOT allowed points (config index) = [7,15,27]
#         self.config = config
#         self.tasks = tasks 

#         self.masks = masks
#         self.masking = False
#         if masks is not None:
#             self.masking = True

#         self.conv1 = nn.Conv2d(3, self.config[0], kernel_size=7, stride=2, padding=3,bias=False)
#         self.bn1 = nn.BatchNorm2d(self.config[0])
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         strides = [1 for i in range(6+8+12+6)]
#         strides[6] = 2
#         strides[14] = 2
#         strides[26] = 2
#         self.strides = strides

#         self.layers = []
#         self.shared_shortcuts = []
#         for i in range(self.branch_point): # shared network till layer index - branch_point-1
#             if i%2 == 0:
#                 self.layers += [Basic1(config[i], config[i+1], self.strides[i])]
#             else:
#                 self.layers += [Basic2(config[i], config[i+1], self.strides[i])]
#                 if i==7:
#                     self.shared_shortcuts += [Downsample(self.config[6], self.config[8], stride=2)]
#                 if i==15:
#                     self.shared_shortcuts += [Downsample(self.config[14], self.config[16], stride=2)]
#                 if i==27:
#                     self.shared_shortcuts += [Downsample(self.config[25], self.config[27], stride=2)]
                
#         self.layers = nn.ModuleList(self.layers)
#         self.shared_shortcuts = nn.ModuleList(self.shared_shortcuts)

#         self.left_branch, self.left_shortcuts = self.create_branch()
#         self.right_branch, self.right_shortcuts = self.create_branch()

#         if (self.branch_point - 1) % 2: 
#             idx = self.branch_point - 1
#             self.left_downsample_shortcut = Downsample(self.config[idx+1], self.config[idx+1+2])
#             self.right_downsample_shortcut = Downsample(self.config[idx+1], self.config[idx+1+2])
#         elif (self.branch_point - 2) % 2:
#             idx = self.branch_point - 2
#             self.left_downsample_shortcut = Downsample(self.config[idx+1], self.config[idx+1+2])
#             self.right_downsample_shortcut = Downsample(self.config[idx+1], self.config[idx+1+2])

#     def create_branch(self):
#         branch = []
#         shortcuts = []
        
#         for i in range(self.branch_point, 6+8+12+6): # shared network till layer 27
#             if i%2 == 0:
#                 branch += [Basic1(self.config[i], self.config[i+1], self.strides[i])]
#             else:
#                 branch += [Basic2(self.config[i], self.config[i+1], self.strides[i])]
#                 if i==7:
#                     shortcuts += [Downsample(self.config[6], self.config[8], stride=2)]
#                 if i==15:
#                     shortcuts += [Downsample(self.config[14], self.config[16], stride=2)]
#                 if i==27:
#                     shortcuts += [Downsample(self.config[25], self.config[27], stride=2)]
        
#         branch = nn.ModuleList(branch)
#         shortcuts = nn.ModuleList(shortcuts)

#         return branch, shortcuts

#     def shared_forward(self, x, res_mask=None):
#         for i in range(self.branch_point):
#             if i%2 == 0:
#                 res = x
#                 x = self.layers[i](x)

#                 x_filters = get_filters(x)
#                 layer_weight = get_conv_sum(self.layers[i].conv1.weight)

#                 # if self.masking and i==14:
#                 #     keep_masks = self.masks['layers'][i].sum()
                
#                 if self.masking:
#                     print(x[0].sum(dim=-1).sum(dim=-1), 'before masking', get_filters(x), 'out-of-conv-layer:', layer_weight)
#                     zeroed_filter_idx = (x[0].sum(dim=-1).sum(dim=-1)==0).nonzero().squeeze()
                    
#                     # print('zeroed_filter_idx', zeroed_filter_idx)
                    
#                     # selected_filters = torch.index_select(self.layers[i].conv1.weight, 0, zeroed_filter_idx)
#                     # print('selected_filters:', selected_filters.shape)
#                     # for kernel in selected_filters[0]:
#                     #     print(kernel)
                    
#                     x = x * self.masks['layers'][i]
#                     print(x[0].sum(dim=-1).sum(dim=-1), 'after masking', get_filters(x))
#                     sys.exit()
#                     # if i==14:
#                     #     x_masked_filters = get_filters(x)
#                     #     print(f'x_filters(no mask):{x_filters} keep_masks:{keep_masks} x_masked: {x_masked_filters}')
#                     #     sys.exit()
                
#             else:
#                 if i == 7:
#                     res = self.shared_shortcuts[0](res)
#                     if self.masking:
#                         res_mask = self.masks['shared_shortcuts'][0]
#                 if i == 15:
#                     res = self.shared_shortcuts[1](res)
#                     if self.masking:
#                         res_mask = self.masks['shared_shortcuts'][1]
#                 if i == 27:
#                     res = self.shared_shortcuts[2](res)
#                     if self.masking:
#                         res_mask = self.masks['shared_shortcuts'][2]
                
#                 x = self.layers[i](x) 
#                 x_filters = get_filters(x)
#                 layer_weight = get_conv_sum(self.layers[i].conv1.weight)

#                 x += res

#                 if self.masking:
#                     x = x * res_mask

#             print(f'shared i:{i} x: {x.shape}, x_filters(no mask):{x_filters} remaining: {get_filters(x)} layer_weight:{layer_weight}')
#         return x, res
        
#     def branch_forward(self, branch, shortcuts, downsample_shortcut, y, res, masks=None):

#         if self.masking:
#             masks_list = masks['masks_list']
#             shortcut_masks_list = masks['shortcut_masks_list']
#             downsample_shortcut_mask = masks['downsample_shortcut_mask']

#         for i in range(self.branch_point, 6+8+12+6):
#             if i%2 == 0:
#                 if i == self.branch_point:
#                     res = downsample_shortcut(y)
#                     if self.masking:
#                         res_mask = downsample_shortcut_mask
#                 else:
#                     res = y
                
#                 y = branch[i-self.branch_point](y)
#                 if self.masking:
#                     y = y * masks_list[i-self.branch_point]
#             else:
#                 if i == 7:
#                     res = shortcuts[-3](res)
#                     if self.masking:
#                         res_mask = shortcut_masks_list[-3]
#                 elif i == 15:
#                     res = shortcuts[-2](res)
#                     if self.masking:
#                         res_mask = shortcut_masks_list[-2]
#                 elif i == 27:
#                     res = shortcuts[-1](res)
#                     if self.masking:
#                         res_mask = shortcut_masks_list[-1]
#                 elif i == self.branch_point:
#                     res = downsample_shortcut(res)
#                     if self.masking:
#                         res_mask = downsample_shortcut_mask

#                 y = branch[i-self.branch_point](y)
#                 y += res
#                 if self.masking:
#                     y = y * res_mask
#             # print(f'branched i:{i} y: {y.shape}')
#             print(f'branch i:{i} y: {y.shape}, remaining: {get_filters(y)}')
#         return y

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)


#         if self.masking:
#             res_mask = self.masks['conv1']

#             print(x[0].sum(dim=-1).sum(dim=-1), get_filters(x))
#             x = x * res_mask 
#             print(x[0].sum(dim=-1).sum(dim=-1), get_filters(x))

#             # sys.exit()

#             print(res_mask.sum())
#             print(f'conv1 x: {x.shape}, remaining: {get_filters(x)}')

#             x, res = self.shared_forward(x, res_mask)

#             left_masks = {}
#             left_masks['masks_list'] = self.masks['left_branch']
#             left_masks['shortcut_masks_list'] = self.masks['left_shortcuts']
#             left_masks['downsample_shortcut_mask'] = self.masks['left_downsample_shortcut']

#             right_masks = {}
#             right_masks['masks_list'] = self.masks['right_branch']
#             right_masks['shortcut_masks_list'] = self.masks['right_shortcuts']
#             right_masks['downsample_shortcut_mask'] = self.masks['right_downsample_shortcut']

#             y_semantic = self.branch_forward(self.left_branch, self.left_shortcuts, self.left_downsample_shortcut, x, res, left_masks)
#             y_others = self.branch_forward(self.right_branch, self.right_shortcuts, self.right_downsample_shortcut, x, res, right_masks)
#         else:
#             x, res = self.shared_forward(x)
#             y_semantic = self.branch_forward(self.left_branch, self.left_shortcuts, self.left_downsample_shortcut, x, res)
#             y_others = self.branch_forward(self.right_branch, self.right_shortcuts, self.right_downsample_shortcut, x, res)
        
#         out = {}
#         out['segment_semantic'] = y_semantic

#         out['depth_zbuffer'] = y_others
#         out['normal'] = y_others

#         return out

# class DeepLabBranchedResNet(nn.Module):
#     def __init__(self, config, tasks, cls_dict, branch_point, head_channels=512, masks=None):
#         super(DeepLabBranchedResNet, self).__init__()
        
#         self.config = config
#         self.tasks = tasks 
#         self.resnet = BranchedResNet(config, tasks, branch_point, masks=masks)

#         self.heads = nn.ModuleDict()
#         for task in self.tasks:
#             self.heads[task] = SmallASPPHeadNode(head_channels, cls_dict[task])
        
#     def forward(self, x):
#         x = self.resnet(x)
#         out = {task: self.heads[task](x[task]) for task in self.tasks}
#         return out

# def generate_branched_resnet_mask_map(total_model_mask):
#     shared_masks = [] # 2 for downsample
#     left_masks = []  # 2 for downsample
#     right_masks = [] # 2 for downsample
#     left_shortcuts = []
#     right_shortcuts = []
#     shared_shortcuts = []
#     masks = {}

#     masks['conv1'] = total_model_mask['resnet.conv1'].clone().view(1, -1, 1, 1)

#     for k, v in total_model_mask.items():
#         # print('k:', k)
#         if 'resnet.layers' in k:
#             shared_masks += [v.clone().view(1,-1,1,1)]
#         elif 'shared_shortcut' in k:
#             shared_shortcuts += [v.clone().view(1,-1,1,1)]

#         elif 'left_branch' in k:
#             left_masks += [v.clone().view(1,-1,1,1)]
#         elif 'left_shortcuts' in k:
#             left_shortcuts += [v.clone().view(1,-1,1,1)]  

#         elif 'right_branch' in k:
#             right_masks += [v.clone().view(1,-1,1,1)]
#         elif 'right_shortcuts' in k:
#             right_shortcuts += [v.clone().view(1,-1,1,1)]  
#         elif 'left_downsample_shortcut' in k:
#             left_downsample_shortcut = v.clone().view(1,-1,1,1)
#         elif 'right_downsample_shortcut' in k:
#             right_downsample_shortcut = v.clone().view(1,-1,1,1)
    
#     masks['layers'] = shared_masks
#     masks['shared_shortcuts'] = shared_shortcuts
#     masks['left_branch'] = left_masks
#     masks['left_shortcuts'] = left_shortcuts
#     masks['right_branch'] = right_masks
#     masks['right_shortcuts'] = right_shortcuts
#     masks['left_downsample_shortcut'] = left_downsample_shortcut
#     masks['right_downsample_shortcut'] = right_downsample_shortcut
#     return masks

# def deeplab_pruned_branchedresnet34(tasks, cls_dict, config, branch_point, masks=None):
#     head_channels = config[-1]
#     model = DeepLabBranchedResNet(config, tasks, cls_dict, branch_point, head_channels=head_channels, masks=masks)
#     return model 

# tasks = ['segment_semantic','normal','depth_zbuffer']
# T = len(tasks)
# cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}

# config = [64] + [64 for i in range(6)] + [128 for i in range(8)] + [256 for i in range(12)] + [512 for i in range(6)]
# config = [int(x) for x in config]
# # print(config)
# for i, c in enumerate(config):
#     print(i, c)

# # branching starts at this layer index. NOT allowed points (config index) = [7,15,27]
# branch_point = 20
# model = deeplab_pruned_branchedresnet34(tasks, cls_num, config, branch_point)

# # print(model)

# import torch
# x = torch.rand(1,3,321,321)
# y = model(x)