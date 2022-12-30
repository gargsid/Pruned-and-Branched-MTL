import sys, os 
sys.path.append('..')
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
# from layers.gate_layer import GateLayer
from nyuv2_data.pixel2pixel import ASPPHeadNode, SmallASPPHeadNode
from .gate_layer import GateLayer

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Basic1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, mask=None):
        super(Basic1, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.mask = mask
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if not isinstance(self.mask, type(None)):
            out = out * self.mask
        return out 
    
class Basic2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, mask=None):
        super(Basic2, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.mask = mask
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        if not isinstance(self.mask, type(None)):
            out = out * self.mask
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

class FlatResNet(nn.Module):
    def __init__(self, config, mask=False):
        super(FlatResNet, self).__init__()
        
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
        for i in range(1, 17):
            self.blocks += [BasicBlock(self.config[i-1], self.config[i], stride=self.strides[i])]

        self.blocks = nn.ModuleList(self.blocks)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv_gate2(x)
        
        x = self.relu(x)
        x = self.maxpool(x)

        for i, block in enumerate(self.blocks):
            x = block(x)

        return x

class DeepLabFlatResNet(nn.Module):
    def __init__(self, config, tasks, cls_dict, head_channels=512):
        super(DeepLabFlatResNet, self).__init__()
        
        self.config = config
        self.tasks = tasks 
        self.resnet = FlatResNet(config)

        self.heads = nn.ModuleDict()
        for task in self.tasks:
            self.heads[task] = SmallASPPHeadNode(head_channels, cls_dict[task])
        
    def forward(self, x):
        x = self.resnet(x)
        out = {task: self.heads[task](x) for task in self.tasks}
        return out

def deeplab_pruned_flatresnet34(tasks, cls_dict, config):
    head_channels = config[-1]
    model = DeepLabFlatResNet(config, tasks, cls_dict, head_channels=head_channels)
    return model 


# tasks = ['segment_semantic','normal','depth_zbuffer']
# T = len(tasks)
# cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}

# base_config = [64] + [64]*3 + [128]*4 + [256]*6 + [512]*3
# model = FlatResNet(base_config)
# print(model)

# x = torch.rand(1,3,321,321)
# y = model(x)

# class FlatResNet(nn.Module):
#     def __init__(self, config, masks=None):
#         super(FlatResNet, self).__init__()
        
#         self.config = config
#         self.masks = masks
#         if isinstance(masks, type(None)):
#             self.masks = [None] * (len(config)+3)

#         self.conv1 = nn.Conv2d(3, self.config[0], kernel_size=7, stride=2, padding=3,bias=False)
#         self.bn1 = nn.BatchNorm2d(self.config[0])
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         strides = [1 for i in range(6+8+12+6)]
#         strides[6] = 2
#         strides[14] = 2
#         strides[26] = 2

#         self.layers = []
#         for i in range(6+8+12+6):
#             if i%2 == 0:
#                 self.layers += [Basic1(config[i], config[i+1], strides[i], mask=self.masks[i+1])]
#             else:
#                 self.layers += [Basic2(config[i], config[i+1], strides[i], mask=self.masks[i+1])]
#             # print(i, self.layers[i])
        
#         self.layers = nn.ModuleList(self.layers)

#         self.downsample_layers = nn.ModuleList([
#             Downsample(self.config[6], self.config[7], stride=2, mask=self.masks[-3]),
#             Downsample(self.config[14], self.config[15], stride=2, mask=self.masks[-2]),
#             Downsample(self.config[26], self.config[27], stride=2, mask=self.masks[-1]),
#         ])
#         self.dmap = {7:0, 15:1, 27:2}

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         if not isinstance(self.masks[0], type(None)):
#             x = x * self.masks[0]

#         x = self.maxpool(x)

#         # mask = x.sum(dim=-1).sum(dim=-1)[0]
#         # print('conv:', 'x', len(mask.nonzero()))

#         for i in range(6+8+12+6):
#             if i%2 == 0:
#                 res = x
#                 x = self.layers[i](x)
#             else:
#                 x = self.layers[i](x) 
#                 if i in [7,15,27]:
#                     res = self.downsample_layers[self.dmap[i]](res)
#                 x += res
#             # mask = x.sum(dim=-1).sum(dim=-1)[0]
#             # print('i:', i, 'x:', x.shape, 'nonzero:', len(mask.nonzero()))
#         # sys.exit()
#         # same as output of layer4 in ResNets
#         return x




# def generate_flatresnet34_mask_map(total_model_mask):
#     layers_to_mask_map = [None] * 36
    
#     for k, v in total_model_mask.items():
#         if k == 'resnet.conv1':
#             layers_to_mask_map[0] = v.clone().view(1, -1, 1, 1)
#         elif k == 'resnet.downsample_layers.0.downsample.0':
#             layers_to_mask_map[-3] = v.clone().view(1, -1, 1, 1)
#         elif k == 'resnet.downsample_layers.1.downsample.0':
#             layers_to_mask_map[-2] = v.clone().view(1, -1, 1, 1)
#         elif k == 'resnet.downsample_layers.2.downsample.0':
#             layers_to_mask_map[-1] = v.clone().view(1, -1, 1, 1)
#         else:
#             substrs = k.split('.')
#             layer_idx = int(substrs[2])
#             if 0 <= layer_idx and layer_idx <= 5:
#                 if layer_idx % 2 == 0: # basic-1
#                     layers_to_mask_map[layer_idx+1] = v.clone().view(1, -1, 1, 1)
#                 else:
#                     layers_to_mask_map[layer_idx+1] = layers_to_mask_map[0].clone()

#             if 6 <= layer_idx and layer_idx <= 13:
#                 if layer_idx % 2 == 0: # basic-1
#                     layers_to_mask_map[layer_idx+1] = v.clone().view(1, -1, 1, 1)
#                 else:
#                     layers_to_mask_map[layer_idx+1] = total_model_mask['resnet.downsample_layers.0.downsample.0'].clone().view(1, -1, 1, 1)

#             if 14 <= layer_idx and layer_idx <= 25:
#                 if layer_idx % 2 == 0: # basic-1
#                     layers_to_mask_map[layer_idx+1] = v.clone().view(1, -1, 1, 1)
#                 else:
#                     layers_to_mask_map[layer_idx+1] = total_model_mask['resnet.downsample_layers.1.downsample.0'].clone().view(1, -1, 1, 1)

#             if 26 <= layer_idx and layer_idx <= 31:
#                 if layer_idx % 2 == 0: # basic-1
#                     layers_to_mask_map[layer_idx+1] = v.clone().view(1, -1, 1, 1)
#                 else:
#                     layers_to_mask_map[layer_idx+1] = total_model_mask['resnet.downsample_layers.2.downsample.0'].clone().view(1, -1, 1, 1)
    
#     return layers_to_mask_map 