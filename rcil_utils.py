import torch
import torch.nn as nn

####### merged branches ####### begin
def merge(conv2d, bn2d, conv_bias=None):
    if conv_bias is not None:
        conv_bias = conv_bias.clone().to(conv2d.weight.device)
    k = conv2d.weight.clone()
    running_mean = bn2d.running_mean
    running_var = bn2d.running_var
    eps = bn2d.eps
    gamma = bn2d.weight.abs() + eps
    beta = bn2d.bias

    gamma = gamma / 2.
    beta = beta / 2.

    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
  
    if conv_bias is not None:
        return k * t, beta - running_mean * gamma / std + t.view(-1) * conv_bias.view(-1)
    else:
        return k * t, beta - running_mean * gamma / std

def mergex(conv2d, bn2d, pos, conv_bias=None):
    if conv_bias is not None:
        conv_bias = conv_bias.clone().to(conv2d.weight.device)    
    k = conv2d.weight.clone()
    running_mean = bn2d.running_mean[pos*256:(1+pos)*256]
    running_var = bn2d.running_var[pos*256:(1+pos)*256]
    eps = bn2d.eps
    gamma = bn2d.weight.abs()[pos*256:(1+pos)*256] + eps
    beta = bn2d.bias[pos*256:(1+pos)*256]

    gamma = gamma / 2.
    beta = beta / 2.

    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    if conv_bias is not None:
        return k * t, beta - running_mean * gamma / std + t.view(-1) * conv_bias.view(-1)
    else:
        return k * t, beta - running_mean * gamma / std

def init_right(conv2d, bn2d, conv2d_new, bn2d_new, init_type):
    if init_type == 'original':
        return conv2d_new, bn2d_new
    return conv2d_new, bn2d_new

def convert_model(model, load_dict=None):
    for name, mm in model.named_modules():
        if hasattr(mm, 'convs'):
            k1, b1 = merge(mm.convs.conv2, mm.convs.bn2, mm.convs.conv2.bias.data)
            k2, b2 = merge(mm.convs.conv2_new, mm.convs.bn2_new, None)
            k = k1 + k2
            b = b1 + b2
            mm.convs.conv2.weight.data[:,:,:,:] = k[:,:,:,:]
            mm.convs.conv2.bias = nn.Parameter(b)
            mm.convs.bn2.bias.data[:] = torch.zeros((mm.convs.bn2.weight.shape[0],))[:]
            mm.convs.bn2.running_var.data[:] = torch.ones((mm.convs.bn2.weight.shape[0],))[:]
            mm.convs.bn2.eps = 0
            mm.convs.bn2.weight.data[:] = torch.ones((mm.convs.bn2.weight.shape[0],))[:]
            mm.convs.bn2.running_mean.data[:] = torch.zeros((mm.convs.bn2.weight.shape[0],))[:]
            mm.convs.bn2.eval()
            mm.convs.conv2.eval()
            for p in mm.convs.bn2.parameters():
                p.requires_grad = False
            for p in mm.convs.conv2.parameters():
                p.requires_grad = False
        elif hasattr(mm, 'map_convs'):
            for i in range(4):
                k1, b1 = mergex(mm.map_convs[i], mm.map_bn, i, mm.map_convs[i].bias.data)
                k2, b2 = mergex(mm.map_convs_new[i], mm.map_bn_new, i, None)
                k = k1 + k2
                b = b1 + b2
                mm.map_convs[i].weight.data[:,:,:,:] = k[:,:,:,:]
                mm.map_convs[i].bias = nn.Parameter(b)
                mm.map_convs[i].eval()
                for p in mm.map_convs[i].parameters():
                    p.requires_grad = False
            mm.map_bn.eval()
            for p in mm.map_bn.parameters():
                p.requires_grad = False
            mm.map_bn.bias.data[:] = torch.zeros((mm.map_bn.weight.shape[0],))[:]
            mm.map_bn.running_var.data[:] = torch.ones((mm.map_bn.weight.shape[0],))[:]
            mm.map_bn.eps = 0
            mm.map_bn.weight.data[:] = torch.ones((mm.map_bn.weight.shape[0],))[:]
            mm.map_bn.running_mean.data[:] = torch.zeros((mm.map_bn.weight.shape[0],))[:]
    return model
####### merged branches ####### end
#################################################################