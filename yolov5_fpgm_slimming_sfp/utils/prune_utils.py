import torch
import torch.nn as nn

from copy import deepcopy
from pathlib import Path
from models.yolo import Detect
from models.common import *
from models.experimental import *
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import yaml
# from utils.metrics import fitness
from utils.general import make_divisible
from scipy.spatial import distance


def get_mask_sfp(model, sfp_ratio, cut_ids, device):
    cfg_mask = {}
    for name, module in model.named_modules():
        print('name', name)
        if isinstance(module, nn.Conv2d):
            if name in cut_ids:
                mask = torch.ones(module.weight.data.size())
                mask, mask_last = sfp_filer(module, sfp_ratio, mask)
                mask = mask.to(device)
            else:
                mask = torch.ones(module.weight.data.size()).to(device)
                mask_last = torch.ones(module.weight.data.size()[0])

            module.weight.data.mul_(mask)
            cfg_mask[name] = mask_last
    return cfg_mask
def sfp_filer(module, sfp_ratio, mask):
    num_retain_init = int(module.weight.data.size()[0] * (1 - sfp_ratio))
    num_retain = max(make_divisible(num_retain_init, 8), 8)
    num_cut = module.weight.data.size()[0] - num_retain
    weight_vec = module.weight.data.view(module.weight.data.size()[0], -1)
    norm2 = torch.norm(weight_vec, 2, 1)
    norm2_np = norm2.cpu().numpy()
    filter_index = norm2_np.argsort()[:num_cut]

    mask_last = torch.ones(module.weight.data.size()[0])
    mask_last[filter_index] = 0
    # print('original len', module.weight.data.size()[0])
    # print('the number of cut', num_cut)
    # print('the index of filter', filter_index)
    # print('mask_last', mask_last)



    mask = mask.view(-1)
    kernel_length = module.weight.data.size()[1] * module.weight.data.size()[2] * module.weight.data.size()[3]
    for x in range(0, len(filter_index)):
        mask[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0
    # print('mask', mask)

    mask = mask.view(module.weight.data.size())

    return mask, mask_last

def if_zero(model, cut_ids):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if name in cut_ids:
                a = module.weight.data.view(-1)
                b = a.cpu().numpy()

                print("layer: %s, number of nonzero weight is %d, zero is %d" % (
                    name, np.count_nonzero(b), len(b) - np.count_nonzero(b)))


def get_mask_fpgm(model, fpgm_ratio, cut_ids, device):
    cfg_mask = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if name in cut_ids:
                mask = torch.ones(module.weight.data.size())
                mask, mask_last = fpgm_filer(module, fpgm_ratio, mask)
                mask = mask.to(device)
            else:
                print(device)
                mask = torch.ones(module.weight.data.size()).to(device)
                mask_last = torch.ones(module.weight.data.size()[0])
            module.weight.data.mul_(mask)
            cfg_mask[name] = mask_last
    return cfg_mask
def fpgm_filer(module, fpgm_ratio, mask):

    num_retain_init = int(module.weight.data.size()[0] * (1 - fpgm_ratio))
    num_retain = max(make_divisible(num_retain_init, 8), 8)
    num_cut = module.weight.data.size()[0] - num_retain
    weight_vec = module.weight.data.view(module.weight.data.size()[0], -1).cpu().numpy()
    # norm2 = torch.norm(weight_vec, 2, 1)
    # norm2_np = norm2.cpu().numpy()
    similar_matrix = distance.cdist(weight_vec, weight_vec, 'euclidean')
    similar_sum = np.sum(np.abs(similar_matrix), axis=0)
    filter_index = similar_sum.argsort()[:num_cut]

    mask_last = torch.ones(module.weight.data.size()[0])
    mask_last[filter_index] = 0

    mask = mask.view(-1)
    kernel_length = module.weight.data.size()[1] * module.weight.data.size()[2] * module.weight.data.size()[3]
    for x in range(0, len(filter_index)):
        mask[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

    mask = mask.view(module.weight.data.size())

    return mask, mask_last


def updateBN(model, s, cut_id):
    cut_ids_bn = [i.replace('conv', 'bn') for i in cut_id]
    for name,module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) and name in cut_ids_bn:
            module.weight.grad.data.add_(s*torch.sign(module.weight.data))  # L1

def module_pruning_layer(d):
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    cut_ids = []
    ignore_ids = []
    from_layer = []
    from_to = {}
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        name_base = "model.{}.".format(i)
        if m is Conv:
            name_base_conv = name_base + 'conv'
            cut_ids.append(name_base_conv)
            if i > 0:
                from_to[name_base_conv] = from_layer[f]
            from_layer.append(name_base_conv)

        elif m is C3:
            name_base_conv1 = name_base + 'cv1.conv'
            name_base_conv2 = name_base + 'cv2.conv'
            name_base_conv3 = name_base + 'cv3.conv'
            ignore_ids.append(name_base_conv1)
            cut_ids.append(name_base_conv2)
            cut_ids.append(name_base_conv3)
            from_to[name_base_conv1] = from_layer[f]
            from_to[name_base_conv2] = from_layer[f]
            from_to[name_base_conv3] = [name_base_conv1, name_base_conv2]
            from_layer.append(name_base_conv3)
            for j in range(n):
                name_base_m_conv1 = name_base + "m.{}.cv1.conv".format(j)
                name_base_m_conv2 = name_base + "m.{}.cv2.conv".format(j)
                cut_ids.append(name_base_m_conv1)
                ignore_ids.append(name_base_m_conv2)
                from_to[name_base_m_conv1] = name_base_conv1
                from_to[name_base_m_conv2] = name_base_m_conv1

        elif m is Detect:
            for j in range(3):
                name_base_conv = name_base + "m.{}".format(j)
                ignore_ids.append(name_base_conv)
                from_to[name_base_conv] = from_layer[f[j]]
        elif m is Concat:
            name_base_concate = [from_layer[x] for x in f]
            from_layer.append(name_base_concate)
        elif m is SPPF:
            named_m_conv1 = name_base+'cv1.conv'
            named_m_conv2 = name_base+'cv2.conv'
            cut_ids.append(named_m_conv1)
            ignore_ids.append(named_m_conv2)
            from_to[named_m_conv1] = from_layer[f]
            from_to[named_m_conv2] = [named_m_conv1]*4
            from_layer.append(named_m_conv2)
        else:
            from_layer.append(from_layer[f])

    return cut_ids, ignore_ids, from_to


def get_filtermask(bn_module, thre):
    w_copy = bn_module.weight.data.abs().clone()

    num_retain_init = int(sum(w_copy.gt(thre).float()))

    length = w_copy.shape[0]

    num_retain = max(make_divisible(num_retain_init, 8), 8)

    # print(length)
    # print('num_retain',num_retain)
    _, index = torch.topk(w_copy, num_retain)
    mask = torch.zeros(length)
    mask[index.cpu()] = 1

    return mask



def update_pruning_yaml(pruning_yaml, model, maskconv, cut_ids):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # print('name', name)
            if name in cut_ids:
                update_pruning_yaml_loop(name, pruning_yaml, maskconv)
    
    return pruning_yaml

def update_pruning_yaml_loop(name, d, maskconv):
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    ch = [3]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = a # eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        name_base = "model.{}.".format(i)
        if m is Conv:
            c2 = args[0]
            
            if c2 != no:  # if not output
                if isinstance(args[-1], float) and m is not SPP and m is not SPPF:
                    c2 = c2 * args[-1]
                c2 = make_divisible(c2 * gw, 8)
            name_base_conv = name_base + 'conv'
            # print(name_base_conv)
            if name == name_base_conv:
                # print(maskconv[name])
                # print(name)
                # print(name_base_conv)
                # print('........................................')
                # print(name)
                # print(maskconv[name].sum().item())
                # print(c2)
                args[-1] = maskconv[name].sum().item() / c2
                # print(args[-1])
       
        elif m is C3:
            c2 = args[0]
            if c2 != no:  # if not output
                c2_ = make_divisible(c2 * gw, 8)
                if isinstance(args[-1], float) and m is not SPP and m is not SPPF:
                    c2 = c2 * args[-1]
                c2 = make_divisible(c2 * gw, 8)
            name_base_conv1 = name_base + 'cv1.conv'
            name_base_conv2 = name_base + 'cv2.conv'
            name_base_conv3 = name_base + 'cv3.conv'   
            if name == name_base_conv1:
                # args[-3][0] = maskconv[name].sum().item() / c2 
                continue
            if name == name_base_conv2:
                args[-3][1] = maskconv[name].sum().item() / c2 
            if name == name_base_conv3:
                args[-1] = maskconv[name].sum().item() / c2 

            for j in range(n):
                name_base_m_conv1 = name_base + "m.{}.cv1.conv".format(j)
                if name == name_base_m_conv1:
                    # c1_ino = maskconv[name_base_conv1].sum().item()
                    # print(maskconv[name].sum().item(), '***************************')
                    # print(c2)
                    args[-2][j] = maskconv[name].sum().item() / (c2_ * 0.5)
        elif m is SPPF:
            c2 = args[0]
            if c2 != no: 
                if isinstance(args[-1],float):
                    c2 = c2 * args[-1]
                c2 = make_divisible(c2 * gw, 8)
            named_m_conv1 = name_base+'cv1.conv'
            named_m_conv2 = name_base+'cv2.conv'
            if name == named_m_conv1:
                args[-1] = 0.5 * maskconv[name].sum().item() /  c2

def weights_inheritance(model, candidates_pruning_model, from_to, maskconv):
    original_model_state = model.state_dict()
    pruning_model_model_state = candidates_pruning_model.state_dict()
    assert original_model_state.keys() == pruning_model_model_state.keys()
    last_id=0
    for name, module in model.named_modules():
        try:
            last_id = max(last_id, int(name.split('.')[1]))
        except:
            pass
    

    #  print(last_id)

    # for original_name, original_module in model.named_modules():
    #     print(original_name)

    for ((original_name, original_module), (pruning_name, pruning_module)) in zip(model.named_modules(), candidates_pruning_model.named_modules()):

        assert original_name == pruning_name
        if isinstance(original_module, nn.Conv2d) and  not original_name.startswith(f"model.{last_id}"):
            # print(pruning_name, '************************************')
            # print(maskconv[original_name])
            if original_name in from_to.keys():
                # print(pruning_name)
                former_name = from_to[original_name]
                # print(former_name)
                # print(former_name)
                if isinstance(former_name, str):
                    # print(maskconv[original_name])
                    out_id = np.squeeze(np.argwhere(np.asarray(maskconv[original_name].cpu().numpy())))
                    # print(out_id)
                    in_id = np.squeeze(np.argwhere(np.asarray(maskconv[former_name].cpu().numpy())))
                    w = original_module.weight.data[:, in_id, :, :].clone()
                    w = w[out_id, :, :, :].clone()
                    if len(w.shape) == 3:
                        w = np.unsqueeze(0)

                    pruning_module.weight.data = w
                if isinstance(former_name, list):
                    out_id = np.squeeze(np.argwhere(np.asarray(maskconv[original_name].cpu().numpy())))
                    # print(out_id)
                    in_id = []
                    in_id_all = []

                    for i in range(len(former_name)):
                        in_id = [j for j in range(maskconv[former_name[i]].shape[0]) if maskconv[former_name[i]][j] ==1]
                        if i > 0:
                            in_id = [k + len(in_id_all) for k in in_id]

                        in_id_all.extend(in_id)

                    w = original_module.weight.data[:, in_id_all, :, :].clone()
                    w = w[out_id, :, :, :].clone()

                    pruning_module.weight.data = w
            else:
                out_id = np.squeeze(np.argwhere(np.asarray(maskconv[original_name].cpu().numpy())))
                # print(out_id)
                w = original_module.weight.data[out_id, :, :, :].clone()
                assert len(w.shape) == 4
                pruning_module.weight.data = w
        if  isinstance(original_module, nn.BatchNorm2d):
            # print('................', pruning_name)
            out_id = np.squeeze(np.argwhere(np.asarray(maskconv[original_name[:-2] + 'conv'].cpu().numpy())))
            # print(out_id)
            pruning_module.weight.data = original_module.weight.data[out_id].clone()
            pruning_module.bias.data = original_module.bias.data[out_id].clone()
            pruning_module.running_mean = original_module.running_mean[out_id].clone()
            pruning_module.running_var = original_module.running_var[out_id].clone()
        
        if isinstance(original_module, nn.Conv2d) and  original_name.startswith(f"model.{last_id}"): # --------------------------------
            former_name = from_to[original_name]
            in_id = np.squeeze(np.argwhere(np.asarray(maskconv[former_name].cpu().numpy())))
            pruning_module.weight.data = original_module.weight.data[:, in_id, :, :]
            pruning_module.bias.data = original_module.bias.data


