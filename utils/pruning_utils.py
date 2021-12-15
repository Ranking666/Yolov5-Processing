import argparse
import logging
import math
import os
import random
import sys
import time
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
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_git_status, check_img_size, check_requirements, \
    check_file, check_yaml, check_suffix, print_args, print_mutation, set_logging, one_cycle, colorstr, methods
from utils.downloads import attempt_download
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device, \
    torch_distributed_zero_first
from utils.datasets import create_dataloader
from utils.autoanchor import check_anchors
from utils.loss import ComputeLoss
from tqdm import tqdm
from utils.metrics import fitness
import val
from utils.general import make_divisible

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

def filtermask(module, ratio):
    weight = module.weight.data.abs().clone()
    weight = torch.sum(weight, dim=(1, 2, 3))
    length = weight.size()[0]
    # print(length)
    remin_length = int(length * ratio)
    remin_length = make_divisible(remin_length, 8)
    # print(remin_length)
    # print(remin_length, '...............')
    # if remin_length < 2:
    #     remin_length = 2
    
    _, index = torch.topk(weight, remin_length)
    mask = torch.zeros(length)
    mask[index] = 1
    # print(mask.shape)
    return mask

def update_pruning_yaml(pruning_yaml, maskconv, cut_ids):
    for name in cut_ids:
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
            if name == name_base_conv:
                args[-1] = maskconv[name].sum().item() / c2
       
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
        if isinstance(original_module, nn.Conv2d) and str(last_id) not in original_name:
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
        
        if isinstance(original_module, nn.Conv2d) and str(last_id) in original_name: # --------------------------------
            former_name = from_to[original_name]
            in_id = np.squeeze(np.argwhere(np.asarray(maskconv[former_name].cpu().numpy())))
            pruning_module.weight.data = original_module.weight.data[:, in_id, :, :]
            pruning_module.bias.data = original_module.bias.data

class AdaptiveBN_Eval(object):
    def __init__(self, model,opt,device,hyp):
        super().__init__()
        self.model = model
        self.opt = opt
        self.device = device
        # print(device)
        self.hyp = hyp
        batch_size = opt.batch_size
        cuda = device.type != 'cpu'
        # print(cuda)
        RANK = int(os.getenv('RANK', -1))
        print('.........', RANK)
        init_seeds(1 + RANK)
        with torch_distributed_zero_first(RANK):
            data_dict =  check_dataset(opt.data)  # check if None
        train_path, val_path = data_dict['train'], data_dict['val']
        nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
        names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
        assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {opt.data}'  # check

        # Image sizes
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
        imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

        train_loader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt.single_cls,
                                              hyp=hyp, rank=RANK,workers=opt.workers)
        mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
        nb = len(train_loader)  # number of batches
        assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {opt.data}. Possible class labels are 0-{nc - 1}'
        if RANK in [-1, 0]:
            val_loader = create_dataloader(val_path, imgsz, batch_size, gs, opt.single_cls,
                                       hyp=hyp, rank=-1,workers=opt.workers, pad=0.5)[0]

        
            labels = np.concatenate(dataset.labels, 0)
            check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision
        
        hyp['box'] *= 3. / nl  # scale to layers
        hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
        hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
        hyp['label_smoothing'] = opt.label_smoothing
        model.nc = nc  # attach number of classes to model
        model.hyp = hyp  # attach hyperparameters to model
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
        model.names = names

        # Start training
        t0 = time.time()
        nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        last_opt_step = -1
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        # scaler = amp.GradScaler(enabled=cuda)
        # stopper = EarlyStopping(patience=opt.patience)
        # compute_loss = ComputeLoss(model)  # init loss class
        self.train_loader = train_loader
        self.nb = nb
        self.rank = RANK
        self.device = device
        self.cuda = cuda
        self.batch_size = batch_size
        self.data_dict = data_dict
        self.single_cls = opt.single_cls
        self.nc = nc
        self.val_loader = val_loader
        self.imgsz = imgsz


    def __call__(self, candidates_pruning_model):

        candidates_pruning_model.train()
        pbar = enumerate(self.train_loader)
        if self.rank in [-1, 0]:
            pbar = tqdm(pbar, total=self.nb)  # progress bar

        for i, (imgs, targets, paths, _) in pbar:

            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            with amp.autocast(enabled=self.cuda):
                pred = candidates_pruning_model(imgs)  # forward
                # print('..............', pred)

            
            if i > 30:
                break


        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        maps = np.zeros(self.nc)  # mAP per class
        
        results, maps, _ = val.run(self.data_dict,
                                    batch_size=self.batch_size,
                                    imgsz=self.imgsz,
                                    model=candidates_pruning_model,
                                    single_cls=self.single_cls,
                                    dataloader=self.val_loader,plots=False)

        return results

        # fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        #     if fi > best_fitness:
        #         best_fitness = fi
        

            


        
