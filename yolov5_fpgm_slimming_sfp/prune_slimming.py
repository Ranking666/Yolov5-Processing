import argparse
from functools import update_wrapper
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
from torch.nn.modules.module import Module
os.chdir(sys.path[0])
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp, device_count
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm

from models.yolo import *
from models.common import *
from models.experimental import *
from utils.torch_utils import prune, select_device, de_parallel
from utils.prune_utils import *

from utils.general import (LOGGER, check_dataset, check_file, check_git_status, check_img_size, check_requirements,set_logging,
                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods, one_cycle,
                           print_args, print_mutation, strip_optimizer)


def prune_rand(model, cut_ids, opt, ckpt):

    cut_ids_bn = [i.replace('conv', 'bn') for i in cut_ids]
    total = 0
    for name,moudle in model.named_modules():
        if isinstance(moudle, nn.BatchNorm2d) and name in cut_ids_bn:
            total += moudle.weight.data.shape[0]

    bn_weight = torch.zeros(total)
    index =0
    for name,moudle in model.named_modules():
        if isinstance(moudle, nn.BatchNorm2d) and name in cut_ids_bn:
            size = moudle.weight.data.shape[0]
            bn_weight[index:(index+size)] = moudle.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn_weight)
    thre_index = int(total * opt.percent)
    thre = y[thre_index]

    print('thresh is', thre)


    # print('cut_ids', cut_ids)
    # input()
    pruned = 0
    cfg = {}
    cfg_mask = {}
    for name, module in model.named_modules():
        # print('********************')
        if isinstance(module, nn.BatchNorm2d):
            if name in cut_ids_bn:
                mask = get_filtermask(module, thre)
                # print(name)
                # print(mask)
            else:
                mask = torch.ones(module.weight.data.shape)
            # print('mask.shape')
            # print(mask.shape)
            # print(module.weight.data.shape)
            # module.weight.data.mul_(mask)
            # module.bias.data.mul_(mask)
            cfg[name] = torch.sum(mask)
            cfg_mask[name[:-2] + 'conv'] = mask
            # print('.........................')
    
    # input()
    with open(opt.cfg) as f:
        oriyaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

    # print(cfg_mask)
    pruned_yaml = update_pruning_yaml(oriyaml, model, cfg_mask, cut_ids)

    # with open(opt.path, "w", encoding='utf-8') as f:
    #     yaml.safe_dump(pruned_yaml,f,encoding='utf-8', allow_unicode=True, default_flow_style=True, sort_keys=False)

    compact_model = Model(pruned_yaml, nc=nc).to(device)

    weights_inheritance(model, compact_model, from_to, cfg_mask)
    with open(opt.path, "w", encoding='utf-8') as f:
        yaml.safe_dump(pruned_yaml,f,encoding='utf-8', allow_unicode=True, default_flow_style=True, sort_keys=False)
        # yaml.dump(pruned_yaml, f, Dumper=ruamel.yaml.RoundTripDumper)
    # with open(opt.path[:-5]+'_.yaml', "w", encoding='utf-8') as fd:
    #     yaml.safe_dump(pruned_yaml,fd,encoding='utf-8', allow_unicode=True, sort_keys=False)
    ckpt = {'epoch': -1,
            'best_fitness':0,
            'model': deepcopy(de_parallel(compact_model)).half(),
            'ema': None,
            'updates': None,
            'optimizer': None,
            'wandb_id': None}
    torch.save(ckpt, opt.weights[:-3]+'-Slimming_pruned.pt')
                    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default= '/home/lhf/yolov5-master/runs/train/exp51/weights/last.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5s_pruning.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/VisDrone.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default= 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    # parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    # parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    # parser.add_argument('--name', default='exp', help='save to project/name')

    # parser.add_argument('--min_ratio', type=float, default=0.5, help='min parameters remain ratio')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    # parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--path', type=str, default='models/yolov5s_pruned.yaml', help='pruned model path')
    # parser.add_argument('--candidates_num', type=int, default=700, help='the number of candidates')
    parser.add_argument('--pruned_weights', type=str, default= 'runs/prune/pruned_weight_slimming.pt', help='initial weights path')
    parser.add_argument('--percent', type=float, default=0.7, help='min parameters remain ratio')

    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)
    opt.hyp = check_yaml(opt.hyp)
    set_logging()
    if isinstance(opt.hyp, str):
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    device = select_device(opt.device)

    # load repruning model

    data_dict = check_dataset(opt.data)
    nc = 1 if opt.single_cls else int(data_dict['nc'])
    # print('................, ncncncn')
    # print(nc)
    # input()
    model = Model(opt.cfg, nc=nc).to(device)

    check_suffix(opt.weights, '.pt')  # check weights
    ckpt = torch.load(opt.weights, map_location='cpu')
    exclude =  []  # exclude keys
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32

    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    for k, v in csd.items():
        print(k)
    model.load_state_dict(csd, strict=True)  # load
    # prunging models
    cut_ids, ignore_ids, from_to = module_pruning_layer(model.yaml)
    # rand_prune_and_eval(model,ignore_idx,opt)
    # print(cut_ids)

    prune_rand(model, cut_ids, opt, ckpt)
