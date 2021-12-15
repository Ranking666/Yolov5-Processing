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
from utils.general import set_logging, check_suffix, check_dataset
from utils.torch_utils import prune, select_device, intersect_dicts
from utils.pruning_utils import *


def prune_rand(model, cut_ids, opt, ckpt):
    orignate_flops = model.flops
    with open(opt.cfg) as f:
        orignate_yaml = yaml.safe_load(f)

    # print(orignate_yaml)

    max_ratio = 1.0

    ABE = AdaptiveBN_Eval(model, opt, device, hyp)
    best_fitness = 0
    maskbn = {}
    maskconv = {}

    candidates = 0
    
    while True:
        pruning_yaml = deepcopy(orignate_yaml)
        # print(pruning_yaml)
        for name, module in model.named_modules():
            # print(name)
            # print(module)
            if isinstance(module, nn.Conv2d):
                if name in cut_ids:
                    ratio = random.uniform(opt.min_ratio, max_ratio)
                    mask = filtermask(module, ratio)
                else:
                    mask = torch.ones(module.weight.data.size()[0])
                # print(mask.shape, '\n')
                maskbn[name[:-4] + 'bn'] = mask
                maskconv[name] = mask

        pruning_yaml = update_pruning_yaml(pruning_yaml, maskconv, cut_ids)
        # for key, value in pruning_yaml.items():
        #     print(key)
        #     print(value, '\n')
        pruning_yaml_model = deepcopy(pruning_yaml)
        candidates_pruning_model = Model(pruning_yaml_model).to(device)
        del pruning_yaml_model
        current_flops =candidates_pruning_model.flops

        weights_inheritance(model, candidates_pruning_model, from_to, maskconv)
        print('ok') 
        
        results = ABE(candidates_pruning_model)
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        if fi > best_fitness:
            best_fitness = fi
            with open(opt.path, "w", encoding='utf-8') as f:
                yaml.safe_dump(pruning_yaml,f, encoding= 'utf-8', allow_unicode=True, default_flow_style=True, sort_keys=False)


            pruned_ckpt = {'epoch': -1,
            'best_fitness': best_fitness,
            'model': deepcopy(de_parallel(candidates_pruning_model)).half(),
            'ema': ckpt['ema'],
            'updates': ckpt['updates'],
            'optimizer': ckpt['optimizer'],
            'wandb_id': ckpt['wandb_id']}
                
            torch.save(pruned_ckpt, opt.pruned_weights)
        candidates = candidates + 1
        del pruning_yaml
        del candidates_pruning_model

        if candidates > opt.candidates_num:
            break

                    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default= '/home/lhf/yolov5-master/runs/train/exp51/weights/last.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5s_pruning.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/VisDrone.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default= 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')

    parser.add_argument('--min_ratio', type=float, default=0.5, help='min parameters remain ratio')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--path', type=str, default='models/yolov5s_pruned.yaml', help='pruned model path')
    parser.add_argument('--candidates_num', type=int, default=700, help='the number of candidates')
    parser.add_argument('--pruned_weights', type=str, default= 'runs/pruned_weight.pt', help='initial weights path')

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

    model = Model(opt.cfg, nc=nc).to(device)

    check_suffix(opt.weights, '.pt')  # check weights
    ckpt = torch.load(opt.weights, map_location=device)
    exclude =  []  # exclude keys
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32

    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    for k, v in csd.items():
        print(k)
    model.load_state_dict(csd, strict=False)  # load
    # prunging models
    cut_ids, ignore_ids, from_to = module_pruning_layer(model.yaml)
    # rand_prune_and_eval(model,ignore_idx,opt)
    # print(cut_ids)

    prune_rand(model, cut_ids, opt, ckpt)
