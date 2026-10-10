"""Strict pretrained inference using unchanged official network forward code."""
import ast
import collections
from itertools import repeat
import json
import math
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import yaml

from runtime_v5 import sha

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'third_party'/'display_deps'))


class Registry:
    def register(self):return lambda cls:cls


def architecture(family):
    namespace={'ARCH_REGISTRY':Registry(),'torch':torch,'nn':nn,'F':F,'init':init,
               '_BatchNorm':nn.modules.batchnorm._BatchNorm,'math':math,'warnings':warnings,
               'collections':collections,'repeat':repeat,'__name__':'display_upstream'}
    util=ROOT/'third_party'/'BasicSR_display'/'basicsr'/'archs'/'arch_util.py'
    if family=='hat':
        tree=ast.parse(util.read_text(encoding='utf-8'),filename=str(util))
        names={'default_init_weights','make_layer','pixel_unshuffle','_no_grad_trunc_normal_','trunc_normal_','_ntuple'}
        tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        exec(compile(tree,str(util),'exec'),namespace)
        namespace['to_2tuple']=namespace['_ntuple'](2)
    paths={'hat':ROOT/'third_party'/'HAT'/'hat'/'archs'/'hat_arch.py',
           'swinir':ROOT/'third_party'/'SwinIR'/'models'/'network_swinir.py'}
    file=paths[family];tree=ast.parse(file.read_text(encoding='utf-8'),filename=str(file))
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and (n.level or (n.module or '').startswith('basicsr')))
               and not (isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and '__name__' in ast.unparse(n.test))]
    exec(compile(tree,str(file),'exec'),namespace)
    return namespace[{'hat':'HAT','swinir':'SwinIR'}[family]],{'architecture_file':str(file),'architecture_sha256':sha(file),'helper_sha256':sha(util)}


def load_model(spec,device='cuda'):
    if not torch.cuda.is_available() and device=='cuda':raise RuntimeError('CUDA required for this run')
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    cls,meta=architecture(spec['family'])
    if spec['family']=='swinir':
        large=spec['task']=='real_sr'
        args=dict(upscale=spec['scale'],in_chans=3,img_size=64,window_size=8,img_range=1.,
                  depths=[6]*(9 if large else 6),embed_dim=240 if large else 180,
                  num_heads=[8]*9 if large else [6]*6,mlp_ratio=2,
                  upsampler='nearest+conv' if large else 'pixelshuffle',resi_connection='3conv' if large else '1conv')
        key='params_ema' if large else 'params'
    elif spec['family']=='hat':
        config=ROOT/'third_party'/'HAT'/'options'/'test'/spec['config']
        cfg=yaml.safe_load(config.read_text(encoding='utf-8'));args=dict(cfg['network_g']);args.pop('type')
        if args['upscale']!=spec['scale']:raise ValueError('HAT native scale mismatch')
        key=cfg['path']['param_key_g'];meta.update(config_file=str(config),config_sha256=sha(config))
    else:raise ValueError('Only retained HAT/SwinIR families are supported')
    model=cls(**args)
    checkpoint=torch.load(spec['weight'],map_location='cpu',weights_only=True)
    if key not in checkpoint:raise ValueError(f'Missing official checkpoint key {key}')
    model.load_state_dict(checkpoint[key],strict=True)
    meta.update(spec=spec,weight_sha256=sha(spec['weight']),checkpoint_key=key,
                state_keys=len(checkpoint[key]),parameters=sum(p.numel() for p in model.parameters()),
                network_config=args,precision=spec.get('runtime_precision','float32')+', TF32 disabled',
                parameter_dtype='float32',core_forward_modified=False,
                adaptation='native grayscale replicated to RGB; no infrared fine tuning',device=device)
    return model.eval().requires_grad_(False).to(device),meta


def padded_tensor(x,spec):
    """Match the official inference callers, including their border conventions."""
    h,w=x.shape[-2:]
    if spec['family']=='swinir':
        ph,pw=(h//8+1)*8-h,(w//8+1)*8-w
        x=torch.cat([x,torch.flip(x,[2])],2)[...,:h+ph,:]
        x=torch.cat([x,torch.flip(x,[3])],3)[...,:w+pw]
    else:
        block=16
        x=F.pad(x,(0,(-w)%block,0,(-h)%block),'reflect')
    return x


@torch.inference_mode()
def infer(model,spec,gray,device='cuda'):
    gray=np.asarray(gray,np.float32)
    if gray.ndim!=2 or min(gray.shape)<16 or not np.isfinite(gray).all() or gray.min()<0 or gray.max()>1:
        raise ValueError('Expected normalized finite native grayscale')
    torch.cuda.synchronize();started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    x=torch.from_numpy(gray.copy())[None,None].repeat(1,3,1,1).to(device)
    x=padded_tensor(x,spec);torch.cuda.synchronize();forward_start=time.perf_counter()
    precision=spec.get('runtime_precision','float32')
    if precision not in ('float32','float16_autocast'):raise ValueError('Unsupported runtime precision')
    with torch.autocast('cuda',dtype=torch.float16,enabled=precision=='float16_autocast'):
        output=model(x)
    torch.cuda.synchronize();forward_ms=(time.perf_counter()-forward_start)*1000
    output=output[...,:gray.shape[0]*spec['scale'],:gray.shape[1]*spec['scale']]
    if output.shape[1]!=3 or not torch.isfinite(output).all():raise ValueError('Invalid model output')
    rgb=output[0].permute(1,2,0).float().cpu().numpy()
    metrics={'forward_ms':forward_ms,'inference_ms':(time.perf_counter()-started)*1000,
             'peak_cuda_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,
             'raw_min':float(rgb.min()),'raw_max':float(rgb.max()),'out_of_range_fraction':float(((rgb<0)|(rgb>1)).mean()),
             'rgb_channel_spread_mean_dn':float(np.ptp(rgb,axis=2).mean()*255),
             'native_size_wh':[rgb.shape[1],rgb.shape[0]],'native_scale':spec['scale'],'runtime_precision':precision}
    # Bound channels before luminance, retain raw numeric range separately.
    rgb=np.clip(rgb,0,1);luma=(rgb*np.array([.299,.587,.114],np.float32)).sum(axis=2)
    return luma,rgb,metrics
