"""Strict pretrained loading; official FP32 or explicitly recorded optimized inference."""
import ast
import collections
from itertools import repeat
import json
import math
from pathlib import Path
import sys
import time
from types import MethodType
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


PRECISIONS=('float32','float16_autocast','float16_safe','bfloat16_autocast')
ATTENTION_BACKENDS=('official','sdpa')
ERROR_LIMITS={'mean_abs_dn':.10,'p99_abs_dn':.5,'max_abs_dn':3}
TEMPORAL_ERROR_LIMITS={'mean_abs_dn':.20,'p99_abs_dn':1,'max_abs_dn':6}
HAT_FP32_CONVS=('conv_first',*(f'layers.{i}.conv' for i in range(6)),
                'conv_after_body','conv_before_upsample.0','upsample.0','upsample.2','conv_last')


def within_error_limits(metrics,temporal=False):
    limits=TEMPORAL_ERROR_LIMITS if temporal else ERROR_LIMITS
    return isinstance(metrics,dict) and all(type(metrics.get(k)) in (int,float)
        and math.isfinite(metrics[k]) and 0<=metrics[k]<=v for k,v in limits.items())


class FP32Conv2d(nn.Conv2d):
    def forward(self,x):
        with torch.autocast('cuda',enabled=False):
            return super().forward(x.float())


def fp32_block_forward(self,x,*args,**kwargs):
    with torch.autocast('cuda',enabled=False):
        return self._display_fp32_forward(x.float(),*args,**kwargs)


def hat_attention(q,k,v,bias,scale):
    dim=q.shape[-1];padding=(-dim)%8
    # HAT has 30 channels per head; the efficient CUDA kernel needs a multiple of 8.
    q=q*scale
    q,k,v=(F.pad(t,(0,padding)) for t in (q,k,v))
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        value=F.scaled_dot_product_attention(q,k,v,attn_mask=bias.to(q.dtype),dropout_p=0.,scale=1.)
    return value[...,:dim]


def hat_window_sdpa(self,x,rpi,mask=None):
    b,n,c=x.shape
    qkv=self.qkv(x).reshape(b,n,3,self.num_heads,c//self.num_heads).permute(2,0,3,1,4)
    q,k,v=qkv.unbind(0)
    bias=self.relative_position_bias_table[rpi.reshape(-1)].view(n,n,-1).permute(2,0,1).contiguous()[None]
    if mask is not None:
        bias=(bias+mask[:,None]).repeat(b//mask.shape[0],1,1,1)
    value=hat_attention(q,k,v,bias,self.scale).transpose(1,2).reshape(b,n,c)
    return self.proj_drop(self.proj(value))


def hat_ocab_sdpa(self,x,x_size,rpi):
    h,w=x_size;b,_,c=x.shape;ws=self.window_size;ows=self.overlap_win_size
    shortcut=x;x=self.norm1(x).view(b,h,w,c)
    qkv=self.qkv(x).reshape(b,h,w,3,c).permute(3,0,4,1,2)
    q=qkv[0].permute(0,2,3,1).reshape(b,h//ws,ws,w//ws,ws,c)
    q=q.permute(0,1,3,2,4,5).reshape(-1,ws*ws,c)
    kv=self.unfold(torch.cat((qkv[1],qkv[2]),dim=1))
    kv=kv.reshape(b,2,c,ows*ows,-1).permute(1,0,4,3,2).reshape(2,-1,ows*ows,c).contiguous()
    bw,nq,_=q.shape;nk=ows*ows;d=c//self.num_heads
    q=q.reshape(bw,nq,self.num_heads,d).permute(0,2,1,3)
    k,v=(t.reshape(bw,nk,self.num_heads,d).permute(0,2,1,3) for t in kv.unbind(0))
    bias=self.relative_position_bias_table[rpi.reshape(-1)].view(nq,nk,-1).permute(2,0,1).contiguous()[None]
    value=hat_attention(q,k,v,bias,self.scale).transpose(1,2).reshape(bw,ws,ws,c)
    value=value.reshape(b,h//ws,w//ws,ws,ws,c).permute(0,1,3,2,4,5).reshape(b,h*w,c)
    x=self.proj(value)+shortcut
    return x+self.mlp(self.norm2(x))


def require_runtime_qualification(run,spec,report,video=False):
    precision=spec.get('runtime_precision','float32');attention=spec.get('attention_backend','official')
    if precision=='float32' and attention=='official':return None
    if report is None:raise ValueError('Non-default inference requires --precision-checks from check_display_precision.py')
    report=Path(report);checks=json.loads(report.read_text(encoding='utf-8'))
    hashes={'protocol_sha256':sha(Path(run)/'protocol.json'),'inference_adapter_sha256':sha(__file__),
            'qualification_checker_sha256':sha(ROOT/'check_display_precision.py'),
            'support_sha256':sha(ROOT/'display_support.py'),'reader_sha256':sha(ROOT/'ir_sr.py')}
    if checks['status']!='completed' or any(checks.get(k)!=v for k,v in hashes.items()):
        raise ValueError('Qualification is incomplete or belongs to different code/protocol')
    environment=checks.get('environment',{})
    expected={'torch_version':torch.__version__,'gpu_name':torch.cuda.get_device_name(0),
              'cuda_version':torch.version.cuda,'cudnn_benchmark':torch.backends.cudnn.benchmark}
    if any(environment.get(k)!=v for k,v in expected.items()):raise ValueError('Runtime environment differs from qualification')
    matches=[m for m in checks['models'] if m['model']==spec['name'] and m['precision']==precision
             and m.get('attention_backend')==attention and m.get('weight_sha256')==sha(spec['weight'])]
    if not matches or not all(m['passed'] and m['cases'] and all(c['passed'] for c in m['cases']) for m in matches):
        raise ValueError('Requested model/precision/attention has not passed qualification')
    architecture_file=ROOT/'third_party'/('HAT/hat/archs/hat_arch.py' if spec['family']=='hat' else 'SwinIR/models/network_swinir.py')
    for match in matches:
        meta=match['runtime_metadata']
        if meta['architecture_sha256']!=sha(architecture_file) or meta['helper_sha256']!=sha(ROOT/'third_party/BasicSR_display/basicsr/archs/arch_util.py'):
            raise ValueError('Official architecture/helpers differ from qualification')
        if spec['family']=='hat' and meta['config_sha256']!=sha(ROOT/'third_party/HAT/options/test'/spec['config']):
            raise ValueError('Official HAT configuration differs from qualification')
    cases=[c for m in matches for c in m['cases']]
    for case in cases:
        errors=[case,case.get('display_error'),*case.get('repeat_native_errors',[]),*case.get('repeat_display_errors',[])]
        if not all(within_error_limits(e) for e in errors):raise ValueError('Qualification error exceeds current limits')
        if case.get('runtime_error_change') is not None and not within_error_limits(case['runtime_error_change'],temporal=True):
            raise ValueError('Qualification temporal error exceeds current limits')
    groups=json.loads((Path(run)/'protocol.json').read_text(encoding='utf-8'))['groups']
    required={k for k,g in groups.items() if g['role']=='development'}
    if not all(any(c['sequence']==k and c['role']=='development' and c['scale']==s for c in cases)
               for k in required for s in spec['display_scales']):
        raise ValueError('Qualification lacks full development ROI coverage')
    if video:
        for key,group in groups.items():
            if group['role']!='review':continue
            for scale in spec['display_scales']:
                frames={c['index']:c for c in cases if c['sequence']==key and c['role']=='review' and c['scale']==scale}
                pairs={i for i,c in frames.items() if i-1 in frames and c.get('runtime_error_change') is not None}
                if len(pairs)<3:raise ValueError('Video qualification needs at least three adjacent frame pairs per review ROI')
    return {'file':str(report.resolve()),'sha256':sha(report)}


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
    precision=spec.get('runtime_precision','float32');attention=spec.get('attention_backend','official')
    if precision not in PRECISIONS or attention not in ATTENTION_BACKENDS:raise ValueError('Unsupported runtime configuration')
    if (precision=='float16_safe' or attention=='sdpa') and spec['name']!='hat_real_x4':
        raise ValueError('Selective FP16 and SDPA are currently implemented only for hat_real_x4')
    if precision=='bfloat16_autocast' and not torch.cuda.is_bf16_supported():raise ValueError('CUDA BF16 is not supported')
    protected=[]
    if precision=='float16_safe':
        # Preserve feature/residual/reconstruction convolutions.
        protected=list(HAT_FP32_CONVS)
        for name in protected:model.get_submodule(name).__class__=FP32Conv2d
    if attention=='sdpa':
        for module in model.modules():
            if type(module).__name__=='WindowAttention':module.forward=MethodType(hat_window_sdpa,module)
            elif type(module).__name__=='OCAB':module.forward=MethodType(hat_ocab_sdpa,module)
    protected_blocks=[]
    if precision=='float16_safe':
        # Stress frames are sensitive to early features and overlapping cross-window attention.
        for name,module in model.named_modules():
            if name=='layers.0' or type(module).__name__=='OCAB':
                module._display_fp32_forward=module.forward
                module.forward=MethodType(fp32_block_forward,module);protected_blocks.append(name)
    meta.update(spec=spec,weight_sha256=sha(spec['weight']),checkpoint_key=key,
                state_keys=len(checkpoint[key]),parameters=sum(p.numel() for p in model.parameters()),
                network_config=args,precision=precision+', TF32 disabled',
                parameter_dtype='float32',core_forward_modified=bool(protected) or attention!='official',
                protected_fp32_modules=protected,protected_fp32_blocks=protected_blocks,attention_backend=attention,
                inference_adapter_file=str(Path(__file__).resolve()),inference_adapter_sha256=sha(__file__),
                adaptation='native grayscale replicated to RGB; no infrared fine tuning',device=device)
    model._display_precision=precision;model._display_attention=attention
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
    if precision not in PRECISIONS:raise ValueError('Unsupported runtime precision')
    if precision!=model._display_precision or spec.get('attention_backend','official')!=model._display_attention:
        raise ValueError('Inference settings differ from loaded model')
    if precision=='float16_safe' and any(type(model.get_submodule(n)) is not FP32Conv2d for n in HAT_FP32_CONVS):
        raise ValueError('Load the model with float16_safe before inference')
    dtype=torch.bfloat16 if precision=='bfloat16_autocast' else torch.float16
    with torch.autocast('cuda',dtype=dtype,enabled=precision!='float32'):
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
