"""Verify official strict loading and caller equivalence before visual screening."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from display_model_adapter import ROOT, infer, load_model
from runtime_v5 import atomic_json,utc_now
from prepare_display_models import model_specs


def reference_output(model,spec,gray):
    rgb=np.repeat(gray[:,:,None],3,axis=2)
    x=torch.from_numpy(rgb.transpose(2,0,1).copy())[None].cuda()
    if spec['family']=='swinir':
        file=ROOT/'third_party'/'SwinIR'/'main_test_swinir.py'
        tree=ast.parse(file.read_text(encoding='utf-8'))
        test=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test')
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        block=next(n for n in main.body if isinstance(n,ast.For))
        caller=next(n for n in block.body if isinstance(n,ast.With))
        ns={'torch':torch,'img_lq':x,'model':model,'window_size':8,'args':SimpleNamespace(scale=spec['scale'],tile=None)}
        exec(compile(ast.Module(body=[test,caller],type_ignores=[]),str(file),'exec'),ns)
        return ns['output']
    if spec['family']!='hat':raise ValueError('Unsupported retained family')
    file=ROOT/'third_party/HAT/hat/models/hat_model.py'
    tree=ast.parse(file.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HATModel')
    cls.bases=[];cls.decorator_list=[]
    cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('pre_process','process','post_process')]
    ns={'torch':torch,'F':F,'np':np};exec(compile(ast.Module(body=[cls],type_ignores=[]),str(file),'exec'),ns)
    caller=ns['HATModel']();caller.opt={'network_g':{'window_size':16},'scale':spec['scale']};caller.lq=x;caller.net_g=model
    caller.pre_process();caller.process();caller.post_process();return caller.output



def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.run/'protocol.json').read_text(encoding='utf-8'));records=[]
    if not {s['name'] for s in protocol['models']} <= {s['name'] for s in model_specs()}:
        raise ValueError('Historical protocol contains retired models; prepare a new four-model run. Original model_checks.json is preserved.')
    torch.set_num_threads(4);torch.manual_seed(0)
    for spec in protocol['models']:
        model,meta=load_model(spec);checks=[]
        for h,w in ((32,32),(33,35)):
            gray=np.linspace(.03,.97,h*w,dtype=np.float32).reshape(h,w);original=gray.copy()
            with torch.inference_mode():reference=reference_output(model,spec,gray).clamp(0,1)[0].permute(1,2,0).cpu().numpy()
            luma,rgb,timing=infer(model,spec,gray)
            if rgb.shape!=(h*spec['scale'],w*spec['scale'],3):raise ValueError('Native shape mismatch')
            error=float(np.max(np.abs(reference-rgb)))
            if error>1e-5:raise ValueError(f'Official caller discrepancy {spec["name"]}: {error}')
            np.testing.assert_array_equal(gray,original)
            checks.append({'input_shape_hw':[h,w],'official_caller_max_error':error,'finite':bool(np.isfinite(rgb).all())})
        records.append({'model':spec['name'],'load':meta,'checks':checks})
        atomic_json(a.run/'model_checks.json',{'status':'checking','models':records,'updated_at':utc_now()})
        print('Verified',spec['name'],meta['parameters'],checks,flush=True)
        del model;torch.cuda.empty_cache()
    atomic_json(a.run/'model_checks.json',{'status':'completed','models':records,'passed':True,'updated_at':utc_now()})


if __name__=='__main__':main()
