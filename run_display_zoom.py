"""Single-image operator-display inference with recorded source and model."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from display_support import prepare,resize,save_image
from display_model_adapter import infer,load_model,PRECISIONS,ATTENTION_BACKENDS,require_runtime_qualification
from ir_sr import read_gray
from prepare_display_models import model_specs
from runtime_v5 import atomic_json,run_lock,sha,utc_now


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--model',required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--scale',type=int,choices=(2,3),required=True);p.add_argument('--roi',nargs=4,type=int);p.add_argument('--bits',type=int,default=8)
    p.add_argument('--precision',choices=PRECISIONS,default='float32')
    p.add_argument('--attention-backend',choices=ATTENTION_BACKENDS,default='official')
    p.add_argument('--precision-checks',type=Path)
    a=p.parse_args();cfg=json.loads((a.run/'protocol.json').read_text(encoding='utf-8'))
    if a.model not in {s['name'] for s in model_specs()}:raise ValueError('Model retired or unknown; use HAT or SwinIR')
    spec=next(s for s in cfg['models'] if s['name']==a.model)
    if a.scale not in spec['display_scales']:raise ValueError('This model was not configured for the requested display scale')
    spec={**spec,'runtime_precision':a.precision,'attention_backend':a.attention_backend}
    qualification=require_runtime_qualification(a.run,spec,a.precision_checks)
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    with run_lock(a.run/'gpu_inference.lock'):
        model,meta=load_model(spec)
        source=read_gray(a.input,a.bits);low,_,geom=prepare(source,a.scale,roi=a.roi)
        infer(model,spec,low)
        tick=time.perf_counter();source=read_gray(a.input,a.bits);low,_,geom=prepare(source,a.scale,roi=a.roi)
        native,rgb,timing=infer(model,spec,low);result=resize(native,(640,512))
        ready_ms=(time.perf_counter()-tick)*1000
        save_image(a.out/'超分结果.png',result);save_image(a.out/'双三次.png',resize(low,(640,512)))
        Image.fromarray(np.rint(rgb*255).astype(np.uint8)).save(a.out/'网络原生RGB输出.png')
        atomic_json(a.out/'metadata.json',{'objective':'operator_display','input':str(a.input.resolve()),'source_sha256':sha(a.input),
            'model':meta,'geometry':geom,'timing':timing,'warm_image_read_to_ready_ms':ready_ms,'runtime_qualification':qualification,
            'post_resize':spec['scale']!=a.scale or native.shape!=(512,640),'training':False,'created_at':utc_now()})
        atomic_json(a.out/'status.json',{'status':'completed','updated_at':utc_now()})


if __name__=='__main__':main()
