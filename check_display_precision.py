"""Qualify optional CUDA mixed precision against full-size FP32 development outputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from display_support import prepare
from display_model_adapter import ROOT,load_model,infer
from runtime_v5 import atomic_json,run_lock,utc_now
from display_support import checked_read
from prepare_display_models import model_specs


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--models',nargs='+',required=True);a=p.parse_args()
    cfg=json.loads((a.run/'protocol.json').read_text(encoding='utf-8'));checks=[];torch.set_num_threads(4)
    available={s['name'] for s in cfg['models']} & {s['name'] for s in model_specs()}
    if not set(a.models)<=available:raise ValueError('Requested retired, unknown or unconfigured model')
    with run_lock(a.run/'gpu_inference.lock'):
        for spec in cfg['models']:
            if spec['name'] not in a.models:continue
            model,meta=load_model(spec);runtime={**spec,'runtime_precision':'float16_autocast'};rows=[]
            for key,group in cfg['groups'].items():
                if group['role']!='development':continue
                record=next(r for r in cfg['source_inputs'] if r['sequence']==key and int(r['frame_index'])==75)
                source=checked_read(ROOT/cfg['source_data'],record)
                for scale in spec['display_scales']:
                    low,_,_=prepare(source,scale,roi=group['roi'][str(scale)])
                    ref,_,fp32=infer(model,spec,low)
                    try:
                        infer(model,runtime,low)
                        pred,_,fp16=infer(model,runtime,low);difference=np.abs(pred-ref)*255
                        limits={'mean_abs_dn':float(difference.mean()),'p99_abs_dn':float(np.percentile(difference,99)),'max_abs_dn':float(difference.max())}
                        passed=limits['mean_abs_dn']<=.10 and limits['p99_abs_dn']<=.5 and limits['max_abs_dn']<=3
                        rows.append(dict(sequence=key,scale=scale,passed=passed,**limits,fp32_inference_ms=fp32['inference_ms'],fp16_inference_ms=fp16['inference_ms']))
                    except (RuntimeError,ValueError) as error:rows.append(dict(sequence=key,scale=scale,passed=False,error=str(error)))
            checks.append({'model':spec['name'],'precision':'float16_autocast','passed':all(r['passed'] for r in rows),'cases':rows,'source':'development bag only'})
            atomic_json(a.run/'precision_checks.json',{'status':'checking','models':checks,'updated_at':utc_now()})
            print('Precision',spec['name'],checks[-1]['passed'],flush=True);del model;torch.cuda.empty_cache()
    atomic_json(a.run/'precision_checks.json',{'status':'completed','models':checks,'updated_at':utc_now()})


if __name__=='__main__':main()
