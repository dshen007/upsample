"""Continuous operator-display comparison without ORB/VINS ranking."""
import argparse
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import torch

from display_support import prepare,resize,save_image
from display_model_adapter import infer,load_model
from display_support import load_geometry,temporal_metrics,write_table
from runtime_v5 import atomic_json,run_lock,sha,utc_now
from display_support import checked_read
from zoom_restoration_metrics import image_metrics
from prepare_display_models import model_specs

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--models',nargs='+',required=True)
    p.add_argument('--revision',default='');p.add_argument('--precision',choices=('float32','float16_autocast'),default='float32')
    a=p.parse_args();out=a.run/('video'+a.revision)
    if not set(a.models)<={s['name'] for s in model_specs()}:raise ValueError('Requested retired or unknown model')
    cfg=json.loads((a.run/'protocol.json').read_text(encoding='utf-8'))
    specs=[s for s in cfg['models'] if s['name'] in a.models]
    if len(specs)!=len(a.models):raise ValueError('Unknown model')
    if a.precision!='float32':
        checks=json.loads((a.run/'precision_checks.json').read_text(encoding='utf-8'))
        if checks['status']!='completed' or any(not any(c['model']==s['name'] and c['passed'] for c in checks['models']) for s in specs):
            raise ValueError('Requested mixed precision has not passed full-size development checks')
    out.mkdir(exist_ok=False)
    specs=[{**s,'runtime_precision':a.precision} for s in specs]
    torch.set_num_threads(4);cv2.setNumThreads(4);rows=[];started=time.perf_counter()
    snapshot=out/'code_snapshot';snapshot.mkdir()
    for n in ('evaluate_display_video.py','display_model_adapter.py','display_support.py','zoom_restoration_metrics.py','ir_sr.py','runtime_v5.py','prepare_display_models.py'):shutil.copyfile(ROOT/n,snapshot/n)
    atomic_json(out/'protocol.json',{'objective':'operator_display','models':a.models,'scenario':'crop','future_frame_wait_ms':0,'runtime_precision':a.precision,
        'vins_gate':False,'parent_sha256':sha(a.run/'protocol.json'),'created_at':utc_now(),
        'timing_scope':'cached source to prepared full output; excludes decoding/PNG/video encoding; inference time excludes loading/warmup',
        'temporal_proxy':'source SIFT homography, diagnostic only; moving targets and terrain parallax can contaminate residual'})
    groups={k:v for k,v in cfg['groups'].items() if v['role']=='review'}
    with run_lock(a.run/'gpu_inference.lock'):
        for spec in specs:
            net,meta=load_model(spec);atomic_json(out/(spec['name']+'_model.json'),meta)
            for key,group in groups.items():
                records=[r for r in cfg['source_inputs'] if r['sequence']==key]
                if len(records)!=150:raise ValueError('Expected 150 consecutive frames')
                source=[checked_read(ROOT/cfg['source_data'],r) for r in records]
                bag,clip=key.split('/');hs,_=load_geometry(out,bag,clip,source,[r['png_sha256'] for r in records],ROOT/'data'/'display_protocol_v1')
                for scale in spec['display_scales']:
                    low,_,geom=prepare(source[0],scale,roi=group['roi'][str(scale)])
                    infer(net,spec,low);matrix=np.asarray(geom['source_to_output']);inv=np.linalg.inv(matrix)
                    previous={};writers={}
                    for name in ('bicubic',spec['name']):
                        file=out/'videos'/key/f'x{scale}'/(name+'.mp4');file.parent.mkdir(parents=True,exist_ok=True)
                        writer=cv2.VideoWriter(str(file),cv2.VideoWriter_fourcc(*'mp4v'),25,(640,512),False)
                        if not writer.isOpened():raise RuntimeError('Video writer unavailable')
                        writers[name]=writer
                    try:
                        for index,x in enumerate(source):
                            tick=time.perf_counter();low,_,_=prepare(x,scale,roi=group['roi'][str(scale)])
                            native,_,timing=infer(net,spec,low);result=resize(native,(640,512));pipeline_ms=(time.perf_counter()-tick)*1000
                            outputs={'bicubic':resize(low,(640,512)),spec['name']:result}
                            hom=matrix@hs[index]@inv if hs[index] is not None else None
                            for name,value in outputs.items():
                                value=np.rint(value*255).astype(np.uint8).astype(np.float32)/255
                                row=dict(sequence=key,index=index,scale=scale,model=name)
                                if name!='bicubic':row.update(timing,process_ms=pipeline_ms)
                                row.update(temporal_metrics(previous.get(name),value,hom))
                                if index in cfg['indices']:row.update(image_metrics(value,low,outputs['bicubic']))
                                rows.append(row);previous[name]=value
                                save_image(out/'images'/key/f'x{scale}'/name/f'frame_{index:05d}.png',value)
                                writers[name].write(np.rint(value*255).astype(np.uint8))
                            if index%25==0:
                                write_table(out/'frame_metrics.csv',rows)
                                atomic_json(out/'status.json',{'status':'evaluating','model':spec['name'],'sequence':key,'scale':scale,'index':index,
                                    'rows':len(rows),'elapsed_seconds':time.perf_counter()-started,'updated_at':utc_now()})
                    finally:
                        for writer in writers.values():writer.release()
                    print('Completed video',spec['name'],key,scale,flush=True)
            del net;torch.cuda.empty_cache()
    write_table(out/'frame_metrics.csv',rows)
    atomic_json(out/'status.json',{'status':'completed','rows':len(rows),'models':a.models,'source_frames':450,
                                  'elapsed_seconds':time.perf_counter()-started,'updated_at':utc_now()})


if __name__=='__main__':main()
