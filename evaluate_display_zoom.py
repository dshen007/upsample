"""Fixed full-image operator-display screen; no feature-tracking gate."""
import argparse
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image
import torch

from display_support import prepare,resize,save_image
from display_model_adapter import infer,load_model
from display_support import ref_metrics,write_table
from runtime_v5 import atomic_json,run_lock,sha,utc_now
from prepare_display_models import model_specs
from display_support import checked_read
from zoom_restoration_metrics import image_metrics

ROOT=Path(__file__).resolve().parent


def make_input(source,scale,scenario,roi):
    if scenario=='crop':return prepare(source,scale,roi=roi)
    shape=(320,256) if scale==2 else (213,171)
    low=cv2.resize(source,shape,interpolation=cv2.INTER_AREA)
    return low,source,{'scenario':'sampling','reference_is_true_hr':False,'low_size_wh':list(shape),'output_size_wh':[640,512]}


def image_result(result,low,baseline,reference):
    result=np.rint(result*255).astype(np.uint8).astype(np.float32)/255
    metrics=image_metrics(result,low,baseline)
    metrics['bicubic_difference_mae_dn']=float(np.abs(result-baseline).mean()*255)
    if reference is not None:metrics.update(ref_metrics(result,reference))
    return result,metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--role',choices=('development','review'),required=True)
    p.add_argument('--models',nargs='+');p.add_argument('--revision',default='');p.add_argument('--denoise',action='store_true')
    a=p.parse_args();out=a.run/(a.role+a.revision);out.mkdir(exist_ok=False)
    cfg=json.loads((a.run/'protocol.json').read_text(encoding='utf-8'))
    checks=json.loads((a.run/'model_checks.json').read_text(encoding='utf-8'))
    if checks['status']!='completed' or not checks.get('passed'):raise ValueError('Models not verified')
    active={s['name'] for s in model_specs()}
    if a.models and not set(a.models)<=active:raise ValueError('Requested retired or unknown model')
    specs=[s for s in cfg['models'] if s['name'] in active and (a.models is None or s['name'] in a.models)]
    if not specs:raise ValueError('No model selected')
    rows=[];models=[];started=time.perf_counter();torch.set_num_threads(4);cv2.setNumThreads(4)
    files=('evaluate_display_zoom.py','display_model_adapter.py','display_support.py','zoom_restoration_metrics.py','ir_sr.py')
    snapshot=out/'code_snapshot';snapshot.mkdir()
    for file in files:shutil.copyfile(ROOT/file,snapshot/file)
    atomic_json(out/'protocol.json',{'objective':'operator_display','parent_sha256':sha(a.run/'protocol.json'),
        'model_names':[s['name'] for s in specs],'source_sha256':{f:sha(ROOT/f) for f in files},
        'pre_denoise':a.denoise,'role':a.role,'created_at':utc_now()})
    groups={k:v for k,v in cfg['groups'].items() if v['role']==a.role}
    sources={}
    for key,group in groups.items():
        for index in cfg['indices']:
            record=next(r for r in cfg['source_inputs'] if r['sequence']==key and int(r['frame_index'])==index)
            source=checked_read(ROOT/cfg['source_data'],record);sources[key,index]=source
            for scenario in ('crop','sampling'):
                for scale in (2,3):
                    low,reference,geom=make_input(source,scale,scenario,group['roi'][str(scale)])
                    baseline=np.rint(resize(low,(640,512))*255).astype(np.uint8).astype(np.float32)/255
                    folder=out/'images'/scenario/key/f'x{scale}'/'bicubic'
                    save_image(folder/f'frame_{index:05d}.png',baseline)
                    _,metric=image_result(baseline,low,baseline,reference)
                    rows.append(dict(sequence=key,index=index,role=a.role,scenario=scenario,scale=scale,model='bicubic',**metric))
    with run_lock(a.run/'gpu_inference.lock'):
        for spec in specs:
            model,meta=load_model(spec);models.append(meta);warmed=set()
            for key,group in groups.items():
                for scenario in ('crop','sampling'):
                    for scale in spec['display_scales']:
                        for index in cfg['indices']:
                            tick=time.perf_counter();source=sources[key,index]
                            low,reference,geom=make_input(source,scale,scenario,group['roi'][str(scale)])
                            source_low=low
                            if a.denoise:
                                from display_support import adaptive_denoise
                                low=adaptive_denoise(low,.5969774032023695,.75)
                            baseline=np.rint(resize(source_low,(640,512))*255).astype(np.uint8).astype(np.float32)/255
                            if low.shape not in warmed:
                                warm_start=time.perf_counter();infer(model,spec,low);warmed.add(low.shape)
                                tick+=time.perf_counter()-warm_start
                            native,rgb,timing=infer(model,spec,low)
                            result=resize(native,(640,512));process_ms=(time.perf_counter()-tick)*1000
                            result,metric=image_result(result,source_low,baseline,reference)
                            folder=out/'images'/scenario/key/f'x{scale}'/spec['name']
                            path=folder/f'frame_{index:05d}.png';save_image(path,result)
                            nativefolder=out/'native_rgb'/scenario/key/f'x{scale}'/spec['name'];nativefolder.mkdir(parents=True,exist_ok=True)
                            Image.fromarray(np.rint(rgb*255).astype(np.uint8)).save(nativefolder/f'frame_{index:05d}.png')
                            rows.append(dict(sequence=key,index=index,role=a.role,scenario=scenario,scale=scale,model=spec['name'],
                                             process_ms=process_ms,pre_denoise=a.denoise,**timing,**metric))
                            atomic_json(folder/f'frame_{index:05d}.json',{'source_index':index,'geometry':geom,'native_scale':spec['scale'],
                                'post_resize':spec['scale']!=scale or result.shape!=native.shape,'output_sha256':sha(path)})
                            if index in (75,149):
                                atomic_json(out/'status.json',{'status':'evaluating','model':spec['name'],'sequence':key,'scenario':scenario,
                                                              'scale':scale,'index':index,'rows':len(rows),'elapsed_seconds':time.perf_counter()-started,'updated_at':utc_now()})
                        print('Completed display',a.role,spec['name'],key,scenario,scale,flush=True)
            write_table(out/'image_metrics.csv',rows);atomic_json(out/'models.json',models)
            del model;torch.cuda.empty_cache()
    atomic_json(out/'status.json',{'status':'completed','rows':len(rows),'unique_reference_instants':len(sources),
        'models':len(specs),'elapsed_seconds':time.perf_counter()-started,'updated_at':utc_now()})


if __name__=='__main__':main()
