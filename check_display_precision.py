"""Qualify optional inference runtimes against unchanged official FP32 outputs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from display_support import prepare,resize,save_image,checked_read
from display_model_adapter import ROOT,load_model,infer,PRECISIONS,ATTENTION_BACKENDS,ERROR_LIMITS,TEMPORAL_ERROR_LIMITS,within_error_limits
from runtime_v5 import atomic_json,run_lock,utc_now,sha
from prepare_display_models import model_specs


def error_metrics(pred,ref):
    difference=np.abs(pred-ref)*255
    return {'mean_abs_dn':float(difference.mean()),'p99_abs_dn':float(np.percentile(difference,99)),
            'max_abs_dn':float(difference.max())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--models',nargs='+',required=True)
    p.add_argument('--precisions',nargs='+',choices=PRECISIONS,default=['float16_autocast'])
    p.add_argument('--attention-backends',nargs='+',choices=ATTENTION_BACKENDS,default=['official'])
    p.add_argument('--roles',nargs='+',choices=('development','review'),default=['development'])
    p.add_argument('--indices',nargs='+',type=int,default=[75]);p.add_argument('--consecutive',type=int,default=0)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--out',type=Path,help='New qualification JSON; existing files are never overwritten')
    a=p.parse_args()
    if a.repeats<1 or a.warmup<1 or a.consecutive<0:raise ValueError('Invalid repeat, warmup or consecutive count')
    cfg=json.loads((a.run/'protocol.json').read_text(encoding='utf-8'));checks=[];torch.set_num_threads(4)
    available={s['name'] for s in cfg['models']} & {s['name'] for s in model_specs()}
    if not set(a.models)<=available:raise ValueError('Requested retired, unknown or unconfigured model')
    report=a.out or a.run/'precision_checks.json'
    if report.exists():raise FileExistsError(report)
    images=report.with_name(report.stem+'_images');images.mkdir(parents=True,exist_ok=False)
    protocol_hash=sha(a.run/'protocol.json');started=time.perf_counter()
    manifest={}
    for key,group in cfg['groups'].items():
        if group['role'] not in a.roles:continue
        records={int(r['frame_index']):r for r in cfg['source_inputs'] if r['sequence']==key}
        indices=sorted(set(a.indices)|set(range(a.indices[0],a.indices[0]+a.consecutive)))
        if not set(indices)<=records.keys():raise ValueError('Requested frames are not in the frozen protocol')
        manifest[key]={'group':group,'records':[records[i] for i in indices]}
    if not manifest:raise ValueError('No source groups selected')
    environment={'torch_version':torch.__version__,'gpu_name':torch.cuda.get_device_name(0),
                 'cuda_version':torch.version.cuda,'cudnn_benchmark':torch.backends.cudnn.benchmark}
    def write_status(status):
        atomic_json(report,{'status':status,'models':checks,'protocol_sha256':protocol_hash,
            'inference_adapter_sha256':sha(ROOT/'display_model_adapter.py'),
            'qualification_checker_sha256':sha(Path(__file__)),
            'support_sha256':sha(ROOT/'display_support.py'),'reader_sha256':sha(ROOT/'ir_sr.py'),
            'environment':environment,'roles':a.roles,'indices':a.indices,'consecutive':a.consecutive,
            'warmup':a.warmup,'repeats':a.repeats,'timing_scope':'prepared grayscale ROI through inference and CPU display resize; excludes read/decode, loading, warmup, saving and video encoding',
            'reference':'unchanged official FP32; this measures runtime equivalence, not restoration accuracy',
            'limits':ERROR_LIMITS,'temporal_error_limits':TEMPORAL_ERROR_LIMITS,
            'elapsed_seconds':time.perf_counter()-started,'updated_at':utc_now()})
    with run_lock(a.run/'gpu_inference.lock'):
        for base in cfg['models']:
            if base['name'] not in a.models:continue
            official={**base,'runtime_precision':'float32','attention_backend':'official'}
            model,meta=load_model(official);references=[]
            for key,item in manifest.items():
                group=item['group']
                for record in item['records']:
                    source=checked_read(ROOT/cfg['source_data'],record);index=int(record['frame_index'])
                    for scale in base['display_scales']:
                        low,_,geom=prepare(source,scale,roi=group['roi'][str(scale)])
                        benchmark=index in a.indices
                        if benchmark:
                            for _ in range(a.warmup):infer(model,official,low)
                        timings=[];process=[]
                        for _ in range(a.repeats if benchmark else 1):
                            tick=time.perf_counter();ref,_,timing=infer(model,official,low);display=resize(ref,(640,512))
                            process.append((time.perf_counter()-tick)*1000);timings.append(timing)
                        folder=Path(base['name'])/key/f'x{scale}'
                        save_image(images/folder/'float32_official'/f'frame_{index:05d}.png',display)
                        save_image(images/folder/'bicubic'/f'frame_{index:05d}.png',resize(low,(640,512)))
                        references.append({'sequence':key,'role':group['role'],'index':index,'scale':scale,'low':low,
                            'ref':ref,'display':display,'folder':folder,'geometry':geom,'source':record,
                            'benchmark':benchmark,'timings':timings,'process':process})
            del model;torch.cuda.empty_cache()
            for precision in a.precisions:
                for attention in a.attention_backends:
                    runtime={**base,'runtime_precision':precision,'attention_backend':attention}
                    model,runtime_meta=load_model(runtime);rows=[];previous={}
                    for case in references:
                        try:
                            if case['benchmark']:
                                for _ in range(a.warmup):infer(model,runtime,case['low'])
                            timings=[];process=[];native_errors=[];display_errors=[]
                            for _ in range(a.repeats if case['benchmark'] else 1):
                                tick=time.perf_counter();pred,_,timing=infer(model,runtime,case['low']);display=resize(pred,(640,512))
                                process.append((time.perf_counter()-tick)*1000);timings.append(timing)
                                native_errors.append(error_metrics(pred,case['ref']))
                                display_errors.append(error_metrics(display,case['display']))
                            native={k:max(e[k] for e in native_errors) for k in ERROR_LIMITS}
                            display_error={k:max(e[k] for e in display_errors) for k in ERROR_LIMITS}
                            delta=display-case['display'];previous_key=(case['sequence'],case['scale'])
                            prior=previous.get(previous_key);temporal=None
                            if prior is not None and prior[0]+1==case['index']:
                                # Change in runtime error, not motion-compensated scene flicker or restoration accuracy.
                                temporal=error_metrics(delta,prior[1])
                            previous[previous_key]=(case['index'],delta)
                            passed=within_error_limits(native) and within_error_limits(display_error)
                            if temporal is not None:passed=passed and within_error_limits(temporal,temporal=True)
                            row={'sequence':case['sequence'],'role':case['role'],'index':case['index'],'scale':case['scale'],
                                 'source':case['source'],'geometry':case['geometry'],'passed':passed,**native,
                                 'display_error':display_error,'repeat_native_errors':native_errors,'repeat_display_errors':display_errors,
                                 'runtime_error_change':temporal,'benchmark':case['benchmark'],
                                 'fp32_inference_ms':[t['inference_ms'] for t in case['timings']],
                                 'runtime_inference_ms':[t['inference_ms'] for t in timings],
                                 'fp32_process_ms':case['process'],'runtime_process_ms':process,
                                 'fp32_peak_cuda_allocated_mb':max(t['peak_cuda_allocated_mb'] for t in case['timings']),
                                 'runtime_peak_cuda_allocated_mb':max(t['peak_cuda_allocated_mb'] for t in timings)}
                            save_image(images/case['folder']/(precision+'_'+attention)/f"frame_{case['index']:05d}.png",display)
                        except (RuntimeError,ValueError) as error:
                            previous.pop((case['sequence'],case['scale']),None)
                            row={'sequence':case['sequence'],'role':case['role'],'index':case['index'],'scale':case['scale'],
                                 'passed':False,'error':str(error)}
                        rows.append(row)
                        print('Runtime',base['name'],precision,attention,case['sequence'],case['scale'],case['index'],row['passed'],flush=True)
                    checks.append({'model':base['name'],'precision':precision,'attention_backend':attention,
                        'weight_sha256':meta['weight_sha256'],'runtime_metadata':runtime_meta,
                        'passed':bool(rows) and all(r['passed'] for r in rows),'cases':rows,'source':'selected frozen groups; not independent operator acceptance'})
                    write_status('checking');del model;torch.cuda.empty_cache()
    write_status('completed')
    passed=all(m['passed'] for m in checks)
    print(json.dumps({'report':str(report),'passed':passed,'cases':sum(len(m['cases']) for m in checks)}),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
