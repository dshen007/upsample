"""Verify delivered full images, video decoding and frozen pretrained provenance."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from runtime_v5 import atomic_json,sha,utc_now
from prepare_display_models import model_specs


def read_rows(path):
    with path.open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def require(ok,message):
    if not ok:raise ValueError(message)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--retained-only',action='store_true',help='Verify retained weights after the authorized cleanup; keep the original nine-model audit unchanged')
    a=p.parse_args();active={s['name'] for s in model_specs()}
    download=json.loads((a.run/'downloads.json').read_text(encoding='utf-8'))['models']
    cfg=json.loads((a.run/'protocol.json').read_text(encoding='utf-8'))
    for spec in cfg['models']:
        if a.retained_only and spec['name'] not in active:continue
        require(sha(spec['weight'])==download[spec['name']]['sha256'],'Stored weight changed '+spec['name'])
    for stage,count in (('development',840),('review',840),('development_denoise',360),('video',3600)):
        status=json.loads((a.run/stage/'status.json').read_text(encoding='utf-8'))
        require(status['status']=='completed' and status['rows']==count,'Incomplete '+stage)
        file=a.run/stage/('frame_metrics.csv' if stage=='video' else 'image_metrics.csv')
        require(len(read_rows(file))==count,'CSV count '+stage)
    for stage in ('development','review','development_denoise'):
        models=json.loads((a.run/stage/'models.json').read_text(encoding='utf-8'))
        for model in models:
            if a.retained_only and model['spec']['name'] not in active:continue
            name=model['spec']['name'];require(model['weight_sha256']==download[name]['sha256'],'Used weight changed '+name)
            require(not model['core_forward_modified'],'Changed official forward code')
            require(sha(model['architecture_file'])==model['architecture_sha256'],'Architecture changed '+name)
            if 'config_file' in model:
                require(sha(model['config_file'])==model['config_sha256'],'Official model config changed '+name)
    runtime=json.loads((a.run/'video_runtime.json').read_text(encoding='utf-8'))
    encoding=json.loads((a.run/'video_encoding.json').read_text(encoding='utf-8'))
    require(encoding['status']=='completed' and len(encoding['videos'])==18,'Preview encoding checks incomplete')
    for record in encoding['videos']:
        require(sha(record['file'])==record['video_sha256'],'Verified preview changed')
        require(record['quality']['frame_mae_dn_max']<=.75 and record['quality']['frame_psnr_db_min']>=40,
                'Preview compression quality exceeded limits')
    precision=json.loads((a.run/'precision_checks.json').read_text(encoding='utf-8'))
    require(precision['status']=='completed','Precision qualification incomplete')
    require(len(precision['models'])==2 and all(len(m['cases'])==6 for m in precision['models']),
            'Missing full-size precision cases')
    for name in runtime['models']:
        model=json.loads((a.run/'video'/(name+'_model.json')).read_text(encoding='utf-8'))
        require(model['weight_sha256']==download[name]['sha256'],'Video weight changed '+name)
        require(not model['core_forward_modified'],'Video forward code changed')
        require(model['spec']['runtime_precision']==runtime['precision'],'Video precision mismatch')
        if runtime['precision']=='float16_autocast':
            require(any(m['model']==name and m['passed'] and all(c['passed'] for c in m['cases'])
                        for m in precision['models']),'Unqualified video mixed precision')
    video_rows=read_rows(a.run/'video'/'frame_metrics.csv')
    for name in ('bicubic',*runtime['models']):
        for scale in ('2','3'):
            records=[r for r in video_rows if r['model']==name and r['scale']==scale]
            require(len({(r['sequence'],int(r['index'])) for r in records})==450,
                    'Missing continuous source frames '+name)
    checks=json.loads((a.run/'model_checks.json').read_text(encoding='utf-8'))
    require(checks['passed'] and len(checks['models'])==9,'Model verification incomplete')
    for model in checks['models']:
        require(model['load']['weight_sha256']==download[model['model']]['sha256'],'Check weight changed')
        require(all(x['official_caller_max_error']<=1e-5 for x in model['checks']),'Official caller discrepancy')
    images=list(a.out.rglob('*.png'));videos=list(a.out.rglob('*.mp4'))
    require(len(images)==48 and len(videos)==18,'Wrong delivery file counts')
    for file in images:
        image=cv2.imdecode(np.fromfile(file,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
        require(image is not None and image.shape==(512,640) and image.dtype==np.uint8,'Bad image '+str(file))
    for file in videos:
        cap=cv2.VideoCapture(str(file));count=0
        require(abs(cap.get(cv2.CAP_PROP_FPS)-25)<.01,'Bad preview FPS')
        while True:
            ok,frame=cap.read()
            if not ok:break
            require(frame.shape[:2]==(512,640),'Wrong video size');count+=1
        cap.release();require(count==150,'Truncated video '+str(file))
    manifest=json.loads((a.out/'manifest.json').read_text(encoding='utf-8'))
    require(len(manifest)==66,'Unexpected manifest count')
    for record in manifest:require(sha(a.out/record['file'])==record['sha256'],'Delivery hash changed')
    cli=a.run/'cli_replay_swinir_x3'/'超分结果.png'
    comparison=a.run/'review'/'images'/'crop'/'infrared_20260916_112100'/'clip_90pct'/'x3'/'swinir_real_large_x4'/'frame_00075.png'
    x=cv2.imdecode(np.fromfile(cli,dtype=np.uint8),0);y=cv2.imread(str(comparison),0)
    require(x is not None and np.array_equal(x,y),'Independent CLI differs from full-image review')
    result={'passed':True,'objective':'operator_display','official_model_weights':4 if a.retained_only else 9,'historical_official_caller_checks':18,
        'full_size_gray_pngs':48,'preview_videos':18,'preview_frames_each':150,'preview_encoding_checks':18,'cli_replay_exact':True,
        'still_screen_records':1680,'denoise_ablation_records':360,'continuous_records':3600,
        'vins_gate_used':False,'training_started':False,'created_at':utc_now()}
    if a.retained_only:
        atomic_json(a.run/'post_cleanup_delivery_verification.json',result)
    else:
        atomic_json(a.out/'交付验证.json',result);atomic_json(a.run/'delivery_verification.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
