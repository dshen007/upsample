"""Freeze operator-display inputs and fetch only officially linked weights."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import requests

from runtime_v5 import atomic_json, sha, utc_now

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'third_party'/'display_deps'))
import gdown


def model_specs():
    sw='https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/'
    return [
        dict(name=f'swinir_classic_x{s}',family='swinir',scale=s,display_scales=[s],task='classical_sr',
             file=f'001_classicalSR_DF2K_s64w8_SwinIR-M_x{s}.pth',url=sw+f'001_classicalSR_DF2K_s64w8_SwinIR-M_x{s}.pth') for s in (2,3)
    ]+[
        dict(name='swinir_real_large_x4',family='swinir',scale=4,display_scales=[2,3],task='real_sr',
             file='003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth',url=sw+'003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth'),
        dict(name='hat_real_x4',family='hat',scale=4,display_scales=[2,3],task='real_sr',
             file='Real_HAT_GAN_SRx4.pth',drive_id='1Ma12vCWT27P9M99-s2RXnynKN-OQsBrv',config='HAT_GAN_Real_SRx4.yml')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);a=p.parse_args()
    run=a.run or ROOT/'outputs'/'display_zoom'/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    run.mkdir(parents=True,exist_ok=False)
    (ROOT/'outputs'/'display_zoom'/'active_run.txt').write_text(str(run.resolve()),encoding='utf-8')
    source_protocol=ROOT/'data'/'display_protocol_v1'
    audit=json.loads((source_protocol/'audit.json').read_text(encoding='utf-8'))
    specs=model_specs();folder=ROOT/'third_party'/'display_weights';folder.mkdir(exist_ok=True)
    for spec in specs:spec['weight']=str((folder/spec['file']).resolve())
    commits={name:subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'third_party'/name,text=True).strip()
             for name in ('HAT','SwinIR','BasicSR_display')}
    atomic_json(run/'protocol.json',{'objective':'operator_display','source_inputs':audit['inputs'],'groups':audit['groups'],
        'indices':[15,30,45,60,75,90,105,120,135,149],'display_size':[640,512],
        'scenarios':['crop','sampling'],'sampling':'INTER_AREA whole reference, same fixed 8-bit grayscale mapping',
        'models':specs,'source_commits':commits,'vins_gate':False,'training_started':False,
        'source_data':'data/rosbag_frames_v1','legacy_reference_gallery':str((ROOT/'outputs/操作员显示大模型结果/run_20261008_162935').resolve()),'future_frame_wait_ms':0,
        'latency_budget':'not yet specified; measure on current PC','created_at':utc_now()})
    lock=threading.Lock();states={s['name']:{'status':'pending'} for s in specs}
    def update(name,**values):
        with lock:
            states[name].update(values)
            atomic_json(run/'downloads.json',{'models':states,'updated_at':utc_now()})
    def fetch(spec):
        target=Path(spec['weight']);partial=target.with_suffix('.pth.part')
        update(spec['name'],status='downloading',url=spec.get('url') or 'https://drive.google.com/uc?id='+spec['drive_id'])
        try:
            if not target.exists():
                if spec.get('drive_id'):
                    last=[0.]
                    def progress(done,total):
                        if time.monotonic()-last[0]>2:
                            last[0]=time.monotonic();update(spec['name'],bytes=done,total=total)
                    gdown.download(id=spec['drive_id'],output=str(partial),quiet=True,progress=progress,timeout=(20,90),retries=1)
                else:
                    with requests.get(spec['url'],stream=True,timeout=(20,90)) as response:
                        response.raise_for_status();total=int(response.headers.get('Content-Length',0));done=0;last=0.
                        with partial.open('wb') as stream:
                            for block in response.iter_content(1024*1024):
                                if not block:continue
                                stream.write(block);done+=len(block)
                                if time.monotonic()-last>2:last=time.monotonic();update(spec['name'],bytes=done,total=total)
                        if total and done!=total:raise ValueError('Incomplete download')
                if partial.stat().st_size<1024*1024:raise ValueError('Weight download unexpectedly small')
                partial.rename(target)
            update(spec['name'],status='downloaded',bytes=target.stat().st_size,sha256=sha(target))
            print('Downloaded',spec['name'],target.stat().st_size,flush=True)
        except Exception as error:
            update(spec['name'],status='failed',error=str(error));print('FAILED',spec['name'],str(error),flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fetch,s) for s in specs]):future.result()
    failed=[n for n,v in states.items() if v['status']!='downloaded']
    atomic_json(run/'status.json',{'status':'weights_ready' if not failed else 'weights_partially_ready','failed':failed,'updated_at':utc_now()})
    print(str(run.resolve()),flush=True)


if __name__=='__main__':main()
