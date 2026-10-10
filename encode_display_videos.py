"""Encode and verify display previews against the authoritative full-frame PNGs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from runtime_v5 import atomic_json,sha,utc_now

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'third_party/display_deps'))
import imageio_ffmpeg


def quality(file,folder):
    cap=cv2.VideoCapture(str(file));rows=[]
    if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS)-25)>.01:raise ValueError('Bad encoded preview')
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            index=len(rows);reference=cv2.imread(str(folder/f'frame_{index:05d}.png'),0)
            if reference is None or reference.shape!=(512,640):raise ValueError('Missing reference PNG')
            output=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            if output.shape!=reference.shape:raise ValueError('Encoded size changed')
            difference=output.astype(np.float32)-reference
            mse=float(np.mean(difference**2))
            rows.append({'index':index,'mae_dn':float(np.abs(difference).mean()),
                         'psnr_db':float(10*np.log10(255**2/max(mse,1e-12)))})
    finally:cap.release()
    if len(rows)!=150:raise ValueError('Encoded frame count changed')
    metrics={'frames':len(rows),'frame_mae_dn_mean':float(np.mean([x['mae_dn'] for x in rows])),
             'frame_mae_dn_max':max(x['mae_dn'] for x in rows),'frame_psnr_db_min':min(x['psnr_db'] for x in rows),
             'frame_psnr_db_median':float(np.median([x['psnr_db'] for x in rows]))}
    if metrics['frame_mae_dn_max']>.75 or metrics['frame_psnr_db_min']<40:
        raise ValueError('Preview compression exceeded quality limits: '+str(metrics))
    return metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--models',nargs='+');a=p.parse_args()
    cfg=json.loads((a.run/'protocol.json').read_text(encoding='utf-8'))
    selected=json.loads((a.run/'display_candidates.json').read_text(encoding='utf-8'))['models']
    names=a.models if a.models else ['bicubic',*selected]
    if any(n not in ('bicubic',*selected) for n in names):raise ValueError('Unknown preview model')
    if a.models is None and json.loads((a.run/'video/status.json').read_text(encoding='utf-8'))['status']!='completed':
        raise ValueError('Wait for continuous comparison before encoding all outputs')
    executable=imageio_ffmpeg.get_ffmpeg_exe()
    version=subprocess.check_output([executable,'-version'],text=True).splitlines()[0]
    binary_sha=sha(executable);records=[]
    for key,group in cfg['groups'].items():
        if group['role']!='review':continue
        for scale in (2,3):
            for name in names:
                folder=a.run/'video/images'/key/f'x{scale}'/name
                files=[folder/f'frame_{i:05d}.png' for i in range(150)]
                if not all(x.exists() for x in files):raise ValueError('Incomplete PNG sequence')
                hashes=[sha(x) for x in files]
                source_hash=hashlib.sha256('\n'.join(hashes).encode()).hexdigest()
                file=a.run/'video/hq_videos'/key/f'x{scale}'/(name+'.mp4');metadata=file.with_suffix('.json')
                file.parent.mkdir(parents=True,exist_ok=True)
                if file.exists():
                    previous=json.loads(metadata.read_text(encoding='utf-8'))
                    if previous['source_png_sequence_sha256']!=source_hash or previous['video_sha256']!=sha(file):
                        raise ValueError('Refusing to replace a changed preview')
                    records.append(previous);continue
                attempts=[]
                for crf in (6,2):
                    temporary=file.with_name(name+f'.crf{crf}.pending.mp4')
                    command=[executable,'-hide_banner','-loglevel','error','-n','-framerate','25','-start_number','0',
                             '-i',str(folder/'frame_%05d.png'),'-frames:v','150','-an',
                             '-vf','scale=in_range=full:out_range=full','-c:v','libx264','-threads','1',
                             '-preset','fast','-crf',str(crf),'-pix_fmt','yuv420p','-color_range','pc',
                             '-movflags','+faststart',str(temporary)]
                    if not temporary.exists():subprocess.run(command,check=True,timeout=120)
                    try:metrics=quality(temporary,folder)
                    except ValueError as error:
                        attempts.append({'crf':crf,'passed':False,'reason':str(error)});continue
                    attempts.append({'crf':crf,'passed':True});temporary.replace(file);break
                else:raise ValueError('Both preview quality settings failed: '+str(attempts))
                record={'file':str(file.resolve()),'source_png_sequence_sha256':source_hash,'video_sha256':sha(file),
                        'codec':'H.264 libx264','crf':crf,'pixel_format':'yuv420p','color_range':'full','attempts':attempts,
                        'display_fps':25,'encoding_threads':1,'ffmpeg_version':version,'ffmpeg_binary_sha256':binary_sha,
                        'wrapper_version':imageio_ffmpeg.__version__,'png_frame_sha256':hashes,'quality':metrics,'created_at':utc_now()}
                atomic_json(metadata,record);records.append(record)
                print('Verified high-quality preview',key,scale,name,metrics,flush=True)
    all_records=[json.loads(x.read_text(encoding='utf-8')) for x in (a.run/'video/hq_videos').rglob('*.json')]
    complete=len(all_records)==18
    atomic_json(a.run/'video_encoding.json',{'status':'completed' if complete else 'partial','videos':all_records,
        'limits':{'frame_mae_dn_max':.75,'frame_psnr_db_min':40},'reference':'authoritative model PNG, not SR ground truth',
        'created_at':utc_now()})


if __name__=='__main__':main()
