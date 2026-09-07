"""CPU audit harness. Run from the repository root with the extension installed.

No arguments: run the initial sweep, emitting JSONL. Each setting uses a fresh
process, one warmup and five timed calls. Optional worker arguments:
    python benchmarks/cpu_latency.py tiny my-config 0
The last argument is 0 for the fixture, 1 for a synthetic 2400x3200 resize.
Worker mode honors OCR_*, REC_*, RAYON_* and AUDIT_THREADS environment variables.
Signatures compare ordered text and boxes, excluding floating-point confidence.
"""
import os,sys,json,time,statistics,subprocess,hashlib
configs=[('default',{}),('pool16',{'REC_POOL':'16'}),('budget320',{'REC_BUDGET':'320'}),('pool16-budget320',{'REC_POOL':'16','REC_BUDGET':'320'}),('det8-pool16-budget320',{'OCR_DET_THREADS':'8','REC_POOL':'16','REC_BUDGET':'320'}),('det4-pool16-budget320',{'OCR_DET_THREADS':'4','REC_POOL':'16','REC_BUDGET':'320'}),('det16-pool16-budget320',{'OCR_DET_THREADS':'16','REC_POOL':'16','REC_BUDGET':'320'}),('det8-pool8-threads8-budget320',{'OCR_DET_THREADS':'8','REC_POOL':'8','REC_BUDGET':'320','AUDIT_THREADS':'8'}),('det8-pool16-budget800',{'OCR_DET_THREADS':'8','REC_POOL':'16'}),('det8-pool16-budget1600',{'OCR_DET_THREADS':'8','REC_POOL':'16','REC_BUDGET':'1600'})]
if len(sys.argv)>1:
 import faster_paddle
 from PIL import Image
 import io
 size,name,large=sys.argv[1:]
 b=open('tests/fixtures/document.png','rb').read()
 if large=='1':
  im=Image.open(io.BytesIO(b)).convert('RGB').resize((2400,3200)); f=io.BytesIO(); im.save(f,format='PNG'); b=f.getvalue()
 t=time.perf_counter(); e=faster_paddle.OcrEngine(model_size=size,threads=int(os.environ['AUDIT_THREADS']) if 'AUDIT_THREADS' in os.environ else None); init=time.perf_counter()-t
 e.ocr(b); ts=[]
 for _ in range(5):
  t=time.perf_counter(); r=e.ocr(b); ts.append((time.perf_counter()-t)*1000)
 signature=[(v['topLeftCoord'],v['bottomRightCoord'],v['text']) for v in r['bounds'].values()]
 print(json.dumps(dict(size=size,config=name,large=large,init_ms=init*1000,median_ms=statistics.median(ts),min_ms=min(ts),max_ms=max(ts),times_ms=ts,boxes=len(r['bounds']),signature=hashlib.sha256(json.dumps(signature).encode()).hexdigest())))
else:
 for large in ['0','1']:
  for size in ['tiny','small']:
   for name,env in configs if large=='0' else [configs[0],configs[4],configs[7]]:
    e={k:v for k,v in os.environ.items() if not k.startswith(('OCR_','REC_','RAYON_','AUDIT_'))}; e.update(env)
    p=subprocess.run([sys.executable,__file__,size,name,large],env=e,text=True,capture_output=True,check=True)
    print(p.stdout.strip(),flush=True)
