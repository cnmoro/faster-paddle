"""Recognizer-only synthetic width probe; run from the repository root.

Requires numpy and onnxruntime. Prints JSONL; profiler traces go to the system
temporary directory. These timings do not measure OCR accuracy or full latency.
"""
import onnxruntime as ort,numpy as np,time,statistics,json,tempfile,os
from collections import defaultdict
for size in ['tiny','small']:
 opt=ort.SessionOptions(); opt.intra_op_num_threads=1; opt.enable_mem_pattern=False
 s=ort.InferenceSession(f'models/{size}/rec.onnx',opt,providers=['CPUExecutionProvider'])
 for w in [64,128,192,320,640]:
  x=np.random.default_rng(0).uniform(-1,1,(1,3,48,w)).astype('float32')
  try:
   for _ in range(3): y=s.run(None,{'x':x})[0]
   ts=[]
   for _ in range(10):
    t=time.perf_counter(); y=s.run(None,{'x':x})[0]; ts.append((time.perf_counter()-t)*1000)
   print(json.dumps(dict(size=size,width=w,shape=y.shape,median_ms=statistics.median(ts))),flush=True)
  except Exception as e: print(json.dumps(dict(size=size,width=w,error=str(e))),flush=True)
 del s
 opt.enable_profiling=True; opt.profile_file_prefix=os.path.join(tempfile.gettempdir(),'faster-paddle-audit-'+size)
 s=ort.InferenceSession(f'models/{size}/rec.onnx',opt,providers=['CPUExecutionProvider'])
 x=np.random.default_rng(0).uniform(-1,1,(1,3,48,320)).astype('float32')
 for _ in range(5):s.run(None,{'x':x})
 p=s.end_profiling(); events=json.load(open(p)); agg=defaultdict(float); nodes=defaultdict(float)
 for e in events:
  if e.get('cat')=='Node' and e['name'].endswith('_kernel_time'):
   agg[e['args'].get('op_name','?')]+=e['dur'];nodes[e['name']]+=e['dur']
 print(json.dumps(dict(size=size,profile_ops_us=sorted(agg.items(),key=lambda x:-x[1]),profile_top_nodes_us=sorted(nodes.items(),key=lambda x:-x[1])[:8])),flush=True)
 del s
