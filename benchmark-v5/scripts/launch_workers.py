"""Run independent inference shards without distributed-training rendezvous."""
import os,signal,subprocess,sys,time
workers=[]
def stop(*args):
 for p in workers:
  if p.poll() is None:p.terminate()
 for p in workers:
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.kill();p.wait()
signal.signal(signal.SIGTERM,lambda *args:(stop(),sys.exit(143)))
try:
 for rank in range(4):
  env=dict(os.environ,RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='4')
  workers.append(subprocess.Popen([sys.executable,*sys.argv[1:]],env=env))
 while any(p.poll() is None for p in workers):
  if any(p.poll() not in (None,0) for p in workers):raise RuntimeError('Inference shard failed')
  time.sleep(.5)
 if any(p.returncode for p in workers):raise RuntimeError('Inference shard failed')
finally:stop()
