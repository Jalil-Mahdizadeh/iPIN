from bench_utils import EXTERNAL,read,sha
ROOT=EXTERNAL/'experiments/x_pair_test2_v1'
def verify(path,item):
 assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],str(path)
