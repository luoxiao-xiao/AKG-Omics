"""Same PCC/RMSE/CMD/graph-SSIM definitions, with scalable evaluation.
CMD: float32 GPU correlation matrices and Frobenius inner product.
SSIM: original function evaluated in feature blocks with the SAME global data range.
"""
import inspect,types,random,numpy as np,torch,scipy.sparse as sp
VERSION='v2_gpu_CMD_feature_block_SSIM_global_range'
def cmd(x,y):
 # Same predeclared noise scale as original; metric RNG is independent of training.
 rng=np.random.RandomState(1729)
 x=x+rng.normal(0,1e-8,x.shape);y=y+rng.normal(0,1e-8,y.shape)
 def correlation(a):
  z=torch.as_tensor(a,dtype=torch.float32,device=('cuda' if torch.cuda.is_available() else 'cpu'));z.sub_(z.mean(0));c=z.T@z;del z
  std=torch.diagonal(c).clamp_min(0).sqrt();c.div_(std[:,None]);c.div_(std[None,:]);c.clamp_(-1,1);return c
 with torch.no_grad():
  cx=correlation(x);del x;cy=correlation(y);del y
  # tr(Cx Cy) = sum_ij Cx_ij Cy_ji; avoids a cubic matrix multiplication.
  numerator=(cx.double()*cy.T.double()).sum();den=cx.double().square().sum().sqrt()*cy.double().square().sum().sqrt();v=float(1-numerator/(den+1e-8))
 return v

def ssim(x,y,g,u,block=64):
 src=inspect.getsource(u.structural_similarity_on_graph_data)
 assert 'R = x.max() - x.min()' in src
 src=src.replace('R = x.max() - x.min()','R = GLOBAL_DATA_RANGE')
 ns={**u.__dict__,'GLOBAL_DATA_RANGE':x.max()-x.min()};exec(compile(src,'frozen_ssim_global_range','exec'),ns);fn=ns['structural_similarity_on_graph_data']
 values=[]
 for start in range(0,x.shape[1],block):values.append(fn(x[:,start:start+block],y[:,start:start+block],g))
 return float(np.concatenate(values).mean())

def evaluate(truth,pred,g,u):
 np.random.seed(1729);random.seed(1729);m={}
 for k in ['pcc','rmse']:
  _,v=u.Compute_metrics(truth.copy(),pred.copy(),metric=k);m[k.upper()]=float(v)
 print('METRIC_CMD_START',flush=True);m['CMD']=cmd(truth,pred);print('METRIC_CMD_DONE',m['CMD'],flush=True)
 ga=g.astype(np.float64);ga=(sp.diags(1/np.asarray(ga.sum(1)).ravel())@ga).tocsr();assert np.max(np.abs(np.asarray(ga.sum(1)).ravel()-1))<1e-10
 print('METRIC_SSIM_START',flush=True);m['SSIM_row_normalized']=ssim(truth,pred,ga,u);print('METRIC_SSIM_DONE',m['SSIM_row_normalized'],flush=True)
 assert all(np.isfinite(v) for v in m.values());return m
