"""Fixed metric definitions, shared between deposited and fresh predictions."""
import numpy as np
from sklearn.metrics import (average_precision_score,precision_recall_curve,
    roc_auc_score,auc,precision_score,recall_score,f1_score,matthews_corrcoef,confusion_matrix)

def metrics(labels,scores,threshold=0.5):
    y=np.asarray(labels,dtype=int);s=np.asarray(scores,dtype=np.float64)
    assert len(y)==len(s) and len(y)>0 and np.isfinite(s).all()
    assert set(y)=={0,1}
    p,r,_=precision_recall_curve(y,s)
    pred=s>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(n=len(y),positive=int(y.sum()),prevalence=float(y.mean()),
                average_precision=float(average_precision_score(y,s)),pr_auc_trapezoid=float(auc(r,p)),
                auroc=float(roc_auc_score(y,s)),threshold=threshold,
                precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred)),
                f1=float(f1_score(y,pred)),mcc=float(matthews_corrcoef(y,pred)),
                tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),unique_scores=len(np.unique(s)),
                score_min=float(s.min()),score_max=float(s.max()))
