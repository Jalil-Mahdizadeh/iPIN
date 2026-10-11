"""Fixed small heads and TRAIN-only OOF fusion; no VAL labels select any new parameter."""
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.exceptions import ConvergenceWarning
from study import config, evidence, fuse


def fit(x, y, indices, pca=None, seed=None):
    cfg = config()['fitting']
    if len(np.unique(y[indices])) != 2 or not np.isfinite(x).all():
        raise ValueError('Invalid head inputs')
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        sc = StandardScaler().fit(x[indices]); z = sc.transform(x)
        state = {'standard_mean':sc.mean_.tolist(), 'standard_scale':sc.scale_.tolist()}
        if pca is not None:
            pc = PCA(n_components=pca,svd_solver='full').fit(z[indices]); z = pc.transform(z)
            state.update({'pca_mean':pc.mean_.tolist(),'pca_components':pc.components_.tolist()})
        cl = LogisticRegression(C=cfg['C'],solver='lbfgs',max_iter=cfg['max_iter'],
            random_state=config()['seed'] if seed is None else seed).fit(z[indices],y[indices])
    state.update({'coef':cl.coef_.tolist(),'intercept':cl.intercept_.tolist(),'classes':cl.classes_.tolist(),
        'fit_indices':indices.tolist(),'fit_class_counts':np.bincount(y[indices],minlength=2).tolist()})
    result = cl.decision_function(z)
    np.testing.assert_allclose(evidence(state,x),result,atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(sc.mean_,x[indices].mean(0),atol=1e-12,rtol=1e-12)
    return state,result


def crossfit(x, y, train, folds, pca=None, seed=None):
    train = np.asarray(train,dtype=bool); folds = np.asarray(folds)
    if not np.array_equal(train,folds>=0) or set(folds[train]) != set(range(config()['fitting']['folds'])):
        raise ValueError('Invalid TRAIN-only OOF assignment')
    oof = np.full(len(y),np.nan); states = []
    for fold in range(config()['fitting']['folds']):
        indices = np.flatnonzero(train & (folds != fold)); held = train & (folds == fold)
        h,z = fit(x,y,indices,pca,seed); oof[held] = z[held]
        states.append({'fold':fold,'held_indices':np.flatnonzero(held).tolist(),'head':h})
    if not np.isfinite(oof[train]).all() or np.isfinite(oof[~train]).any():
        raise ValueError('OOF coverage/cohort mismatch')
    h,z = fit(x,y,np.flatnonzero(train),pca,seed)
    return {'full':h,'folds':states},z,oof


def choose_alpha(base, oof, gate, y, train):
    # Slice before scoring: VAL labels and scores cannot enter this operation.
    b,e,g,yy = [np.asarray(v)[train] for v in (base,oof,gate,y)]
    if not np.isfinite(e).all() or not np.isfinite(b).all() or not (g>0).all():
        raise ValueError('Fusion needs eligible TRAIN OOF evidence')
    choices = [{'alpha':a,'ap':float(average_precision_score(yy,fuse(b,e,g,a)))}
        for a in config()['fitting']['fusion_alpha_grid']]
    best = max(c['ap'] for c in choices)
    alpha = min(c['alpha'] for c in choices if c['ap']==best)
    return {'alpha':alpha,'eligible_train_oof_grid':choices,'calibration_indices':np.flatnonzero(train).tolist()}


def quality_features(source, sample, metadata, diagnostics):
    # Original 68 profile/quality values plus eight declared symmetric nuisance statistics.
    extra = np.zeros((len(sample),8))
    for i,m in enumerate(metadata):
        if not m['available']:
            continue
        d = diagnostics[i]['diagnostics']; lengths = sorted([m['msa']['length_a'],m['msa']['length_b']])
        extra[i] = [np.log1p(lengths[0]),np.log1p(lengths[1]),np.log1p(d['retained_depth']),
            np.log1p(d['paired_diversity']['neff80']),np.log1p(d['valid_interchain_cells']),
            d['minimum_quality'],d['true_support']['fraction_cells_at_least_half_rows'],
            np.log(d['length_asymmetry'])]
    return np.column_stack([source,extra])
