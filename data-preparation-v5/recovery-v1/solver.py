"""Equivalent Bernett degree/GO MILP with an explicit, durable feasible start.

For each protein: Mx - d_above + d_below = d_positive. Bounds on the
deviations imply the original six-times degree cap. GO uses an unnormalised
sum equation to avoid tiny matrix entries. The cost is the original
normalised objective multiplied by SCALE; minimisers are unchanged.
"""
import hashlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import highspy as hp
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from common import read_json, write_json, sha
from sample_ilp import author

SCALE = 1_000_000.0


class Problem:
    def __init__(self, positives, candidates, proteins, go):
        self.pos = np.asarray(positives, dtype=np.int64)
        self.candidates = np.asarray(candidates, dtype=np.int64)
        self.proteins = proteins
        self.n = len(proteins)
        self.m = len(candidates)
        self.k = len(positives)
        self.plus = np.bincount(self.pos.ravel(), minlength=self.n)
        assert (self.plus > 0).all()
        assert (self.candidates[:, 0] < self.candidates[:, 1]).all()
        assert len(np.unique(self.candidates, axis=0)) == self.m
        self.cap = 6 * self.plus
        ctx = author.build_context(self.pos, {p:i for i,p in enumerate(proteins)},
                                   proteins, self.candidates, 1., go_bp=go)
        degree = author.DegreeBias(1.)
        degree.precompute(ctx)
        jaccard = author.JaccardMeanBias(1.)
        jaccard.precompute(ctx)
        assert degree.is_active() and jaccard.is_active()
        assert np.array_equal(degree.active_idx, np.arange(self.n))
        self.degree_cost = degree.coef / degree.U
        self.jac = jaccard.J_cand
        self.go_target = jaccard.j_bar_pos
        self.go_scale = jaccard.U
        self.membership, self.go_sizes = author._build_go_membership(go)
        self.go = go
        self.incidence = ctx.incidence

    def check(self, selected):
        selected = np.asarray(selected, dtype=np.int64)
        assert selected.ndim == 1 and len(selected) == self.k
        assert len(np.unique(selected)) == self.k
        assert selected.min() >= 0 and selected.max() < self.m
        degree = np.bincount(self.candidates[selected].ravel(), minlength=self.n)
        assert np.all(degree <= self.cap)
        return degree

    def metrics(self, selected):
        degree = self.check(selected)
        mean = float(self.jac[selected].mean())
        degree_term = float(self.degree_cost @ np.abs(degree - self.plus))
        go_term = abs(mean - self.go_target) / self.go_scale
        return dict(objective=degree_term + go_term,
                    objective_terms={'degree': degree_term, 'jaccard': go_term},
                    positive_go_jaccard_mean=self.go_target,
                    negative_go_jaccard_mean=mean,
                    degree_absolute_residual_sum=int(np.abs(degree-self.plus).sum()),
                    max_negative_positive_degree_ratio=float((degree/self.plus).max()))

    def feasible_start(self):
        degree = np.zeros(self.n, dtype=np.int64)
        chosen = []
        for i in np.random.default_rng(2).permutation(self.m):
            a,b = self.candidates[i]
            if degree[a] < self.cap[a] and degree[b] < self.cap[b]:
                chosen.append(int(i)); degree[a] += 1; degree[b] += 1
                if len(chosen) == self.k:
                    break
        selected = np.sort(np.asarray(chosen, dtype=np.int64))
        self.check(selected)
        return selected

    def vector(self, selected):
        degree = self.check(selected)
        x = np.zeros(self.m)
        x[selected] = 1
        residual = degree - self.plus
        go_residual = float(self.jac[selected].sum()) - self.k*self.go_target
        return np.concatenate([x, np.maximum(residual,0), np.maximum(-residual,0),
                               [max(go_residual,0), max(-go_residual,0)]])

    def model(self):
        # Mx - above + below = positive degrees; sum(x)=k; Jx - above + below=k*mean.
        eye = sp.eye(self.n, format='csc')
        degree = sp.hstack([self.incidence, -eye, eye, sp.csc_matrix((self.n,2))], format='csc')
        count = sp.csc_matrix((np.ones(self.m), (np.zeros(self.m,dtype=np.int32), np.arange(self.m))),
                             shape=(1,self.m+2*self.n+2))
        go = sp.csc_matrix(np.concatenate([self.jac,np.zeros(2*self.n),[-1.,1.]])[None,:])
        matrix = sp.vstack([degree,count,go],format='csc')
        rhs = np.concatenate([self.plus,[self.k,self.k*self.go_target]]).astype(float)
        costs = SCALE*np.concatenate([np.zeros(self.m),self.degree_cost,self.degree_cost,
                                      [1/(self.k*self.go_scale)]*2])
        upper = np.concatenate([np.ones(self.m),5*self.plus,self.plus,
                                [self.k*(1-self.go_target), self.k*self.go_target]])
        model = hp.HighsModel(); lp = model.lp_
        lp.num_col_,lp.num_row_ = matrix.shape[1],matrix.shape[0]
        lp.col_cost_ = costs
        lp.col_lower_ = np.zeros(len(costs)); lp.col_upper_ = upper
        lp.row_lower_ = rhs; lp.row_upper_ = rhs
        lp.integrality_ = [hp.HighsVarType.kInteger]*self.m + [hp.HighsVarType.kContinuous]*(2*self.n+2)
        lp.a_matrix_.format_ = hp.MatrixFormat.kColwise
        lp.a_matrix_.start_ = matrix.indptr
        lp.a_matrix_.index_ = matrix.indices
        lp.a_matrix_.value_ = matrix.data
        return model,matrix,rhs,costs,upper


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+'.part')
    with temp.open('wb') as f:
        np.savez_compressed(f,**arrays); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


class Checkpoint:
    def __init__(self, problem, directory, identity):
        self.problem = problem; self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.identity = identity
        self.updated_monotonic = time.monotonic()
        self.manifest = self.directory/'best.json'
        self.selected = None; self.objective = math.inf
        if self.manifest.exists():
            previous = read_json(self.manifest)
            assert previous['identity'] == identity, 'Checkpoint inputs changed'
            path = self.directory/previous['selection_file']
            assert sha(path) == previous['selection_sha256']
            self.selected = np.load(path)['selected']
            self.objective = problem.metrics(self.selected)['objective']
            assert abs(self.objective-previous['metrics']['objective']) < 1e-12

    def update(self, selected, origin, force=False):
        selected = np.sort(np.asarray(selected,dtype=np.int64))
        metrics = self.problem.metrics(selected)
        if not force and metrics['objective'] >= self.objective-1e-13:
            return False
        digest = hashlib.sha256(selected.tobytes()).hexdigest()
        filename = f'selection-{digest[:20]}.npz'
        path = self.directory/filename
        if not path.exists():
            atomic_npz(path,selected=selected)
        # Immutable selection first, atomic manifest commit second.
        write_json(self.manifest,dict(identity=self.identity,origin=origin,metrics=metrics,
                                     selection_file=filename,selection_sha256=sha(path),
                                     saved_unix=time.time()))
        self.selected = selected; self.objective = metrics['objective']
        self.updated_monotonic = time.monotonic()
        print(json.dumps(dict(event='feasible_checkpoint',origin=origin,**metrics)),flush=True)
        return True


def optimize(problem, checkpoint, seconds, log_file, threads=8, gap=.01, interrupt_after=None, stall_seconds=None):
    if checkpoint.selected is None:
        checkpoint.update(problem.feasible_start(),'verified_feasible_initialization',force=True)
    initial = checkpoint.objective
    model,matrix,rhs,costs,upper = problem.model()
    vector = problem.vector(checkpoint.selected)
    assert np.max(np.abs(matrix@vector-rhs)) < 1e-7
    assert np.all(vector >= -1e-9) and np.all(vector <= upper+1e-7)
    assert abs(costs@vector/SCALE-initial) < 1e-10
    h = hp.Highs()
    h.HandleUserInterrupt = True
    options = dict(threads=threads,parallel='on',random_seed=2,time_limit=float(seconds),
                   mip_rel_gap=gap,mip_abs_gap=1e-4,mip_lp_solver='ipm',mip_ipm_solver='ipx',
                   mip_max_start_nodes=0,mip_min_logging_interval=5.,log_file=str(log_file),
                   log_to_console=True,presolve='on')
    for name,value in options.items():
        assert h.setOptionValue(name,value) == hp.HighsStatus.kOk,(name,value)
    assert h.passModel(model) == hp.HighsStatus.kOk
    solution = hp.HighsSolution(); solution.col_value = vector; solution.value_valid = True
    assert h.setSolution(solution) == hp.HighsStatus.kOk
    start = time.monotonic(); last_heartbeat = [0.]; acceptance = [False]
    stop_reason = [None]
    callback_error = []
    def incumbent(event):
        try:
            values = np.asarray(event.data_out.mip_solution)[:problem.m]
            assert len(values) == problem.m and np.max(np.abs(values-np.round(values))) < 1e-5
            selected = np.flatnonzero(np.round(values)==1)
            problem.check(selected)
            acceptance[0] = True
            checkpoint.update(selected,'highs_integer_incumbent')
        except Exception as exc:
            callback_error.append(str(exc)); event.interrupt()
    def heartbeat(event):
        elapsed = time.monotonic()-start
        if elapsed-last_heartbeat[0] >= 15 or not last_heartbeat[0]:
            bound = float(event.data_out.mip_dual_bound)/SCALE
            write_json(checkpoint.directory/'heartbeat.json',dict(elapsed_seconds=elapsed,
                best_verified_objective=checkpoint.objective,solver_observed_feasible_incumbent=acceptance[0],
                bound=bound if math.isfinite(bound) else None,
                simplex_iterations=int(event.data_out.simplex_iteration_count),
                ipm_iterations=int(event.data_out.ipm_iteration_count),updated_unix=time.time()))
            last_heartbeat[0] = elapsed
        if elapsed >= seconds or (interrupt_after is not None and elapsed>=interrupt_after):
            stop_reason[0] = 'time_limit' if elapsed>=seconds else 'qualification_interrupt'
            event.interrupt()
        elif stall_seconds is not None and elapsed>=stall_seconds and time.monotonic()-checkpoint.updated_monotonic>=stall_seconds:
            stop_reason[0] = 'no_incumbent_improvement_within_stall_budget'
            event.interrupt()
    h.cbMipSolution += incumbent
    h.cbMipImprovingSolution += incumbent
    h.cbMipInterrupt += heartbeat
    h.cbIpmInterrupt += heartbeat
    h.cbSimplexInterrupt += heartbeat
    print(json.dumps(dict(event='warm_start_submitted',objective=initial,variables=len(vector),
                          rows=matrix.shape[0],nonzeros=matrix.nnz,options=options)),flush=True)
    def interrupted(signum,frame):
        stop_reason[0] = f'external_signal_{signum}'
        h.cancelSolve()
    previous_handlers = {s:signal.signal(s,interrupted) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        status = h.run()
    finally:
        for s,handler in previous_handlers.items():signal.signal(s,handler)
    if callback_error:
        raise RuntimeError(f'Invalid solver callback: {callback_error}')
    info = h.getInfo(); solution = h.getSolution()
    feasible = (solution.value_valid and info.primal_solution_status==hp.SolutionStatus.kSolutionStatusFeasible)
    if feasible:
        values = np.asarray(solution.col_value[:problem.m])
        assert np.max(np.abs(values-np.round(values))) < 1e-5
        checkpoint.update(np.flatnonzero(np.round(values)==1),'highs_returned_integer_solution')
        acceptance[0] = True
    # A timeout cannot replace a checked incumbent with an empty/invalid solver vector.
    problem.check(checkpoint.selected)
    dual = float(info.mip_dual_bound)/SCALE
    lower = max(0.,dual) if math.isfinite(dual) else 0.
    assert lower <= checkpoint.objective+1e-7,(lower,checkpoint.objective)
    actual_gap = max(0.,checkpoint.objective-lower)/abs(checkpoint.objective) if checkpoint.objective else 0.
    report = dict(status=h.getModelStatus().name,highs_run_status=str(status),
        returned_solution_feasible=bool(feasible),solver_observed_feasible_incumbent=acceptance[0],
        initial_objective=initial,objective=checkpoint.objective,dual_bound=lower,actual_mip_gap=actual_gap,
        solver_reported_gap=float(info.mip_gap) if math.isfinite(info.mip_gap) else None,
        nodes=int(info.mip_node_count),solver_seconds=time.monotonic()-start,
        configured_time_limit=seconds,configured_mip_gap=gap,objective_scale=SCALE,
        stop_reason=stop_reason[0],stall_limit_seconds=stall_seconds,
        mathematically_proven_optimal=bool(actual_gap<1e-9),
        formulation='equivalent equality/deviation form of pinned author degree and mean GO objectives',
        options=options)
    write_json(checkpoint.directory/'solver-result.json',report)
    return report
