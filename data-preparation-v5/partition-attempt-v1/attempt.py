"""One bounded, warm-started attempt at the unchanged positive partition problem."""
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import highspy as hp
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from common import read_json, write_json, sha

POLICY = dict(maximum_production_attempts=1, solver_seconds=1800, threads=8,
              target_relative_gap=.01, minimum_retention_increase_fraction=.01,
              training_positives_must_not_decrease=True,
              stop_after_seconds_without_improvement_once_adoptable=300,
              original_and_ilp_tests_unchanged=True, group_assignment_inputs_unchanged=True)


def utc():
    return datetime.now(timezone.utc).isoformat()


class Partition:
    def __init__(self, intra, cross):
        self.intra = np.asarray(intra, dtype=np.int64)
        self.cross = np.asarray(cross, dtype=np.int64)
        self.n = len(intra)
        self.edges = np.transpose(np.nonzero(np.triu(self.cross, 1)))
        self.weights = self.cross[self.edges[:, 0], self.edges[:, 1]]
        self.m = len(self.edges)
        self.degree = np.bincount(self.edges.ravel(), weights=np.repeat(self.weights, 2),
                                  minlength=self.n).astype(np.int64)
        self.total = int(self.intra.sum() + self.weights.sum())
        assert np.all(self.intra >= 0) and np.all(self.weights > 0)

    def counts(self, labels):
        """Independent integer counts; 1 means TRAIN, 0 means DEV."""
        y = np.asarray(labels)
        assert y.shape == (self.n,) and np.all((y == 0) | (y == 1))
        a, b = self.edges.T
        train = int(self.intra[y == 1].sum() + self.weights[(y[a] == 1) & (y[b] == 1)].sum())
        val = int(self.intra[y == 0].sum() + self.weights[(y[a] == 0) & (y[b] == 0)].sum())
        retained = train + val
        return dict(train=train, val=val, retained=retained, discarded=self.total-retained,
                    train_fraction=train/retained if retained else None,
                    val_fraction=val/retained if retained else None,
                    feasible=bool(retained > 0 and 25*train >= 19*retained and 100*val >= 19*retained))

    def vector(self, labels):
        y = np.asarray(labels, dtype=float)
        return np.concatenate([y, y[self.edges[:, 0]] * y[self.edges[:, 1]]])

    def model(self):
        # y_i assigns a whole unchanged group to TRAIN. z_ij=y_i*y_j.
        # R=N-degree@y+2*w@z; T=intra@y+w@z; V=R-T.
        # DEV's product is exactly 1-y_i-y_j+z_ij: no duplicated product variables.
        rows, cols, values = [], [], []
        lower, upper = [], []
        def add(indices, coefficients, lo, hi):
            r = len(lower)
            rows.extend([r]*len(indices)); cols.extend(indices); values.extend(coefficients)
            lower.append(lo); upper.append(hi)
        for e, (i, j) in enumerate(self.edges):
            z = self.n+e
            add([z, i], [1., -1.], -hp.kHighsInf, 0.)
            add([z, j], [1., -1.], -hp.kHighsInf, 0.)
            add([z, i, j], [1., -1., -1.], -1., hp.kHighsInf)
        # Same 76% TRAIN / 19% DEV lower fractions of retained positives.
        train_coeff = np.concatenate([self.intra+.76*self.degree, -.52*self.weights])
        val_coeff = np.concatenate([-self.intra-.81*self.degree, .62*self.weights])
        all_columns = list(range(self.n+self.m))
        add(all_columns, train_coeff.tolist(), .76*self.total, hp.kHighsInf)
        add(all_columns, val_coeff.tolist(), -.81*self.total, hp.kHighsInf)
        matrix = sp.csc_matrix((values, (rows, cols)), shape=(len(lower), self.n+self.m))
        costs = np.concatenate([self.degree, -2*self.weights]).astype(float)
        model = hp.HighsModel(); lp = model.lp_
        lp.num_col_ = self.n+self.m; lp.num_row_ = len(lower)
        lp.col_cost_ = costs; lp.offset_ = -float(self.total)
        lp.col_lower_ = np.zeros(self.n+self.m); lp.col_upper_ = np.ones(self.n+self.m)
        lp.row_lower_ = np.asarray(lower); lp.row_upper_ = np.asarray(upper)
        lp.integrality_ = [hp.HighsVarType.kInteger]*self.n + [hp.HighsVarType.kContinuous]*self.m
        lp.a_matrix_.format_ = hp.MatrixFormat.kColwise
        lp.a_matrix_.start_ = matrix.indptr; lp.a_matrix_.index_ = matrix.indices
        lp.a_matrix_.value_ = matrix.data
        return model, matrix, np.asarray(lower), np.asarray(upper), costs


def qualification():
    """Enumerate tiny assignments and compare raw counts with the new formulation."""
    intra = np.array([4, 6, 8, 2, 7, 5])
    cross = np.triu(np.array([[0, 3, 0, 4, 2, 1], [0, 0, 2, 0, 1, 5],
                              [0, 0, 0, 5, 0, 1], [0, 0, 0, 0, 2, 3],
                              [0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 0]]), 1)
    p = Partition(intra, cross)
    model, matrix, lower, upper, cost = p.model()
    feasible = []
    for bits in itertools.product((0, 1), repeat=p.n):
        y = np.array(bits); vector = p.vector(y); counts = p.counts(y)
        actual = matrix@vector
        encoded_feasible = bool(np.all(actual >= lower-1e-8) and np.all(actual <= upper+1e-8))
        assert encoded_feasible == counts['feasible']
        assert abs(cost@vector-p.total+counts['retained']) < 1e-8
        if counts['feasible']: feasible.append((counts['retained'], y))
    optimum = max(v[0] for v in feasible)
    h = hp.Highs()
    for name, value in dict(threads=8, time_limit=10., mip_rel_gap=0., output_flag=False).items():
        assert h.setOptionValue(name, value) == hp.HighsStatus.kOk
    assert h.passModel(model) == hp.HighsStatus.kOk
    start = hp.HighsSolution(); start.col_value = p.vector(feasible[0][1]); start.value_valid = True
    assert h.setSolution(start) == hp.HighsStatus.kOk
    assert h.run() == hp.HighsStatus.kOk
    assert h.getModelStatus() == hp.HighsModelStatus.kOptimal
    assert abs(h.getObjectiveValue()+optimum) < 1e-8
    y = np.rint(np.asarray(h.getSolution().col_value)[:p.n]).astype(np.int8)
    assert p.counts(y)['retained'] == optimum and p.counts(y)['feasible']
    result = dict(passed=True,enumerated_assignments=2**p.n, feasible_assignments=len(feasible),
                  exhaustive_optimum=optimum,solver_optimum=int(-h.getObjectiveValue()),
                  unchanged_balance_constraints_verified=True,objective_offset_verified=True,
                  full_warm_start_accepted=True,production_data_optimized=False)
    write_json(HERE/'qualification.json', result)
    print(json.dumps(dict(event='qualification_passed', **result)), flush=True)


def main():
    assert os.environ.get('SLURM_JOB_ID') and len(os.sched_getaffinity(0)) >= POLICY['threads']
    assert not (HERE/'started.json').exists(), 'One production attempt already started; no automatic second attempt.'
    complete = read_json(ROOT/'completed.json')
    assert complete['complete'] and complete['audit']['passed']
    for name, digest in complete['files'].items(): assert sha(ROOT/name) == digest, name
    old = read_json(ROOT/'reports/positive-split.json')
    for name, digest in old['input_sha256'].items(): assert sha(ROOT/'work'/name) == digest, name
    groups = read_json(ROOT/'work/protein-groups.json')
    assignment = read_json(ROOT/'work/protein-to-development-split.json')
    order = sorted(set(groups.values())); index = {g:i for i,g in enumerate(order)}
    n = len(order); intra = np.zeros(n,dtype=np.int64); cross = np.zeros((n,n),dtype=np.int64)
    for r in csv.DictReader((ROOT/'work/eligible-positives.csv').open()):
        i, j = sorted([index[groups[r['protein1']]], index[groups[r['protein2']]]])
        if i == j: intra[i] += 1
        else: cross[i,j] += 1
    p = Partition(intra,cross); initial = np.full(n,-1,dtype=np.int8)
    for protein, group in groups.items():
        i = index[group]; value = int(assignment[protein] == 'train')
        assert initial[i] in (-1, value); initial[i] = value
    baseline = p.counts(initial)
    assert baseline['feasible'] and baseline['retained'] == old['retained_positives']
    assert baseline['train'] == old['splits']['train']['positives'] and baseline['val'] == old['splits']['val']['positives']
    assert baseline['discarded'] == old['discarded_crossing_positives']
    qualification()
    identities = {str(path.relative_to(ROOT)):sha(path) for path in [
        Path(__file__), HERE/'launch.py', HERE/'qualification.json', ROOT/'configuration.json',
        ROOT/'completed.json', ROOT/'work/eligible-positives.csv', ROOT/'work/protein-groups.json',
        ROOT/'work/protein-to-development-split.json', ROOT/'reports/positive-split.json',
        ROOT/'scripts/python.sh', ROOT/'environment/runtime.json', ROOT/'environment/requirements.lock']}
    contract = dict(created_utc=utc(), policy=POLICY, identities=identities, baseline=baseline,
                    old_gap=old['actual_mip_gap'], old_retained_upper_bound=-old['dual_bound'],
                    minimum_adoptable_retained=math.ceil(baseline['retained']*1.01),
                    formulation='one group assignment and one McCormick product per group pair; equivalent integer partitions',
                    original_completed_manifest_sha256=sha(ROOT/'completed.json'),
                    output_scope='This attempt writes only its own directory; existing completed dataset is preserved.')
    contract['fingerprint'] = hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    write_json(HERE/'contract.json',contract)
    write_json(HERE/'group-order.json',order)
    model,matrix,lower,upper,cost = p.model()
    vector = p.vector(initial); activity = matrix@vector
    assert np.all(activity >= lower-1e-7) and np.all(activity <= upper+1e-7)
    assert abs(cost@vector-p.total+baseline['retained']) < 1e-7
    best = [baseline.copy(),initial.copy()]; last_improvement=[time.monotonic()]
    errors=[]; observed=[False]; stop_reason=[None]
    def save(y,origin):
        counts=p.counts(y); assert counts['feasible']
        if counts['retained'] < best[0]['retained']: return
        if counts['retained'] == best[0]['retained'] and (HERE/'best.json').exists(): return
        digest=hashlib.sha256(np.asarray(y,dtype=np.int8).tobytes()).hexdigest()
        path=HERE/f'assignment-{digest[:20]}.json'
        write_json(path,dict(group_order=order,train_labels=np.asarray(y,dtype=np.int8).tolist(),counts=counts))
        write_json(HERE/'best.json',dict(contract=contract['fingerprint'],counts=counts,origin=origin,
                   assignment_file=path.name,sha256=sha(path),saved_utc=utc()))
        best[:]=[counts,np.asarray(y,dtype=np.int8).copy()]; last_improvement[0]=time.monotonic()
        print(json.dumps(dict(event='checked_incumbent',origin=origin,**counts)),flush=True)
    save(initial,'existing_completed_partition')
    h=hp.Highs(); h.HandleUserInterrupt=True
    options=dict(threads=POLICY['threads'], parallel='on', random_seed=2,
                 time_limit=float(POLICY['solver_seconds']), mip_rel_gap=.01, mip_abs_gap=.5,
                 mip_lp_solver='ipm', mip_ipm_solver='ipx', mip_heuristic_effort=.2,
                 mip_max_start_nodes=0, mip_min_logging_interval=10.,
                 log_file=str(HERE/'highs.log'), log_to_console=True, presolve='on')
    for key,value in options.items(): assert h.setOptionValue(key,value)==hp.HighsStatus.kOk,(key,value)
    assert h.passModel(model)==hp.HighsStatus.kOk
    solution=hp.HighsSolution();solution.col_value=vector;solution.value_valid=True
    assert h.setSolution(solution)==hp.HighsStatus.kOk
    start=time.monotonic(); last_heartbeat=[0.]
    def incumbent(event):
        try:
            values=np.asarray(event.data_out.mip_solution)[:n]
            assert values.shape==(n,) and np.max(np.abs(values-np.rint(values)))<1e-5
            y=np.rint(values).astype(np.int8); assert p.counts(y)['feasible']
            observed[0]=True;save(y,'highs_integer_incumbent')
        except Exception as exc:
            errors.append(repr(exc));event.interrupt()
    def heartbeat(event):
        elapsed=time.monotonic()-start
        if elapsed-last_heartbeat[0]>=15 or not last_heartbeat[0]:
            bound=float(event.data_out.mip_dual_bound)
            write_json(HERE/'heartbeat.json',dict(elapsed_seconds=elapsed,counts=best[0],
                       raw_dual_bound=bound if math.isfinite(bound) else None,
                       solver_observed_feasible_incumbent=observed[0],updated_utc=utc()))
            last_heartbeat[0]=elapsed
        adoptable=(best[0]['retained']>=contract['minimum_adoptable_retained'] and best[0]['train']>=baseline['train'])
        if elapsed>=POLICY['solver_seconds']:
            stop_reason[0]='30_minute_budget';event.interrupt()
        elif adoptable and time.monotonic()-last_improvement[0]>=POLICY['stop_after_seconds_without_improvement_once_adoptable']:
            stop_reason[0]='adoptable_partition_with_five_minutes_without_improvement';event.interrupt()
    h.cbMipSolution+=incumbent;h.cbMipImprovingSolution+=incumbent
    h.cbMipInterrupt+=heartbeat;h.cbIpmInterrupt+=heartbeat;h.cbSimplexInterrupt+=heartbeat
    def interrupted(signum,frame):
        stop_reason[0]=f'external_signal_{signum}';h.cancelSolve()
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with (HERE/'started.json').open('x') as f:
        json.dump(dict(started_utc=utc(),pid=os.getpid(),host=platform.node(),
                       slurm_job_id=os.environ['SLURM_JOB_ID'],contract=contract['fingerprint'],
                       options=options,model_rows=matrix.shape[0],model_columns=matrix.shape[1]),f,indent=2)
        f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(event='one_attempt_started',baseline=baseline,options=options,
                          rows=matrix.shape[0],columns=matrix.shape[1],adoption_threshold=contract['minimum_adoptable_retained'])),flush=True)
    status=h.run();elapsed=time.monotonic()-start
    if errors:raise RuntimeError(str(errors))
    info=h.getInfo();solution=h.getSolution()
    feasible=bool(solution.value_valid and info.primal_solution_status==hp.SolutionStatus.kSolutionStatusFeasible)
    if feasible:
        values=np.asarray(solution.col_value)[:n];assert np.max(np.abs(values-np.rint(values)))<1e-5
        save(np.rint(values).astype(np.int8),'highs_returned_solution')
    assert best[0]['retained']>=baseline['retained'] and p.counts(best[1])['feasible']
    bound=float(info.mip_dual_bound)
    retained_upper=min(float(p.total),-old['dual_bound'])
    if math.isfinite(bound):retained_upper=min(retained_upper,-bound)
    assert retained_upper+1e-5>=best[0]['retained']
    for name,digest in complete['files'].items():assert sha(ROOT/name)==digest,name
    assert sha(ROOT/'completed.json')==contract['original_completed_manifest_sha256']
    result=dict(completed_utc=utc(),contract=contract['fingerprint'],one_attempt_completed=True,
                solver_status=h.getModelStatus().name,highs_run_status=str(status),solver_seconds=elapsed,
                configured_seconds=POLICY['solver_seconds'],stop_reason=stop_reason[0],
                baseline=baseline,best=best[0],additional_retained=best[0]['retained']-baseline['retained'],
                relative_retention_increase=best[0]['retained']/baseline['retained']-1,
                solver_observed_feasible_incumbent=observed[0],returned_solution_feasible=feasible,
                solver_dual_bound=bound if math.isfinite(bound) else None,
                best_retained_upper_bound_from_either_attempt=retained_upper,
                combined_relative_gap=(retained_upper-best[0]['retained'])/best[0]['retained'],
                raw_solver_relative_gap=float(info.mip_gap) if math.isfinite(info.mip_gap) else None,
                nodes=int(info.mip_node_count),original_completed_dataset_hashes_preserved=True,
                meets_adoption_threshold=bool(best[0]['retained']>=contract['minimum_adoptable_retained'] and best[0]['train']>=baseline['train']),
                adoption_completed=False,model_training_started=False)
    write_json(HERE/'result.json',result)
    print(json.dumps(dict(event='one_attempt_complete',**result)),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write_json(HERE/'error.json',dict(utc=utc(),error=repr(exc),completed_dataset_preserved=True))
        raise
