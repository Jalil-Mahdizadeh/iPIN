"""Stdlib-only native SPRINT execution and source-level qualification."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from policy import HSP_ARGS, PREDICT_ARGS, HSP_THREADS


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def write(path, obj):
    with Path(path).open('x') as f: json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)


def run(name, args, threads=1):
    start = time.monotonic()
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    with Path('/output', name + '.log').open('x') as f:
        subprocess.run(['/usr/bin/time', '-v', *args], env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    elapsed = time.monotonic() - start
    print(json.dumps({'operation': name, 'elapsed_seconds': elapsed, 'threads': threads}), flush=True)
    return elapsed


def hsps(path):
    """Read blocks, validate syntax and canonicalize only inter-block order."""
    result = {}; key = None
    with Path(path).open() as f:
        for line in f:
            fields = line.split()
            if fields[0] == '>':
                assert len(fields) == 4 and fields[2] == 'and'
                key = (fields[1], fields[3]); assert key not in result
                result[key] = []
            else:
                assert key is not None and len(fields) == 3
                a, b, n = map(int, fields); assert min(a, b) >= 0 and n >= 20
                result[key].append((a, b, n))
    assert result and all(result.values())
    return result


def canonical(source, target, fasta):
    blocks = hsps(source)
    lines = Path(fasta).read_text().splitlines()
    lengths = {lines[i][1:]: len(lines[i+1]) for i in range(0, len(lines), 2)}
    for name, n in lengths.items(): assert (0, 0, n) in blocks[(name, name)]
    with Path(target).open('x') as f:
        for (a, b), records in sorted(blocks.items()):
            assert a in lengths and b in lengths
            f.write(f'> {a} and {b}\n')
            for x, y, n in records:
                assert x+n <= lengths[a] and y+n <= lengths[b]
                f.write(f'{x} {y} {n}\n')
    return {'protein_pair_blocks': len(blocks), 'hsp_records': sum(map(len, blocks.values())),
            'proteins_with_full_self_hsp': len(lengths), 'canonical_sha256': sha(target)}


def score_values(path, marker=1):
    values = []
    for line in Path(path).read_text().splitlines():
        s, label = line.split(); assert int(label) == marker
        score = float(s); assert math.isfinite(score) and score >= 0
        values.append(score)
    return values


def close(left, right):
    assert len(left) == len(right) and left
    err = max(abs(x-y)/max(1.0, abs(x), abs(y)) for x, y in zip(left, right))
    assert err <= 1e-5, err
    return err


def qualification():
    toy = Path('/opt/sprint/toy_example'); output = Path('/output')
    fasta = toy/'protein_sequences.fasta'
    timings = {}
    for tag, binary, threads in [('serial', 'compute_HSPs_serial', 1), ('parallel', 'compute_HSPs', HSP_THREADS)]:
        timings[tag] = run('toy-hsp-' + tag, ['/opt/sprint/bin/' + binary, '-p', str(fasta), '-h', str(output/(tag+'.hsp')), *HSP_ARGS], threads)
    assert hsps(output/'serial.hsp') == hsps(output/'parallel.hsp'), 'Serial/parallel native HSP records differ'
    census = canonical(output/'parallel.hsp', output/'canonical.hsp', fasta)
    pairs = (toy/'test_positive.txt').read_text() + (toy/'test_negative.txt').read_text()
    (output/'pairs.txt').write_text(pairs); (output/'empty.txt').touch(exist_ok=False)
    names = [line.split() for line in pairs.splitlines()]
    (output/'reverse.txt').write_text(''.join(f'{b} {a}\n' for a, b in reversed(names)))
    outputs = {}
    cases = [('serial', output/'serial.hsp', 'predict_interactions', output/'pairs.txt', False),
             ('canonical', output/'canonical.hsp', 'predict_interactions', output/'pairs.txt', False),
             ('omp1', output/'canonical.hsp', 'predict_interactions_parallel', output/'pairs.txt', False),
             ('routing', output/'canonical.hsp', 'predict_interactions', output/'pairs.txt', True),
             ('reverse', output/'canonical.hsp', 'predict_interactions', output/'reverse.txt', False)]
    for tag, hsp, binary, candidates, neg in cases:
        path = output/(tag+'.scores')
        timings['prediction_'+tag] = run('toy-predict-'+tag, ['/opt/sprint/bin/'+binary,
            '-p', str(fasta), '-h', str(hsp), '-tr', str(toy/'train_positive.txt'),
            '-pos', str(output/'empty.txt' if neg else candidates), '-neg', str(candidates if neg else output/'empty.txt'),
            '-o', str(path), *PREDICT_ARGS])
        outputs[tag] = score_values(path, 0 if neg else 1)
        assert len(outputs[tag]) == len(names)
    assert any(x > 0 for x in outputs['canonical'])
    errors = {name: close(outputs['canonical'], values[::-1] if name == 'reverse' else values) for name, values in outputs.items()}
    # Empty graph must yield genuine native zeros, not a wrapper fill value.
    run('toy-empty-graph', ['/opt/sprint/bin/predict_interactions', '-p', str(fasta), '-h', str(output/'canonical.hsp'),
        '-tr', str(output/'empty.txt'), '-pos', str(output/'pairs.txt'), '-neg', str(output/'empty.txt'),
        '-o', str(output/'empty-graph.scores'), *PREDICT_ARGS])
    assert score_values(output/'empty-graph.scores') == [0.0] * len(names)
    write(output/'NATIVE_QUALIFICATION.json', {'passed': True, 'at_utc': datetime.now(timezone.utc).isoformat(),
        'unmodified_native_sources': True, 'serial_parallel_hsp_records_equal': True,
        'canonical_order_native_score_relative_errors': errors, 'relative_tolerance': 1e-5,
        'native_empty_graph_all_zero': True, 'route_marker_is_not_ground_truth': True,
        'pair_orientation_and_row_permutation_checked': True, 'toy_pairs': len(names),
        'hsp_census': census, 'wall_seconds': timings,
        'cpu_affinity_count': len(os.sched_getaffinity(0)),
        'native_binaries_manifest': Path('/opt/sprint/build/binaries.sha256').read_text()})


def full_hsp(pilot=False):
    output = Path('/output'); fasta = Path('/data/pilot.fasta' if pilot else '/data/proteins.fasta')
    elapsed = run('native-hsp', ['/opt/sprint/bin/compute_HSPs', '-p', str(fasta), '-h', str(output/'raw.hsp'), *HSP_ARGS], HSP_THREADS)
    census = canonical(output/'raw.hsp', output/'canonical.hsp', fasta)
    write(output/'HSP_COMPLETE.json', {'passed': True, 'at_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': elapsed, 'threads': HSP_THREADS, 'input_fasta_sha256': sha(fasta),
        'pilot': pilot, 'test_pairs_read': False, 'test_truth_read': False, **census})


def predict():
    output = Path('/output')
    request = json.loads(Path('/native_input/NATIVE_INPUT.json').read_text())
    for item in request['files']:
        assert sha(Path('/native_input')/item['path']) == item['sha256']
    frozen = json.loads(Path('/bundle/SCORER_FREEZE.json').read_text())
    for item in frozen['files']:
        assert sha(Path('/bundle')/item['path']) == item['sha256']
    assert request['scorer_freeze_sha256'] == sha('/bundle/SCORER_FREEZE.json')
    (output/'empty.txt').touch(exist_ok=False)
    elapsed = run('native-predict', ['/opt/sprint/bin/predict_interactions', '-p', '/bundle/proteins.fasta',
        '-h', '/bundle/canonical.hsp', '-tr', '/bundle/train_positive.txt', '-pos', '/native_input/pairs.txt',
        '-neg', '/output/empty.txt', '-o', '/output/scores.txt', *PREDICT_ARGS])
    write(output/'NATIVE_COMPLETE.json', {'elapsed_seconds': elapsed, 'scores_sha256': sha(output/'scores.txt'),
        'native_input_sha256': sha('/native_input/NATIVE_INPUT.json'), 'test_truth_read': False,
        'scorer_freeze_sha256': sha('/bundle/SCORER_FREEZE.json'), 'binary': 'predict_interactions (native serial)'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['qualify', 'pilot', 'hsp', 'predict'])
    phase = parser.parse_args().phase
    {'qualify': qualification, 'pilot': lambda: full_hsp(True), 'hsp': full_hsp, 'predict': predict}[phase]()
