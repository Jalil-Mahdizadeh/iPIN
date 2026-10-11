"""Independent synthetic oracles for scientific metrics and sampling."""
import itertools
import numpy as np
from io_v8 import ROOT, write, now
from metrics import (queries, concordance, retrieval, within_anchor,
                     select_quartets, quartet_metrics, component_draws)


def main():
    checks = []
    for s, y in [(np.array([2., 2., 1., 1.]), np.array([1, 0, 1, 0], bool)),
                  (np.ones(3), np.array([1, 0, 0], bool)),
                  (np.array([4., 3., 2., 1.]), np.array([1, 1, 0, 0], bool))]:
        expected = []
        for perm in itertools.permutations(range(len(s))):
            ss, yy = s[list(perm)], y[list(perm)]
            if np.any(ss[1:] > ss[:-1]):
                continue
            row = []
            for k in [1, 2, 5]:
                kk = min(k, len(s)); discounts = 1 / np.log2(np.arange(kk) + 2)
                row.extend([yy[:kk].sum() / yy.sum(), np.dot(yy[:kk], discounts) / discounts[:min(kk, yy.sum())].sum()])
            expected.append(row)
        np.testing.assert_allclose(retrieval(s, y, [1, 2, 5]), np.mean(expected, 0), atol=1e-14, rtol=0)
        checks.append('exhaustive_tie_permutation_' + str(len(checks)))
    p, u = np.array([1., 2., 2.]), np.array([0., 2., 3.])
    pw, uw = np.array([2., 1., 3.]), np.array([4., 2., 1.])
    expected = sum(pw[i] * uw[j] * ((p[i] > u[j]) + .5 * (p[i] == u[j]))
                   for i in range(len(p)) for j in range(len(u))) / (pw.sum() * uw.sum())
    assert abs(concordance(p, u, pw, uw) - expected) < 1e-14
    checks.append('weighted_concordance_bruteforce')
    a = np.array([0, 2, 0, 1, 0, 1]); b = np.array([1, 3, 3, 2, 2, 3])
    y = np.array([1, 1, 0, 0, 0, 0], bool)
    rows, endpoints, count = select_quartets(a, b, y, np.arange(len(a)), 42, 100, 20, 40)
    assert count == 2 and len(rows) == 2
    for r, ep in zip(rows, endpoints):
        assert len(set(ep)) == 4 and y[r[:2]].all() and not y[r[2:]].any()
        assert sorted(np.r_[a[r[:2]], b[r[:2]]]) == sorted(np.r_[a[r[2:]], b[r[2:]]])
    checks.append('both_existing_alternative_matchings')
    comp = np.array([0, 0, 1, 2])
    counts = np.array([[1, 1, 1], [2, 1, 0], [2, 3, 4], [0, 1, 2]])
    unary = np.array([.25, -1.37, 3.41, 2.3])
    scores = np.c_[unary[a] + unary[b], np.array([3., 4., 0., 1., 5., 5.])]
    point, draws, credits = quartet_metrics(scores, rows, endpoints, comp, counts)
    np.testing.assert_array_equal(credits[:, 0], .5)
    for i, c in enumerate(counts):
        w = np.array([np.prod(c[list(set(comp[ep]))]) for ep in endpoints], float)
        if w.sum():
            np.testing.assert_allclose(draws[i], np.average(credits, weights=w, axis=0), atol=1e-14, rtol=0)
        else:
            assert np.isnan(draws[i]).all()
    checks.append('quartet_shared_component_once_and_zero_mass')
    qs = queries(a, b, y)
    qp, qb, qv = within_anchor(scores, qs, comp, counts)
    for i, c in enumerate(counts):
        totals = np.zeros(2); mass = 0.
        for anchor, pi, ui, pp, up in qs:
            ac = comp[anchor]
            pw = np.array([1 if comp[x] == ac else c[comp[x]] for x in pp])
            uw = np.array([1 if comp[x] == ac else c[comp[x]] for x in up])
            if not pw.sum() or not uw.sum():
                continue
            mass += c[ac]
            for j in range(2):
                numerator = sum(pw[k] * uw[l] * ((scores[p, j] > scores[u, j]) + .5*(scores[p, j] == scores[u, j]))
                                for k, p in enumerate(pi) for l, u in enumerate(ui))
                totals[j] += c[ac] * numerator / (pw.sum() * uw.sum())
        if mass:
            np.testing.assert_allclose(qb[i], totals/mass, atol=1e-14, rtol=0)
        else:
            assert np.isnan(qb[i]).all()
    checks.append('within_anchor_shared_component_bruteforce')
    ci, sample, labels = component_draws(comp, 101, 2000)
    assert (sample.sum(1) == len(labels)).all()
    np.testing.assert_array_equal(sample, component_draws(comp, 101, 2000)[1])
    checks.append('bootstrap_count_census_and_determinism')
    reversed_rows = select_quartets(b, a, y, np.arange(len(a)), 42, 100, 20, 40)[0]
    # Reversing each query must preserve the set of admissible edge quartets.
    assert {tuple(sorted(r.tolist())) for r in rows} == {tuple(sorted(r.tolist())) for r in reversed_rows}
    checks.append('query_orientation_invariance')
    bad = False
    try:
        select_quartets(np.r_[a, a[0]], np.r_[b, b[0]], np.r_[y, y[0]], np.arange(len(a)+1), 1, 100, 20, 40)
    except ValueError:
        bad = True
    assert bad
    checks.append('duplicate_candidate_fail_closed')
    write(ROOT / 'qualification/metrics.json', {'at_utc': now(), 'passed': True, 'checks': checks,
                                             'source': 'synthetic fixtures only; no biological scores or outcomes'})
    print({'qualification_passed': True, 'checks': len(checks)}, flush=True)


if __name__ == '__main__':
    main()
