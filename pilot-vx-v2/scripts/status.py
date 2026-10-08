"""Read-only host status; imports no numerical runtime and no TEST inputs."""
import json
import subprocess
from study import ROOT, read, config


def main():
    result = {}
    for name in ['provenance/submission.json', 'provenance/batch-start.json', 'results/decision.json', 'results/batch-complete.json', 'results/batch-error.json']:
        p = ROOT / name
        if p.exists(): result[name] = read(p)
    result['workers'] = [read(p) for p in sorted((ROOT / 'results').glob('worker-*.json'))]
    submit = ROOT / 'provenance/submission.json'
    if submit.exists():
        job = read(submit)['job_id']
        q = subprocess.run(['squeue', '-j', job, '-o', '%.18i %.9T %.10M %.6D %R'], capture_output=True, text=True)
        result['scheduler'] = q.stdout.strip()
        account = subprocess.run(['sacct', '-j', job, '-X', '--noheader', '--parsable2',
            '--format=JobID,State,ElapsedRaw,AllocCPUS,AllocTRES'], capture_output=True, text=True)
        result['scheduler_accounting'] = account.stdout.strip()
        for line in account.stdout.splitlines():
            fields = line.split('|')
            if len(fields) >= 5 and fields[0] == job and fields[2].isdigit():
                b = config()['budgets']; seconds = int(fields[2])
                result['allocated_resources'] = {'state': fields[1], 'batch_gpu_hours': 4 * seconds / 3600,
                    'batch_cpu_core_hours': int(fields[3]) * seconds / 3600,
                    'cumulative_gpu_hours_with_qualification_reserves': b['prior_gpu_hours_charged'] + b['v2_qualification_gpu_hours_reserved'] + 4 * seconds / 3600,
                    'final_accounting': fields[1] in ['COMPLETED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY'] or fields[1].startswith('CANCELLED')}
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
