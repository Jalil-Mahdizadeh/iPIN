"""Read-only Arrhenius resource snapshot for the improvement proposal."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

commands = {
    'gpu': ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'],
    'partition': ['scontrol', 'show', 'partition', 'gpu'],
    'node': ['scontrol', 'show', 'node', 'n423'],
    'interactive_allocation': ['scontrol', 'show', 'job', '3085063'],
    'association': ['sacctmgr', '-n', '-P', 'show', 'assoc', 'where', 'user=jalil',
                    'account=naiss2025-3-10-gpu',
                    'format=Cluster,Account,User,DefaultQOS,QOS,GrpTRESMins,MaxTRESMins,MaxJobs,MaxSubmitJobs'],
    'storage_quota': ['storagequota', '--directory', '/nobackup/proj/disk/theo-storage', '--json'],
    'apptainer': ['apptainer', '--version'],
    'project_usage': ['projinfo', 'naiss2025-3-10-gpu'],
}
data = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'commands': {}}
for name, cmd in commands.items():
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    output = run.stdout + run.stderr
    if name == 'project_usage':
        output = '\n'.join(line for line in output.splitlines()
                           if any(s in line for s in ['Total:', 'NOTE:', 'precise values', 'eventually be fixed']))
    data['commands'][name] = {'command': cmd, 'returncode': run.returncode, 'output': output}
out = Path(__file__).with_name('arrhenius-resources.json')
out.write_text(json.dumps(data, indent=2) + '\n')
print(out)
