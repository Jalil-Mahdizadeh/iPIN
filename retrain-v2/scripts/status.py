"""Read v2 production status without loading models or starting jobs."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
campaign=json.loads((root/'configs/campaign.json').read_text())
records=[]
for name in campaign['groups']['initial-screen']:
    folder=root/'runs'/name
    state=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
    best=json.loads((folder/'best.json').read_text()) if (folder/'best.json').exists() else {}
    phase=state.get('phase','not started')
    if (folder/'stopped.json').exists():phase='stopped'
    if (folder/'completed.json').exists():phase='complete'
    record={'run':name,'phase':phase,'update':state.get('state',{}).get('update',0),
            'last_committed_update':state.get('last_committed_update'),
            'best_update':best.get('update'),'best_pooled_validation_ap':best.get('best_ap')}
    records.append(record)
print(json.dumps({'production_runs':records,'qualification_reports':sorted(p.name for p in (root/'qualification').glob('*.json')
    if 'config' not in p.name)},indent=2))
