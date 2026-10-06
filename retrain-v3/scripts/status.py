"""Read v3 production status without loading models or starting jobs."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
campaign=json.loads((root/'configs/campaign.json').read_text())
records=[]
names=list(campaign['initial_runs'])
pointer=root/'releases/CURRENT_READOUT'
if pointer.exists():
    release=root/'releases'/pointer.read_text().strip()
    names+=json.loads((release/'release.json').read_text())['enabled_runs']
pointer=root/'releases/CURRENT_FINAL'
if pointer.exists():
    release=root/'releases'/pointer.read_text().strip()
    names+=json.loads((release/'release.json').read_text())['enabled_runs']
for name in names:
    folder=root/'runs'/name
    state=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
    best=json.loads((folder/'best.json').read_text()) if (folder/'best.json').exists() else {}
    phase=state.get('phase','not started')
    if (folder/'stopped.json').exists():
        phase='stopped'
        state['state']=json.loads((folder/'stopped.json').read_text())['state']
    if (folder/'SCREEN_COMPLETE.json').exists():phase='screen complete'
    if (folder/'latest.json').exists():state['last_committed_update']=json.loads((folder/'latest.json').read_text())['update']
    if (folder/'completed.json').exists():phase='complete'
    if name=='clean-residue-mean-official-seed2' and (folder/'completed.json').exists():
        phase='training complete; benchmarking deferred'
        state['state']=json.loads((folder/'completed.json').read_text())['state']
        if (root/'FINALIZED.json').exists():phase='v3 finalized'
    record={'run':name,'phase':phase,'update':state.get('state',{}).get('update',0),
            'last_committed_update':state.get('last_committed_update'),
            'best_update':best.get('update'),'best_pooled_validation_ap':best.get('best_ap')}
    records.append(record)
print(json.dumps({'production_runs':records,'qualification_reports':sorted(p.name for p in (root/'qualification').glob('*.json')
    if 'config' not in p.name)},indent=2))
