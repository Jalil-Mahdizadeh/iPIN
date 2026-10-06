"""Optional read-only mutation split audit inputs, never used for fitting."""
import json
from fetch_resources import ROOT,fetch
name='Mutation_effect_dataset'
meta=json.loads((ROOT/f'provenance/hf-{name}.json').read_text())
entries=json.loads((ROOT/f'provenance/hf-tree-{name}.json').read_text())
files=[fetch((name,meta['sha'],entry)) for entry in entries if entry['path'] in ['training_mutation_data.csv','val_mutation_data.csv']]
(ROOT/'provenance/audit-only-downloads.json').write_text(json.dumps(dict(purpose='Read-only split overlap audit; never passed to model inference or training.',files=files),indent=2)+'\n')
