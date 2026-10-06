"""Stream the publisher XLSX using stdlib; keep published scores separate."""
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LIT = ROOT.parent/'literature/plm-interact'
OUT = ROOT/'data/published'
OUT.mkdir(exist_ok=True)
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'

def rows(z, xml, strings):
    with z.open(xml) as f:
        context = ET.iterparse(f, events=('start','end'))
        _, root = next(context)
        for event, elem in context:
            if event != 'end' or elem.tag != NS+'row':
                continue
            cells = {}
            for cell in elem:
                col = ''.join(c for c in cell.attrib['r'] if c.isalpha())
                val = cell.find(NS+'v')
                if val is not None:
                    text = val.text or ''
                    if cell.attrib.get('t') == 's': text = strings[int(text)]
                    cells[col] = text
                elif cell.attrib.get('t') == 'inlineStr':
                    cells[col] = ''.join(cell.itertext())
            yield int(elem.attrib['r']), cells
            elem.clear()
            root.clear()

def write(path, fields, data):
    with open(OUT/path, 'w') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(data)

manifest = []
with zipfile.ZipFile(LIT/'source-data.xlsx') as z:
    strings=[''.join(e.itertext()) for e in ET.fromstring(z.read('xl/sharedStrings.xml'))]
    species=['mouse','fly','worm','yeast','ecoli']
    models=['PLM-interact','TUnA','TT3D','Topsy-Turvy','D-SCRIPT']
    for sn, model in enumerate(models,2):
        data={s:[] for s in species}
        for rn, row in rows(z,f'xl/worksheets/sheet{sn}.xml',strings):
            if rn==1: continue
            for i,s in enumerate(species):
                cols=[chr(65+4*i+j) for j in range(4)]
                if row.get(cols[2],'')=='':continue
                data[s].append(dict(source_row=rn,protein_a=row[cols[0]],protein_b=row[cols[1]],score=row[cols[2]],label=row[cols[3]]))
        for s,d in data.items():
            path=f'figure2_{model}_{s}.csv'
            write(path,['source_row','protein_a','protein_b','score','label'],d)
            manifest.append(dict(path=path, sheet=sn, n=len(d)))
        print('Extracted Figure2',model,flush=True)
    for sn, model in [(9,'PLM-interact'),(10,'TUnA')]:
        d=[]
        for rn,row in rows(z,f'xl/worksheets/sheet{sn}.xml',strings):
            if rn==1:
                score_col=next(k for k,v in row.items() if v.startswith('Pred_score'))
                label_col=next(k for k,v in row.items() if v.lower()=='label')
            if rn>1 and row.get('C','')!='':
                d.append(dict(source_row=rn,protein_a=row['A'],protein_b=row['B'],score=row[score_col],label=row[label_col]))
        path=f'figure4_{model}.csv'
        write(path,['source_row','protein_a','protein_b','score','label'],d)
        manifest.append(dict(path=path,sheet=sn,n=len(d)))
    rs=rows(z,'xl/worksheets/sheet11.xml',strings)
    _,header=next(rs)
    d=[{v:row.get(k,'') for k,v in header.items()} for _,row in rs]
    write('figure5_mutation.csv',list(header.values()),d)
    manifest.append(dict(path='figure5_mutation.csv',sheet=11,n=len(d)))
    for sn,s in enumerate(species,20):
        d=[]
        for rn,row in rows(z,f'xl/worksheets/sheet{sn}.xml',strings):
            if rn>1 and row.get('A','')!='':
                d.append(dict(source_row=rn,original=row['A'],label=row['B'],reverse=row['C'],reverse_label=row['D']))
        path=f'supplement6_{s}.csv'
        write(path,['source_row','original','label','reverse','reverse_label'],d)
        manifest.append(dict(path=path,sheet=sn,n=len(d)))
    # Figure7 contains summary metrics followed by training sequences, not
    # held-out model scores. Record the summary only; no viral inference task.
    summary=[]
    for rn,row in rows(z,'xl/worksheets/sheet13.xml',strings):
        if rn>6:break
        if 3<=rn<=6:
            summary.append(dict(model=row['A'],reported_AUPR=row['B'],reported_F1=row['C'],reported_MCC=row['D']))
    (OUT/'figure7_reported_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for m in manifest:
    m['sha256']=hashlib.sha256((OUT/m['path']).read_bytes()).hexdigest()
(ROOT/'provenance/published-score-extraction.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Finished:',len(manifest),'score tables',flush=True)
