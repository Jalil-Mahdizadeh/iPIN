"""Direct alignment-string check of the one apparent exported-count boundary case."""
import csv,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    meta=json.loads((ROOT/'data/nonhuman/sequences.json').read_text())
    with (ROOT/'work/nonhuman/backtrace-v5.tsv').open() as f:
        cases=[r for r in csv.reader(f,delimiter='\t') if 4*int(r[9])<int(r[10])]
    assert len(cases)==1
    case=cases[0];q,t=int(case[0][1:]),int(case[1][1:])
    query=ROOT/'work/nonhuman/boundary-query.fasta';query.write_text('>'+case[0]+'\n'+meta['sequence'][q]+'\n')
    output=ROOT/'work/nonhuman/boundary-alignment.tsv'
    search=json.loads((ROOT/'provenance/nonhuman/search-v5.json').read_text())
    command=list(search['identity']['command']);command[2]=str(query);command[4]=str(output)
    command[5]=str(ROOT/'work/nonhuman/boundary-tmp');command[command.index('--threads')+1]='4'
    command[command.index('--format-output')+1]='query,target,fident,nident,alnlen,qaln,taln,bits,evalue'
    command+=['-a','1']
    if not output.exists():
        with (ROOT/'logs/nonhuman/boundary-alignment.log').open('w') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    with output.open() as f:rows=[r for r in csv.reader(f,delimiter='\t') if r[1]==case[1]]
    assert len(rows)==1;r=rows[0];qaln,taln=r[5:7]
    assert len(qaln)==len(taln)==int(r[4])==int(case[10])
    assert qaln.replace('-','')==meta['sequence'][q][int(case[11])-1:int(case[12])]
    assert taln.replace('-','')==meta['sequence'][t][int(case[13])-1:int(case[14])]
    identical=sum(a==b and a!='-' for a,b in zip(qaln,taln));assert 4*identical>=len(qaln)
    result={'query':r[0],'target':r[1],'reported_fident':float(r[2]),'reported_nident':int(r[3]),'alignment_length':int(r[4]),
        'identical_residues_counted_directly':identical,'direct_identity':identical/len(qaln),'command':command,
        'source':'work/nonhuman/boundary-alignment.tsv','source_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'raw_sequence_alignment_segments_verified':True,'scores_changed':False,
        'interpretation':'The apparent below-threshold nident export is resolved by direct aligned-residue counting; the retained alignment passes 25% identity.'}
    (ROOT/'provenance/nonhuman/identity-boundary-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Boundary alignment verified:',identical,'/',len(qaln),'identical residues')

if __name__=='__main__':main()
