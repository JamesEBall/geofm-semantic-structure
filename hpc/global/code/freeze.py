"""Freeze global extraction and readout source before scoring. James Edward Ball."""
from pathlib import Path
import hashlib, json, sys, datetime
R=Path(__file__).resolve().parents[1]
OUT=R/'results/extraction-freeze.json'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''): h.update(b)
    return h.hexdigest()
def main():
    if '--check' in sys.argv:
        frozen=json.loads(OUT.read_text())
        for name,digest in frozen['sha256'].items():
            if sha(R/name)!=digest: raise ValueError('Frozen extraction changed: '+name)
        for relative,digest in json.loads((R/'code/acquisition-freeze.json').read_text()).items():
            if sha(R/relative)!=digest: raise ValueError('Frozen acquisition changed: '+relative)
        if '--cohort' in sys.argv:
            for relative,digest in json.loads((R/'results/cohort-freeze.json').read_text()).items():
                if sha(R/relative)!=digest: raise ValueError('Frozen cohort changed: '+relative)
            if not json.loads((R/'results/aggregate-report.json').read_text()).get('admitted',False):
                raise ValueError('Global cohort failed admission')
        print('Global extraction specification verified')
    else:
        if OUT.exists(): raise RuntimeError('Freeze already exists')
        paths=[R/'code'/name for name in ['extract.py','analyse.py','freeze.py','test_analysis.py','global-protocol.md']]
        paths += sorted(R.glob('*.sbatch')) + [R/'code/acquisition-freeze.json']
        OUT.parent.mkdir(parents=True,exist_ok=True)
        OUT.write_text(json.dumps(dict(author='James Edward Ball',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            sha256={str(p.relative_to(R)):sha(p) for p in paths}),indent=2)+'\n')
        print('Frozen global extraction and analysis')
if __name__=='__main__':main()
