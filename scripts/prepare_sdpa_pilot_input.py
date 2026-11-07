"""Package public CPython3.13 wheels only; no private data or model artifacts."""
import argparse,hashlib,json,shutil,zipfile
from email.parser import Parser
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare(wheels, out):
    out.mkdir(parents=True,exist_ok=True);locks=[]
    for p in sorted(wheels.glob('*.whl')):
        with zipfile.ZipFile(p)as z:
            name=next(n for n in z.namelist()if n.endswith('.dist-info/METADATA'))
            meta=Parser().parsestr(z.read(name).decode())
        if meta['Name'].lower().replace('_','-')in {'torch','triton','torchvision','torchaudio'}or meta['Name'].lower().startswith('nvidia-'):
            raise ValueError('pilot must preserve Kaggle CUDA/PyTorch')
        shutil.copy2(p,out/p.name);locks.append(f"{meta['Name']}=={meta['Version']} --hash=sha256:{sha(p)}")
    (out/'requirements-lock.txt').write_text('\n'.join(locks)+'\n')
    files=[dict(path=p.name,size_bytes=p.stat().st_size,sha256=sha(p))for p in sorted(out.iterdir())if p.is_file()and p.name not in {'resources-manifest.json','dataset-metadata.json'}]
    manifest=dict(files=files,transformers='5.22.1',python='3.13',network_required=False,content='Public PyPI packages only; no model weights or user data')
    (out/'resources-manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'dataset-metadata.json').write_text(json.dumps(dict(title='RHPR T5 SDPA runtime',id='carasik/rhpr-t5-sdpa-runtime',licenses=[dict(name='other')]),indent=2))
    print(json.dumps(dict(files=len(files),bytes=sum(f['size_bytes']for f in files),manifest_sha256=sha(out/'resources-manifest.json')),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wheels',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.wheels,a.output)
