# BEGIN ARCHIVE PATH ADAPTER: no research logic below this block is changed.
import os as _archive_os
from pathlib import Path as _ArchivePath

_archive_repo_root = _ArchivePath(__file__).resolve().parents[4]
_archive_workspace_root = _ArchivePath(
    _archive_os.environ.get("RHPR_RESEARCH_WORKSPACE", "/workspace")
).expanduser().resolve()
if (_archive_workspace_root == _archive_repo_root
        or _archive_repo_root in _archive_workspace_root.parents):
    raise ValueError("RHPR_RESEARCH_WORKSPACE must be outside the repository checkout")


def _archive_workspace_path(relative):
    return str(_archive_workspace_root / relative)


def _archive_repo_src_path():
    return str(_archive_repo_root / "src")
# END ARCHIVE PATH ADAPTER

from pathlib import Path
import urllib.request, json, hashlib, struct, time
from concurrent.futures import ThreadPoolExecutor
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(_archive_workspace_path('music-metadata'));ROOT.mkdir(exist_ok=True)
REV='dd6f3a19eef5866e346c3270e098baa641a44948'
BASE=f'https://huggingface.co/datasets/yandex/yambda/resolve/{REV}/'
SIZES={'artist_item_mapping.parquet':38274621,'album_item_mapping.parquet':54528894}

def download(entry):
    name,size=entry; target=ROOT/name; start=time.monotonic(); transferred=0
    if not target.exists() or target.stat().st_size!=size:
        with urllib.request.urlopen(BASE+name,timeout=60) as response, target.open('wb') as out:
            while True:
                block=response.read(1024*1024)
                if not block:break
                out.write(block);transferred+=len(block)
    assert target.stat().st_size==size
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    metadata=pq.read_metadata(target)
    result={'file':name,'bytes':size,'downloaded_bytes':transferred,'sha256':digest,'schema':str(metadata.schema.to_arrow_schema()),'rows':metadata.num_rows,'row_groups':metadata.num_row_groups,'seconds':time.monotonic()-start}
    print(json.dumps(result),flush=True)
    return result

with ThreadPoolExecutor(max_workers=2) as pool:
    files=list(pool.map(download,SIZES.items()))

name='sequential/500m/listens.parquet';size=4273773971
def ranged(start,end,label):
    req=urllib.request.Request(BASE+name+f'?download=true&metadata_probe={label}',headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(req,timeout=40) as response:
        if response.status!=206:
            raise RuntimeError(f'Range request refused with status {response.status}; no source data downloaded')
        value=response.read(end-start+2)
        if len(value)!=end-start+1:raise RuntimeError('Unexpected range response size')
        return value
try:
    tail=ranged(size-8,size-1,'tail');assert tail[-4:]==b'PAR1'
    length=struct.unpack('<I',tail[:4])[0]
    assert 0<length<10_000_000
    footer=ranged(size-8-length,size-9,'footer')
    metadata=pq.read_metadata(pa.BufferReader(b'PAR1'+footer+tail))
    source={'file':name,'source_bytes':size,'downloaded_bytes':len(tail)+len(footer),'schema':str(metadata.schema.to_arrow_schema()),'columns':metadata.schema.to_arrow_schema().names,'rows':metadata.num_rows,'row_groups':metadata.num_row_groups,'footer_sha256':hashlib.sha256(footer).hexdigest(),
            'column_chunk_compressed_bytes':{metadata.row_group(0).column(i).path_in_schema:metadata.row_group(0).column(i).total_compressed_size for i in range(metadata.num_columns)}}
except Exception as exc:
    source={'file':name,'source_bytes':size,'error_type':type(exc).__name__,'error':str(exc)}
result={'revision':REV,'mapping_files':files,'source_footer_audit':source,'network_downloaded_bytes':sum(f['downloaded_bytes'] for f in files)+source.get('downloaded_bytes',0)}
(ROOT/'fetch-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
