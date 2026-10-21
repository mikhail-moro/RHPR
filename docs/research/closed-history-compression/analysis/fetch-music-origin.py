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
import json,urllib.request,struct,time,hashlib,os
from concurrent.futures import ThreadPoolExecutor
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(_archive_workspace_path('music-origin'));ROOT.mkdir(exist_ok=True)
REV='dd6f3a19eef5866e346c3270e098baa641a44948'
NAME='sequential/500m/listens.parquet';SIZE=4273773971
BASE=f'https://huggingface.co/datasets/yandex/yambda/resolve/{REV}/{NAME}'
started=time.monotonic()
def ranged(start,end,label):
    request=urllib.request.Request(BASE+f'?download=true&origin_audit={label}',headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(request,timeout=60) as response:
        if response.status!=206:raise RuntimeError(f'HTTP{response.status}: range not supported, stopped without downloading body')
        value=response.read(end-start+2)
        if len(value)!=end-start+1:raise RuntimeError('Unexpected HTTP range response size')
        return value

tail=ranged(SIZE-8,SIZE-1,'tail');assert tail[-4:]==b'PAR1'
length=struct.unpack('<I',tail[:4])[0];assert 0<length<10_000_000
footer=ranged(SIZE-8-length,SIZE-9,'footer')
metadata=pq.read_metadata(pa.BufferReader(b'PAR1'+footer+tail))
assert metadata.num_row_groups==1
columns=[]
for index in range(metadata.num_columns):
    chunk=metadata.row_group(0).column(index)
    name=chunk.path_in_schema.split('.')[0]
    if name not in ('uid','is_organic'):continue
    starts=[value for value in [chunk.dictionary_page_offset,chunk.data_page_offset] if value is not None and value>=0]
    start=min(starts);size=chunk.total_compressed_size
    columns.append({'column':name,'start':start,'end':start+size-1,'bytes':size,'path':str(ROOT/f'{name}.chunk')})
assert {c['column'] for c in columns}=={'uid','is_organic'}
assert sum(c['bytes'] for c in columns)<60_000_000

def download(chunk):
    path=Path(chunk['path'])
    data=ranged(chunk['start'],chunk['end'],chunk['column'])
    path.write_bytes(data)
    result={**chunk,'sha256':hashlib.sha256(data).hexdigest()}
    print(json.dumps({'phase':'downloaded_column',**result,'seconds':time.monotonic()-started}),flush=True)
    return result
with ThreadPoolExecutor(max_workers=2) as pool:columns=list(pool.map(download,columns))

# A sparse local container preserves the source offsets. Only the requested
# column chunks and footer are physically present; no other source bytes exist.
container=ROOT/'listens-organic.partial.parquet'
with container.open('wb') as stream:
    stream.truncate(SIZE)
    stream.seek(0);stream.write(b'PAR1')
    for chunk in columns:
        stream.seek(chunk['start']);stream.write(Path(chunk['path']).read_bytes())
    stream.seek(SIZE-8-length);stream.write(footer);stream.write(tail)
check=pq.ParquetFile(container)
assert check.schema_arrow.names==['uid','timestamp','item_id','is_organic','played_ratio_pct','track_length_seconds']
stat=container.stat()
result={'revision':REV,'source_file':NAME,'source_file_bytes':SIZE,'source_rows':metadata.num_rows,
        'downloaded_columns':columns,'footer_bytes_downloaded':len(footer)+len(tail),
        'network_bytes_downloaded':sum(c['bytes'] for c in columns)+len(footer)+len(tail),
        'container_path':str(container),'container_logical_bytes':stat.st_size,'container_allocated_bytes':stat.st_blocks*512,
        'only_declared_columns_readable':True,'seconds':time.monotonic()-started,
        'no_item_timestamp_ratio_duration_column_bytes_downloaded':True}
(ROOT/'fetch-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
