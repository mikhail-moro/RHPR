"""Load byte-pinned whole-path graph indices; never fit or regenerate an SID."""
import json
from pathlib import Path
import shutil

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.resources import verify_resources
from rhpr.gr.sid import SidIndex


def load_generator_index(directory, width):
    """Old semantic-code suites retain ordinal construction; graph suites don't."""
    directory = Path(directory)
    record = json.loads((directory/'index-provenance.json').read_text())
    if record.get('preserve_full_sids'):
        return SidIndex.from_full_sids(np.load(directory/'semantic-ids.npy',allow_pickle=False),width)
    return SidIndex(np.load(directory/'semantic-codes.npy',allow_pickle=False),width)


def prepare_full_sid_indices(config, source_ids, output):
    from rhpr.offline_runtime import resolve_dataset_directory
    spec = config['frozen_indices']
    assets = resolve_dataset_directory(spec['path'])
    manifest = verify_resources(assets,spec['manifest_sha256'])
    if (manifest['kind'] != 'frozen_sid_graph_placement'
            or manifest['data_sha256'] != config['data']['sha256']
            or json.loads((assets/'catalogue.json').read_text())['original_item_ids'] != source_ids):
        raise ValueError('frozen graph index catalogue/protocol mismatch')
    entries = {r['path']:r for r in manifest['files']}
    records = {}
    for trial in config['trials']:
        name = trial['id']; candidate = manifest['indices'].get(trial['asset_id'])
        if candidate is None:
            raise ValueError('unknown frozen graph candidate')
        original_control = trial.get('use_original_paths', False)
        if original_control and (original_control is not True or trial['role'] != 'original'
                or trial['semantic_lambda'] != 0 or trial.get('expected_variant') != 'original'):
            raise ValueError('original-path control requires explicit original identity')
        if (candidate['family'] != trial['family'] or (not original_control and
                (candidate['role'] != trial['role']
                 or candidate['semantic_lambda'] != trial['semantic_lambda']))):
            raise ValueError('frozen graph candidate identity mismatch')
        if (not original_control and 'expected_variant' in trial
                and candidate.get('variant') != trial['expected_variant']):
            raise ValueError('frozen graph proposal/signal variant mismatch')
        # Only paths registered in the verified resource manifest can be opened.
        names = [candidate[k] for k in ['sids','original_sids','owner']]
        if any(n not in entries for n in names):
            raise ValueError('graph paths not pinned in resource manifest')
        paths, original, owner = [np.load(assets/n,allow_pickle=False) for n in names]
        if (paths.shape != (len(source_ids),config['quantizer']['num_hierarchies']+1)
                or owner.shape != (len(source_ids),) or owner.dtype.kind not in 'iu'
                or not np.array_equal(np.sort(owner),np.arange(len(source_ids)))
                or original.shape != paths.shape or not np.array_equal(paths,original[owner])):
            raise ValueError('graph candidate owner permutation or full paths mismatch')
        source_path = candidate['sids']
        if original_control:
            source_path = candidate['original_sids']
            paths = original
            owner = np.arange(len(source_ids), dtype=np.int64)
            candidate = {'family': candidate['family'], 'role': 'original',
                'variant': 'original', 'semantic_lambda': 0, 'sids': source_path,
                'original_sids': source_path, 'owner': None,
                'derived_from_asset': trial['asset_id'], 'owner_construction': 'identity'}
        index = SidIndex.from_full_sids(paths,width=config['quantizer']['codebook_size'])
        if (trial['role']=='original' and np.any(owner!=np.arange(len(owner)))):
            raise ValueError('original control was permuted')
        directory = Path(output)/'indices'/name;directory.mkdir(parents=True)
        # Preserve input .npy bytes, including supplied suffix ownership.
        shutil.copyfile(assets/source_path,directory/'semantic-ids.npy')
        if original_control:
            np.save(directory/'owner.npy',owner,allow_pickle=False)
        else:
            shutil.copyfile(assets/candidate['owner'],directory/'owner.npy')
        np.save(directory/'semantic-codes.npy',index.semantic_codes,allow_pickle=False)
        diagnostics=index.diagnostics()
        (directory/'sid-diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
        record=dict(trial=name,preserve_full_sids=True,source=candidate,
            asset_manifest_sha256=spec['manifest_sha256'],quantizer_retrained=False,
            autoencoder_retrained=False,semantic_ids_sha256=file_sha256(directory/'semantic-ids.npy'),
            diagnostics=diagnostics)
        (directory/'index-provenance.json').write_text(json.dumps(record,indent=2)+'\n');records[name]=record
    (Path(output)/'sid-catalogue.json').write_text(json.dumps(dict(original_item_ids=source_ids,
        row_order='internal_item_id_0_based',indices=records),indent=2)+'\n')
    return records
