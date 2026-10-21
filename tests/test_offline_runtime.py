from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from rhpr.offline_runtime import install_offline_runtime,resolve_dataset_directory,prepare_wheel_links
from scripts.build_kaggle_bundle import build


class OfflineRuntimeTest(unittest.TestCase):
    def test_no_runtime_preserves_existing_runner(self):
        self.assertIsNone(install_offline_runtime(None,Path('/unused')))
        self.assertIsNone(install_offline_runtime({},Path('/unused')))

    def test_wrong_python_or_loaded_torch_fails_before_installation(self):
        runtime={'target_python_minor':'0.0'}
        with patch.dict(sys.modules,{'torch':object()}):
            with self.assertRaisesRegex(RuntimeError,'before importing torch'):
                install_offline_runtime(runtime,Path('/unused'))
        with patch.dict(sys.modules):
            sys.modules.pop('torch',None)
            with self.assertRaisesRegex(RuntimeError,'require Python'):
                install_offline_runtime(runtime,Path('/unused'))

    def test_lock_tampering_is_rejected_before_subprocess(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules):
            sys.modules.pop('torch',None)
            directory=Path(temporary);(directory/'requirements.txt').write_text('torch==1\n')
            runtime={'target_python_minor':f'{sys.version_info.major}.{sys.version_info.minor}',
                     'wheel_path':str(directory),'requirements_file':'requirements.txt','requirements_sha256':'wrong'}
            with patch('rhpr.offline_runtime.subprocess.run') as run:
                with self.assertRaisesRegex(ValueError,'lock differs'):
                    install_offline_runtime(runtime,directory)
                run.assert_not_called()

    def test_profile_bundle_bootstraps_before_gpu_imports_and_stays_private_offline(self):
        with tempfile.TemporaryDirectory() as temporary:
            metadata=json.loads(build(ROOT/'experiments/yambda_long_training_profile.json',
                                      'owner/profile','gpu',Path(temporary)).read_text())
            self.assertTrue(metadata['is_private'])
            self.assertFalse(metadata['enable_internet'])
            source=(Path(temporary)/'run.py').read_text()
            ast.parse(source)
            self.assertLess(source.index('install_offline_runtime(json.loads'),source.index('from rhpr.runner import execute'))
            self.assertEqual(len(metadata['dataset_sources']),2)

    def test_sanitized_cuda_wheel_names_are_restored_without_copying_or_changing_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'source';source.mkdir()
            actual=source/'torch-2.10.0cu128-cp313-cp313-linux_x86_64.whl'
            actual.write_bytes(b'original wheel bytes')
            wanted=actual.name.replace('2.10.0cu128','2.10.0+cu128')
            manifest={'wheels':[{'filename':wanted,'size_bytes':actual.stat().st_size}]}
            destination=Path(temporary)/'links'
            repairs=prepare_wheel_links(source,destination,manifest)
            self.assertEqual(repairs,[{'mounted_name':actual.name,'restored_name':wanted}])
            self.assertTrue((destination/wanted).is_symlink())
            self.assertTrue((destination/wanted).samefile(actual))
            self.assertEqual((destination/wanted).read_bytes(),b'original wheel bytes')
            self.assertEqual(prepare_wheel_links(source,destination,manifest),repairs)

    def test_runtime_links_reject_unsafe_names_wrong_sizes_and_wrong_existing_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'source';source.mkdir();(source/'package.whl').write_bytes(b'x')
            destination=Path(temporary)/'links'
            for row in [{'filename':'../package.whl','size_bytes':1},
                        {'filename':'package.whl','size_bytes':2}]:
                with self.assertRaises(ValueError):prepare_wheel_links(source,destination,{'wheels':[row]})
            destination.mkdir(exist_ok=True)
            (destination/'package.whl').write_bytes(b'wrong existing file')
            with self.assertRaises(ValueError):
                prepare_wheel_links(source,destination,{'wheels':[{'filename':'package.whl','size_bytes':1}]})


if __name__=='__main__':unittest.main()
