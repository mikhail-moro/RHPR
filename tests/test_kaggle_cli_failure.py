"""Keep the actual CLI failure visible without exposing authentication."""
import contextlib,importlib.util,io,os,subprocess,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('kaggle_job_failure_fixture',ROOT/'scripts/kaggle_job.py')
job=importlib.util.module_from_spec(s);s.loader.exec_module(job)

class FailureTests(unittest.TestCase):
    def test_server_reason_is_logged_and_sensitive_values_redacted(self):
        failure=subprocess.CalledProcessError(1,['kaggle','kernels','push'],output='request rejected',
            stderr='reason: dataset unavailable; token TOKEN_VALUE; Bearer OTHER_VALUE; https://example.org/file?signature=SIGNED')
        capture=io.StringIO()
        with patch.dict(os.environ,{'KAGGLE_API_TOKEN':'TOKEN_VALUE'}),patch.object(job.subprocess,'run',side_effect=failure),contextlib.redirect_stdout(capture):
            with self.assertRaises(subprocess.CalledProcessError):job.run(['kaggle','kernels','push'])
        log=capture.getvalue();self.assertIn('dataset unavailable',log)
        for secret in ['TOKEN_VALUE','OTHER_VALUE','SIGNED']:self.assertNotIn(secret,log)

    def test_successful_command_preserves_completed_process(self):
        result=subprocess.CompletedProcess(['kaggle'],0,stdout='ok',stderr='')
        with patch.object(job.subprocess,'run',return_value=result),contextlib.redirect_stdout(io.StringIO()):
            self.assertIs(job.run(['kaggle']),result)

if __name__=='__main__':unittest.main()
