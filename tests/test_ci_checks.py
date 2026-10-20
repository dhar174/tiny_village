"""Exercise the real CI worker/supervisor using controlled test programs."""

import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TestCIChecks(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / 'scripts/run_ci_checks.py').is_file(),
                        'The bounded CI runner has not been implemented')
        self.ci = importlib.import_module('scripts.run_ci_checks')
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def worker(self, source, *, kind='unittest'):
        fixture = self.root / 'test_fixture.py'
        fixture.write_text(source)
        report = self.root / 'result.json'
        command = [sys.executable, '-B', str(ROOT/'scripts/run_ci_checks.py'),
                   '--worker', kind, '--report', str(report), str(fixture)]
        batch = self.ci.supervise(command, report, self.root/'batch', timeout=10)
        return batch, json.loads(report.read_text()) if report.exists() else None

    def test_success_counts_real_methods(self):
        batch, report = self.worker('import unittest\nclass Sample(unittest.TestCase):\n def test_a(self): pass\n def test_b(self): pass\n')
        self.assertEqual(batch['status'], 'PASS')
        self.assertEqual(report['counts']['tests_run'], 2)
        self.assertEqual(report['counts']['errors'], 0)

    def test_failing_and_erroring_tests_fail_the_gate(self):
        batch, report = self.worker('import unittest\nclass Sample(unittest.TestCase):\n def test_a(self): self.fail("controlled failure")\n def test_b(self): raise RuntimeError("controlled error")\n')
        self.assertEqual(batch['status'], 'FAIL')
        self.assertEqual(report['counts']['tests_run'], 2)
        self.assertEqual(report['counts']['failures'], 1)
        self.assertEqual(report['counts']['errors'], 1)

    def test_failed_subtests_are_not_reported_as_extra_methods(self):
        batch, report = self.worker('import unittest\nclass Sample(unittest.TestCase):\n def test_a(self):\n  for i in range(2):\n   with self.subTest(i=i): self.fail("controlled subtest")\n')
        self.assertEqual(batch['status'], 'FAIL')
        self.assertEqual(report['counts']['tests_run'], 1)
        self.assertEqual(report['counts']['failures'], 2)

    def test_empty_suite_cannot_pass(self):
        batch, report = self.worker('import unittest\n')
        self.assertEqual(batch['status'], 'FAIL')
        self.assertEqual(report['counts']['tests_run'], 0)

    def test_skipped_suite_cannot_pass(self):
        batch, report = self.worker('import unittest\nclass Sample(unittest.TestCase):\n @unittest.skip("controlled skip")\n def test_a(self): pass\n')
        self.assertEqual(batch['status'], 'FAIL')
        self.assertEqual(report['counts']['skipped'], 1)

    def test_import_error_is_a_reported_failure(self):
        batch, report = self.worker('raise ImportError("controlled missing dependency")\n')
        self.assertEqual(batch['status'], 'FAIL')
        self.assertGreater(report['counts']['errors'], 0)

    def test_success_report_cannot_override_nonzero_exit(self):
        report = self.root/'result.json'
        payload = json.dumps({'status':'PASS','kind':'unittest',
                                      'counts':{'tests_run':1,'failures':0,'errors':0,'skipped':0,
                                                'expected_failures':0,'unexpected_successes':0}})
        source = f'from pathlib import Path; Path({str(report)!r}).write_text({payload!r}); raise SystemExit(7)'
        result = self.ci.supervise([sys.executable, '-c', source],
                                   report, self.root/'batch', timeout=5)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['exit_code'], 7)

    def test_missing_or_incomplete_report_cannot_pass(self):
        for index, status in enumerate((None, 'RUNNING')):
            report = self.root/f'result-{index}.json'
            if status:
                report.write_text(json.dumps({'status':status}))
            result = self.ci.supervise([sys.executable, '-c', 'pass'], report,
                                       self.root/f'batch-{index}', timeout=5)
            self.assertEqual(result['status'], 'FAIL')

    def test_timeout_kills_owned_descendants_and_preserves_unrelated_process(self):
        child_pid_file = self.root/'child.pid'
        source = ("import subprocess,sys,time; from pathlib import Path; "
                  "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
                  f"Path({str(child_pid_file)!r}).write_text(str(p.pid)); time.sleep(60)")
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
        self.addCleanup(lambda: (unrelated.kill(), unrelated.wait()) if unrelated.poll() is None else None)
        result = self.ci.supervise([sys.executable, '-c', source], self.root/'missing.json',
                                   self.root/'batch', timeout=1)
        self.assertEqual(result['status'], 'TIMEOUT')
        self.assertIsNone(unrelated.poll())
        pid = int(child_pid_file.read_text())
        stat = Path(f'/proc/{pid}/stat')
        self.assertTrue(not stat.exists() or stat.read_text().split()[2] == 'Z')
        self.assertEqual(json.loads((self.root/'batch/run.json').read_text())['status'], 'TIMEOUT')

    def test_cancellation_retains_evidence_and_cleans_up(self):
        artifacts = self.root/'cancelled'
        command = [sys.executable, '-B', str(ROOT/'scripts/run_ci_checks.py'), '--profile','core',
                   '--artifacts', str(artifacts)]
        # Signal only the runner, not this test's process group.
        environment = os.environ.copy()
        environment.pop('GITHUB_STEP_SUMMARY', None)  # Controlled cancellation is not a real CI job.
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic()+10
        while time.monotonic()<deadline and not list(artifacts.glob('batches/*/run.json')):
            if process.poll() is not None:
                break
            time.sleep(.02)
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=8)
        self.assertNotEqual(process.returncode, 0)
        summary = json.loads((artifacts/'summary.json').read_text())
        self.assertEqual(summary['status'], 'CANCELLED')

    def test_pytest_counts_parameterized_cases(self):
        batch, report = self.worker('import pytest\n@pytest.mark.parametrize("value",[1,2])\ndef test_value(value): assert value>0\n', kind='pytest')
        self.assertEqual(batch['status'], 'PASS')
        self.assertEqual(report['counts']['tests_run'], 2)

    def test_profile_audit_rejects_visual_package_in_core(self):
        with self.assertRaises(ValueError):
            self.ci.audit_packages('core', {'pygame':'2.6.1'})

    def test_invalid_profile_is_an_argument_error(self):
        result = subprocess.run([sys.executable, str(ROOT/'scripts/run_ci_checks.py'),
                                 '--profile','not-a-profile','--artifacts',str(self.root/'out')],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 2)


if __name__ == '__main__':
    unittest.main()
