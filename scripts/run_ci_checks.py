"""Bounded, explicit supported-profile checks; standard library in the parent."""

import argparse
from datetime import datetime, timezone
import faulthandler
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
# Executing a script by path otherwise places scripts/, not the repository, first.
sys.path.insert(0, str(ROOT))
COUNT_KEYS = ('tests_run', 'failures', 'errors', 'skipped', 'expected_failures', 'unexpected_successes')
LIGHTWEIGHT_EXTRAS = {'pygame', 'faiss-cpu', 'spacy', 'nltk', 'torch', 'transformers',
                      'sentence-transformers', 'pandas', 'scipy', 'scikit-learn',
                      'rake-nltk', 'textblob', 'matplotlib', 'seaborn'}
PROFILES = {
    'core': [('components', 'unittest', ['tests.test_actions', 'tests.test_action_named_target_resolution',
              'tests.test_goap_error_handling', 'tests.test_event_handler_payloads']),
             ('runtime', 'unittest', ['tests.test_memory_runtime', 'tests.test_lightweight_runtime.TestHeadlessRuntime',
              'tests.test_lightweight_runtime.TestMinimalRunner']),
             ('canonical-cli', 'probe', ['canonical']), ('default-cli', 'probe', ['default']),
             ('invalid-cli', 'probe', ['invalid'])],
    'dev': [('runner', 'unittest', ['tests.test_ci_checks']),
            ('cli-smoke', 'pytest', ['tests/test_minimal_demo_smoke.py'])],
    'visual': [('lifecycle', 'unittest', ['tests.test_lightweight_runtime.TestVisualLifecycle']),
               ('graph', 'unittest', ['tests.test_graph_initialization', 'tests.test_graph_ownership']),
               ('cache', 'unittest', ['tests.test_model_cache']),
               ('capture', 'unittest', ['tests.test_subprocess_capture'])],
    'nlp': [('nlp', 'unittest', ['tests.test_memory_optional_profiles.TestInstalledNLP'])],
    'llm-cpu': [('retrieval', 'unittest', ['tests.test_memory_optional_profiles.TestInstalledRetrieval']),
                ('installed-basic', 'unittest', ['tests.test_lightweight_runtime.TestMinimalRunner.test_demo_returns_real_registered_state_and_memories'])],
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def counts(result):
    return dict(zip(COUNT_KEYS, [result.testsRun, len(result.failures), len(result.errors),
                                len(result.skipped), len(result.expectedFailures), len(result.unexpectedSuccesses)]))


def successful_report(report):
    if not isinstance(report, dict) or report.get('status') != 'PASS':
        return False
    if report.get('kind') == 'probe':
        expected, observed = report.get('expected_exit_code'), report.get('observed_exit_code')
        return (type(report.get('probes_run')) is int and report['probes_run'] == 1 and
                type(expected) is int and type(observed) is int and
                0 <= expected <= 255 and observed == expected)
    values = report.get('counts', {})
    return (report.get('kind') in ('unittest', 'pytest') and
            all(isinstance(values.get(key), int) and values[key] >= 0 for key in COUNT_KEYS) and
            values['tests_run'] > 0 and all(values[key] == 0 for key in COUNT_KEYS[1:]))


def worker_unittest(selectors, report):
    state = {'kind':'unittest', 'status':'RUNNING', 'counts':dict.fromkeys(COUNT_KEYS, 0)}
    write_json(report, state)

    class RecordingResult(unittest.TextTestResult):
        def stopTest(self, test):
            super().stopTest(test)
            state['counts'] = counts(self)
            write_json(report, state)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for selector in selectors:
        if selector.endswith('.py'):
            suite.addTests(loader.discover(str(Path(selector).parent), pattern=Path(selector).name))
        else:
            suite.addTests(loader.loadTestsFromName(selector))
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
    state.update(status='PASS', counts=counts(result))
    if not successful_report(state):
        state['status'] = 'FAIL'
    write_json(report, state)
    return 0 if state['status'] == 'PASS' else 1


def worker_pytest(selectors, report):
    import pytest
    junit = Path(report).with_suffix('.xml')
    write_json(report, {'status':'RUNNING', 'kind':'pytest'})
    outcomes = {'expected_failures':0, 'unexpected_successes':0}

    class Outcomes:
        def pytest_runtest_logreport(self, report):
            if hasattr(report, 'wasxfail'):
                if report.skipped:
                    outcomes['expected_failures'] += 1
                elif report.passed:
                    outcomes['unexpected_successes'] += 1

    exit_code = int(pytest.main(['-q', *selectors, '--junitxml='+str(junit)], plugins=[Outcomes()]))
    cases = list(ET.parse(junit).getroot().iter('testcase')) if junit.exists() else []
    values = dict.fromkeys(COUNT_KEYS, 0)
    values.update(tests_run=len(cases), failures=sum(c.find('failure') is not None for c in cases),
                  errors=sum(c.find('error') is not None for c in cases),
                  skipped=sum(c.find('skipped') is not None for c in cases))
    values.update(outcomes)
    values['skipped'] -= outcomes['expected_failures']  # JUnit represents XFAIL as skipped.
    state = {'status':'PASS', 'kind':'pytest', 'counts':values, 'pytest_exit_code':exit_code}
    if exit_code != 0 or not successful_report(state):
        state['status'] = 'FAIL'
    write_json(report, state)
    return 0 if state['status'] == 'PASS' else 1


def worker_probe(selector, report):
    from tests.subprocess_capture import capture_python
    options = {'canonical':(['--mode','minimal','--headless','--no-llm','--characters','2'], 0),
               'default':(['--mode','minimal'], 0), 'invalid':(['--headless'], 2)}
    arguments, expected = options[selector]
    state = {'kind':'probe', 'status':'RUNNING', 'probes_run':0, 'expected_exit_code':expected}
    write_json(report, state)
    source = ('import runpy,sys\nsys.argv='+repr(['main.py', *arguments])+"\nrunpy.run_path('main.py',run_name='__main__')\n")
    captured = capture_python(source, cwd=ROOT, env=os.environ.copy(), test_id='ci-'+selector,
                              fatal_timeout=20, timeout=25)
    print(captured.completed.stdout, end='')
    print(captured.completed.stderr, end='', file=sys.stderr)
    state.update(status='PASS', probes_run=1, observed_exit_code=captured.completed.returncode,
                 capture_artifacts=str(captured.artifact_dir))
    if not successful_report(state):
        state['status'] = 'FAIL'
    write_json(report, state)
    return 0 if state['status'] == 'PASS' else 1


def process_identity(pid):
    """Linux process start time distinguishes a reused leader PID."""
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        return stat[stat.rfind(')')+2:].split()[19]
    except (OSError, IndexError):
        return None


def register_group(registry, parent_group):
    """Bootstrap before exec: no workload can spawn before ownership is durable."""
    registry = Path(registry)
    pid = os.getpid()
    write_json(registry/f'{pid}.json', {'pgid':pid, 'parent':parent_group, 'identity':process_identity(pid)})
    visited = set()
    ancestor = parent_group
    while ancestor not in visited:
        visited.add(ancestor)
        if (registry/f'{ancestor}.closed').exists():
            raise SystemExit('Owning batch already stopped; refusing late descendant startup')
        try:
            ancestor = json.loads((registry/f'{ancestor}.json').read_text())['parent']
        except FileNotFoundError:
            break


def cleanup_group(process, registry):
    """Stop registered descendant sessions too, including after hard worker exit."""
    registry = Path(registry)
    (registry/f'{process.pid}.closed').touch()
    signalled = set()
    deadline = time.monotonic()+2

    def owned_groups():
        records = {int(p.stem):json.loads(p.read_text()) for p in registry.glob('*.json')}
        owned = {process.pid}
        while True:
            extra = {pid for pid, row in records.items() if row['parent'] in owned}
            if extra <= owned:
                break
            owned |= extra
        # A live different leader means the old PID/group has been reused.
        return {pid for pid in owned if pid not in records or
                process_identity(pid) in (None, records[pid]['identity'])}

    while True:
        alive = False
        for pgid in owned_groups():
            (registry/f'{pgid}.closed').touch()
            try:
                os.killpg(pgid, 0)
                alive = True
                if pgid not in signalled:
                    os.killpg(pgid, signal.SIGTERM)
                    signalled.add(pgid)
            except ProcessLookupError:
                pass
        process.poll()
        if not alive:
            return sorted(signalled)
        if time.monotonic() >= deadline:
            break
        time.sleep(.02)  # Re-scan for bootstraps racing with owner termination.
    groups = owned_groups()
    for pgid in groups:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    # SIGKILL delivery is asynchronous. Wait for live members to stop; zombies
    # can remain until the host reaps them and do not represent running work.
    deadline = time.monotonic()+1
    while True:
        live = False
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                stat = path.read_text()
                fields = stat[stat.rfind(')')+2:].split()
                if int(fields[2]) in groups and fields[0] not in ('Z','X'):
                    live = True
                    break
            except (OSError, IndexError, ValueError):
                continue
        if not live:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('Owned process group did not stop after SIGKILL')
        time.sleep(.01)
    return sorted(signalled)


def supervise(command, report, directory, *, timeout=120):
    """Run one owned batch; process status and structured evidence must agree."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory/'run.json'
    if record_path.exists():
        raise ValueError(f'Refusing to overwrite batch evidence: {directory}')
    report = Path(report)
    report.unlink(missing_ok=True)
    record = {'status':'STARTING', 'command':command, 'parent_timeout':timeout,
              'started_utc':datetime.now(timezone.utc).isoformat(), 'exit_code':None}
    write_json(record_path, record)
    process = None
    registry = Path(os.environ.get('TINY_CI_GROUP_REGISTRY', directory/'groups'))
    registry.mkdir(mode=0o700, parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['TINY_CI_GROUP_REGISTRY'] = str(registry)
    bootstrap = ('import os; from scripts.run_ci_checks import register_group; '
                 f'register_group({str(registry)!r},{os.getpgrp()!r}); '
                 f'os.execvp({command[0]!r},{command!r})')
    record['launcher'] = [sys.executable, '-B', '-c', bootstrap]
    try:
        with (directory/'stdout.log').open('wb') as out, (directory/'stderr.log').open('wb') as err:
            process = subprocess.Popen(record['launcher'], cwd=ROOT, env=environment, stdout=out, stderr=err, start_new_session=True)
            record.update(status='RUNNING', pid=process.pid)
            write_json(record_path, record)
            try:
                process.wait(timeout=timeout)
                record['status'] = 'FINISHED'
            except subprocess.TimeoutExpired:
                record['status'] = 'TIMEOUT'
            except KeyboardInterrupt:
                record['status'] = 'CANCELLED'
            finally:
                record['terminated_groups'] = cleanup_group(process, registry)
                process.wait(timeout=3)
        record['exit_code'] = process.returncode
        try:
            record['report'] = json.loads(report.read_text())
        except (OSError, ValueError) as error:
            record['report_error'] = str(error)
        if record['status'] == 'FINISHED':
            record['status'] = 'PASS' if process.returncode == 0 and successful_report(record.get('report')) else 'FAIL'
    except Exception as error:
        record.update(status='FAIL', supervisor_error=f'{type(error).__name__}: {error}')
    finally:
        if process is not None and process.poll() is None:
            record['terminated_groups'] = cleanup_group(process, registry)
            process.wait(timeout=3)
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(record_path, record)
    return record


def audit_packages(profile, packages):
    if profile in ('core', 'dev'):
        extras = LIGHTWEIGHT_EXTRAS & {re.sub(r'[-_.]+', '-', name).lower() for name in packages}
        if extras:
            raise ValueError(f'{profile} must omit rendering/AI packages: {sorted(extras)}')
    needed = {'visual':{'pygame'}, 'nlp':{'spacy','nltk'}, 'llm-cpu':{'torch','transformers','faiss-cpu'}}.get(profile, set())
    missing = needed-packages.keys()
    if missing:
        raise ValueError(f'{profile} missing required packages: {sorted(missing)}')


def run_profile(profile, artifacts):
    artifacts = Path(artifacts).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    summary_path = artifacts/'summary.json'
    if summary_path.exists():
        raise ValueError('Choose a new artifact directory; earlier evidence must be retained')
    lock_names = ['dev','visual'] if profile == 'visual' else [profile]
    locks = {f'requirements/locks/{name}-py312-linux-x86_64.txt':
             hashlib.sha256((ROOT/f'requirements/locks/{name}-py312-linux-x86_64.txt').read_bytes()).hexdigest()
             for name in lock_names}
    summary = {'schema_version':1, 'status':'RUNNING', 'profile':profile, 'python':sys.version,
               'platform':platform.platform(), 'machine':platform.machine(),
               'revision':subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
               'locks':locks, 'batches':{name:{'status':'NOT_RUN'} for name, *_ in PROFILES[profile]}}
    write_json(summary_path, summary)
    try:
        if sys.version_info[:2] != (3,12) or sys.platform != 'linux' or platform.machine() != 'x86_64':
            raise ValueError('Supported target: CPython 3.12, Linux x86-64')
        packages = {re.sub(r'[-_.]+','-',d.metadata['Name']).lower():d.version for d in metadata.distributions()}
        summary['packages'] = packages
        audit_packages(profile, packages)
        cache = os.environ.get('HF_HOME', str(artifacts/'cache/huggingface'))
        os.environ.update(HF_HOME=cache, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                          SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy', PYGAME_HIDE_SUPPORT_PROMPT='1',
                          TINY_TEST_ARTIFACT_DIR=str(artifacts/'children'))
        for name, kind, selectors in PROFILES[profile]:
            directory = artifacts/'batches'/name
            report = directory/'result.json'
            command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker',kind,
                       '--report',str(report), *selectors]
            result = supervise(command, report, directory)
            summary['batches'][name] = result
            write_json(summary_path, summary)
            print(f'{profile}/{name}: {result["status"]}', flush=True)
            if result['status'] == 'CANCELLED':
                summary['status'] = 'CANCELLED'
                break
        else:
            summary['status'] = 'PASS' if all(b['status']=='PASS' for b in summary['batches'].values()) else 'FAIL'
    except KeyboardInterrupt:
        summary['status'] = 'CANCELLED'
    except Exception as error:
        summary.update(status='FAIL', error=f'{type(error).__name__}: {error}')
    finally:
        totals = dict.fromkeys(COUNT_KEYS, 0)
        probes = 0
        for batch in summary['batches'].values():
            result = batch.get('report', {})
            for key in totals:
                totals[key] += result.get('counts', {}).get(key, 0)
            probes += result.get('probes_run', 0)
        summary.update(counts=totals, probes_run=probes)
        write_json(summary_path, summary)
        print(json.dumps({'profile':profile,'status':summary['status'],'counts':totals,'probes_run':probes}), flush=True)
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
                stream.write(f'### {profile}: {summary["status"]}\n\n{totals["tests_run"]} tests; '
                             f'{totals["failures"]} failures; {totals["errors"]} errors; '
                             f'{totals["skipped"]} skips; {probes} CLI probes.\n')
    return 0 if summary['status']=='PASS' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES)
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--worker', choices=('unittest','pytest','probe'), help=argparse.SUPPRESS)
    parser.add_argument('--report', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('selectors', nargs='*', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if not args.report:
            parser.error('--worker requires --report')
        faulthandler.dump_traceback_later(115 if args.worker=='pytest' else 110, exit=True)
        if args.worker == 'unittest':
            return worker_unittest(args.selectors, args.report)
        if args.worker == 'pytest':
            return worker_pytest(args.selectors, args.report)
        return worker_probe(args.selectors[0], args.report)
    if not args.profile or not args.artifacts:
        parser.error('--profile and --artifacts are required')
    def cancelled(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, cancelled)
    try:
        return run_profile(args.profile, args.artifacts)
    except (OSError, ValueError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
