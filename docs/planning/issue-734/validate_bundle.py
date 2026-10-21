"""Validate frozen triage data; never changes GitHub or the simulation."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

def load(name):
    return json.loads((HERE / name).read_text())

def assert_acyclic(graph):
    active, done = set(), set()
    def visit(node):
        assert node not in active, f'Cycle at {node}'
        if node in done:
            return
        active.add(node)
        for target in graph.get(node, []):
            visit(target)
        active.remove(node)
        done.add(node)
    for node in graph:
        visit(node)

manifest = load('snapshot-manifest.json')
issues = load('issues-snapshot.json')['issues']
by = {item['number']: item for item in issues}
rows = load('ledger.json')
by_row = {row['number']: row for row in rows}
assert len(by) == len(issues) == manifest['issue_count'] == 475
assert sum(i['state'] == 'OPEN' for i in issues) == manifest['open_issues'] == 306
assert all(not p['pageInfo']['hasNextPage'] for p in [load('issues-snapshot.json')['pages'][-1]])
cohort = set()
for issue in issues:
    assert not any(issue[field]['pageInfo']['hasNextPage'] for field in ['comments', 'timelineItems', 'blockedBy', 'assignees', 'labels'])
    if issue['createdAt'] >= manifest['historical_cutoff']:
        continue
    state = 'OPEN'
    for event in sorted(issue['timelineItems']['nodes'], key=lambda e: e['createdAt']):
        if event['createdAt'] >= manifest['historical_cutoff']:
            break
        state = 'CLOSED' if event['__typename'] == 'ClosedEvent' else 'OPEN'
    if state == 'OPEN':
        cohort.add(issue['number'])
assert len(cohort) == 261 and cohort == set(manifest['historical_open_issues'])
assert len(rows) == len(by_row) == 307
assert set(by_row) == cohort | {i['number'] for i in issues if i['state'] == 'OPEN'}
assert {r['number'] for r in rows if r['cohort'] == 'original_261'} == cohort
assert sum(r['cohort'] == 'current_only_open' for r in rows) == 46
for name, expected in manifest['hashes'].items():
    assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected
for row in rows:
    issue = by[row['number']]
    for field in ['accountable_owner', 'canonical_topic', 'acceptance', 'reproduction', 'rationale', 'verification', 'readiness']:
        assert row[field], (row['number'], field)
    assert row['disposition'] in ['implement', 'verify', 'duplicate', 'defer']
    assert row['canonical_topic_issue'] in by
    assert row['actual_assignees'] == [a['login'] for a in issue['assignees']['nodes']]
    assert row['native_parent'] == ((issue['parent'] or {}).get('number'))
    assert row['native_prerequisites'] == [b['number'] for b in issue['blockedBy']['nodes']]
    assert row['snapshot_body_sha256'] == hashlib.sha256(issue['body'].encode()).hexdigest()
    assert row['snapshot_updated_at'] == issue['updatedAt']
    assert all(n in by and n != row['number'] for n in row['proposed_execution_prerequisites'])
    if row['disposition'] == 'defer':
        assert row['deferral_condition']
    if row['disposition'] == 'duplicate':
        target = row['duplicate_of']
        assert target in by_row and target != row['number']
        assert issue['body'] == by[target]['body'], 'Exact duplicates must retain exact body evidence'
    if row['proposed_parent']:
        assert row['native_parent'] is None and row['proposed_parent'] in by
    for anchor in row['source_anchors']:
        path, number = anchor.rsplit(':', 1)
        assert path in load('source-anchors.json') and int(number) > 0
parents = {i['number']: [i['parent']['number']] if i['parent'] else [] for i in issues}
for row in rows:
    if row['proposed_parent']:
        parents[row['number']] = [row['proposed_parent']]
assert_acyclic(parents)
dependencies = {i['number']: [b['number'] for b in i['blockedBy']['nodes']] for i in issues}
assert_acyclic(dependencies)
for row in rows:
    dependencies[row['number']] += row['proposed_execution_prerequisites']
assert_acyclic(dependencies)
proposals = load('proposed-issue-updates.json')
assert proposals['status'] == 'PROPOSED_NOT_APPROVED_NOT_APPLIED' and proposals['mutations_performed'] == 0
assert {p['number'] for p in proposals['updates']} == set(by_row)
for proposal in proposals['updates']:
    row = by_row[proposal['number']]
    assert proposal['requires_review'] and proposal['assignees_operation'] == 'preserve'
    assert proposal['base_body_sha256'] == row['snapshot_body_sha256']
    assert proposal['state_operation'] == row['proposed_state']
    assert proposal['proposed_annotation'].count('<!-- issue-734-triage:v1:start -->') == 1
    assert proposal['proposed_annotation'].count('<!-- issue-734-triage:v1:end -->') == 1
    assert 'replace_parent' not in json.dumps(proposal)
assert {r['number'] for r in rows if r['duplicate_of']} == {351, 382}
assert {r['number'] for r in rows if r['proposed_state'] == 'CLOSED:COMPLETED'} == {355, 482, 594, 596, 597}
assert all(by_row[n]['proposed_state'] == 'preserve' for n in [602, 609, 734, 733, 738, 654])
assert by_row[654]['state'] == 'CLOSED'
assert all(by_row[n]['verification'] == 'failed_real_object_probe' for n in [602, 609])
assert not (set(range(738, 773)) & {r['number'] for r in rows if r['proposed_state'].startswith('CLOSED')})
followups = load('proposed-followups.json')
assert followups['status'] == 'PROPOSED_NOT_CREATED' and len(followups['items']) == 3
for f in followups['items']:
    assert f['key'] not in dependencies and f['parent'] in by
    dependencies[f['key']] = f['blocked_by']
    for n in f['blocks']:
        dependencies[n].append(f['key'])
assert_acyclic(dependencies)
prs = load('prs-snapshot.json')['pull_requests']
handoffs = load('pr-handoffs.json')
assert len(prs) == len(handoffs) == manifest['open_prs'] == 37
assert manifest['pr_nested_overflow'] == []
assert {p['number']: p['head_sha'] for p in handoffs} == {p['number']: p['headRefOid'] for p in prs}
assert all(p['handoff_issue'] in by for p in handoffs)
with (HERE / 'ledger.csv').open(newline='') as stream:
    assert [int(r['number']) for r in csv.DictReader(stream)] == [r['number'] for r in rows]
for path, source in load('source-anchors.json').items():
    raw = subprocess.check_output(['git', 'show', f'{manifest["source_sha"]}:{path}'], cwd=REPO)
    assert hashlib.sha256(raw).hexdigest() == source['sha256']
results = load('evidence/audit-results.json')
assert len(results) == 12
for r in results:
    capture = HERE / 'evidence/captures' / Path(r['capture']['artifact_dir']).name
    metadata = json.loads((capture / 'run.json').read_text())
    assert metadata['returncode'] == r['capture']['returncode']
    assert metadata['outcome'] not in ['STARTING', 'STARTED', 'TIMEOUT']
    assert metadata['fatal_timeout'] == 20 and metadata['timeout'] == 25
    assert all((capture / name).is_file() for name in ['stdout.log', 'stderr.log', 'events.jsonl'])
    assert r['parent_returncode'] == 0
for name in ['weather', 'queue', 'live_timing', 'frame_analysis', 'existing_regressions']:
    assert [r for r in results if r['name'] == name][-1]['capture']['returncode'] == 0
for name in ['named_target', 'social', 'visual_interfaces']:
    assert [r for r in results if r['name'] == name][-1]['capture']['returncode'] == 1
final_test = [r for r in results if r['name'] == 'existing_regressions'][-1]
output = (HERE / 'evidence/captures' / Path(final_test['capture']['artifact_dir']).name / 'stdout.log').read_text()
assert '12 passed, 2 warnings' in output
# Catalog integrity passing is deliberately independent of the retained runtime
# failures. This validator is not a simulation acceptance test.
summary = {'status': 'catalog_integrity_passed', 'original_issues': len(cohort), 'current_only_open_issues': 46,
           'ledger_rows': len(rows), 'open_pr_handoffs': len(handoffs), 'parent_dependency_cycles': 0,
           'proposed_parent_additions': sum(r['proposed_parent'] is not None for r in rows),
           'proposed_duplicate_closures': [351, 382], 'proposed_completed_closures': [355, 482, 594, 596, 597],
           'retained_capture_attempts': len(results), 'capture_nonzero_exits': sum(r['capture']['returncode'] != 0 for r in results),
           'existing_focused_tests_passed': 12, 'runtime_failures_preserved': ['named_target', 'social', 'visual_interfaces'],
           'github_issue_mutations_performed': 0, 'phase_gate_approvals_supplied': 0}
print(json.dumps(summary, indent=2))
