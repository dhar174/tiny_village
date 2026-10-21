"""One approved publication of the immutable #734 proposals; default is read-only.

Run from repository root. Uses the existing gh credential without reading secrets.
The approval and proposal hashes are pinned; receipts live outside the frozen bundle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BUNDLE = ROOT / 'docs/planning/issue-734'
REPO = 'repos/dhar174/tiny_village'
APPROVAL = 'https://github.com/dhar174/tiny_village/blob/eb3f1e2/docs/decisions.md#d027--approve-and-publish-the-734-backlog-triage-package'
START = '<!-- issue-734-triage:v1:start -->'
END = '<!-- issue-734-triage:v1:end -->'
COMPLETED = {355, 482, 594, 596, 597}
DUPLICATES = {351: 350, 382: 380}
FIELDS = '''number id fullDatabaseId title body state stateReason updatedAt
assignees(first:100){nodes{login} pageInfo{hasNextPage}}
parent{number} subIssues(first:100){totalCount nodes{number} pageInfo{hasNextPage}}
blockedBy(first:100){nodes{number} pageInfo{hasNextPage}}'''


def save(name, value):
    path = OUT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def api(path, payload=None, method=None):
    command = ['gh', 'api', path]
    if method:
        command += ['--method', method]
    if payload is not None:
        command += ['--input', '-']
    result = subprocess.run(command, input=json.dumps(payload) if payload is not None else None,
                            text=True, capture_output=True, timeout=60, cwd=ROOT)
    if result.returncode:
        raise RuntimeError(f'{path}: {result.stderr.strip()}')
    response = json.loads(result.stdout) if result.stdout.strip() else None
    if isinstance(response, dict) and response.get('errors'):
        raise RuntimeError(json.dumps(response['errors']))
    return response


def read(numbers):
    fields = ' '.join(f'i{n}:issue(number:{n}){{{FIELDS}}}' for n in numbers)
    result = api('graphql', {'query': '{repository(owner:"dhar174",name:"tiny_village"){' + fields + '}}'})
    issues = result['data']['repository']
    for n in numbers:
        issue = issues[f'i{n}']
        if issue is None:
            raise RuntimeError(f'Missing issue #{n}')
        for connection in ('assignees', 'blockedBy'):
            if issue[connection]['pageInfo']['hasNextPage']:
                raise RuntimeError(f'Incomplete {connection} #{n}')
    return {n: issues[f'i{n}'] for n in numbers}


def all_issues():
    issues, cursor = {}, None
    while True:
        query = '''query($cursor:String){repository(owner:"dhar174",name:"tiny_village"){
        issues(first:100,after:$cursor,orderBy:{field:CREATED_AT,direction:ASC}){
        nodes{''' + FIELDS + '''} pageInfo{hasNextPage endCursor}}}}'''
        page = api('graphql', {'query': query, 'variables': {'cursor': cursor}})['data']['repository']['issues']
        for i in page['nodes']:
            assert not i['blockedBy']['pageInfo']['hasNextPage']
            assert not i['assignees']['pageInfo']['hasNextPage']
            issues[i['number']] = i
        if not page['pageInfo']['hasNextPage']:
            return issues
        cursor = page['pageInfo']['endCursor']


def no_cycles(edges):
    visiting, done = set(), set()
    def visit(n):
        if n in visiting:
            raise RuntimeError(f'Relationship cycle at #{n}')
        if n in done:
            return
        visiting.add(n)
        for target in edges.get(n, []):
            visit(target)
        visiting.remove(n)
        done.add(n)
    for n in list(edges):
        visit(n)


def run(apply):
    subprocess.run(['python', str(BUNDLE / 'validate_bundle.py')], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    proposals = json.loads((BUNDLE / 'proposed-issue-updates.json').read_text())['updates']
    followups = json.loads((BUNDLE / 'proposed-followups.json').read_text())['items']
    assert len(proposals) == 307 and len(followups) == 3
    assert {p['number'] for p in proposals if p['state_operation'] == 'CLOSED:COMPLETED'} == COMPLETED
    assert {p['number'] for p in proposals if 'duplicate' in p['state_operation']} == set(DUPLICATES)
    checkpoint = OUT / 'checkpoint.json'
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {
        'approval': APPROVAL, 'merged_review': '2c3bc5753b15aad4a1e345c020c5872d6ba8b25c',
        'proposal_sha256': digest((BUNDLE / 'proposed-issue-updates.json').read_text()),
        'followup_sha256': digest((BUNDLE / 'proposed-followups.json').read_text()),
        'annotations': {}, 'parents': {}, 'closures': {}, 'followups': {}, 'edges': {}}
    assert state['proposal_sha256'] == digest((BUNDLE / 'proposed-issue-updates.json').read_text())
    assert state['followup_sha256'] == digest((BUNDLE / 'proposed-followups.json').read_text())
    current = all_issues()
    if not (OUT / 'before.json').exists():
        save('before.json', current)
    drift = []
    parents = {n: [i['parent']['number']] if i['parent'] else [] for n, i in current.items()}
    for p in proposals:
        n = p['number']; i = current[n]
        if str(n) not in state['annotations'] and START not in i['body'] and digest(i['body']) != p['base_body_sha256']:
            drift.append({'number': n, 'kind': 'body', 'current_sha256': digest(i['body'])})
        op = p['parent_operation']
        if isinstance(op, dict):
            target = op['add_if_unparented']
            if i['parent'] and i['parent']['number'] != target:
                drift.append({'number': n, 'kind': 'parent', 'current_parent': i['parent']['number']})
            parents[n] = [i['parent']['number']] if i['parent'] else [target]
    no_cycles(parents)
    no_cycles({n: [x['number'] for x in i['blockedBy']['nodes']] for n, i in current.items()})
    save('preflight.json', {'issues_read': len(current), 'approved_annotations': 307,
         'parent_additions': 80, 'drift': drift, 'approval': APPROVAL, 'apply_requested': apply})
    print(f'Preflight: {len(current)} issues, {len(drift)} material drifts', flush=True)
    if drift:
        raise RuntimeError('Material drift retained in preflight.json; inspect before publication')
    if not apply:
        return
    save('checkpoint.json', state)
    # Each batch rereads bodies immediately before its ordered GraphQL writes.
    # updateIssue changes only body; state/assignment/relations remain untouched.
    for offset in range(0, len(proposals), 5):
        batch = proposals[offset:offset + 5]
        live = read([p['number'] for p in batch])
        variables, definitions, mutations, expected = {}, [], [], {}
        for p in batch:
            n = p['number']; i = live[n]; body = i['body']
            annotation = p['proposed_annotation'].replace(END, f'- Publication approved under [D027]({APPROVAL}); preserved source evidence remains scoped as above.\n\n' + END)
            desired = annotation + '\n\n' + body
            if START in body:
                if body.count(START) != 1 or body.count(END) != 1 or not body.startswith(annotation + '\n\n'):
                    raise RuntimeError(f'Unexpected existing annotation #{n}')
                desired = body
            elif digest(body) != p['base_body_sha256']:
                raise RuntimeError(f'Body changed since preflight #{n}')
            if str(n) in state['annotations']:
                assert digest(body) == state['annotations'][str(n)]['after_sha256']
            if desired == body:
                state['annotations'].setdefault(str(n), {'after_sha256': digest(body), 'recovered': True})
                continue
            expected[n] = (i, desired)
            definitions.append(f'$v{n}:UpdateIssueInput!')
            variables[f'v{n}'] = {'id': i['id'], 'body': desired}
            mutations.append(f'i{n}:updateIssue(input:$v{n}){{issue{{{FIELDS}}}}}')
        if mutations:
            save('pending-batch.json', {str(n): {'before': i, 'desired_body': b} for n, (i,b) in expected.items()})
            result = api('graphql', {'query': 'mutation(' + ','.join(definitions) + '){' + ' '.join(mutations) + '}', 'variables': variables})['data']
            for n, (before, desired) in expected.items():
                after = result[f'i{n}']['issue']
                for key in ('state', 'stateReason', 'assignees', 'parent', 'blockedBy'):
                    assert after[key] == before[key], (n, key)
                assert after['body'] == desired
                state['annotations'][str(n)] = {'before_sha256': digest(before['body']), 'after_sha256': digest(desired),
                                               'updatedAt': after['updatedAt']}
            save('checkpoint.json', state)
            time.sleep(1)
        if offset % 25 == 0 or offset + 5 >= len(proposals):
            print(f'Annotations verified: {len(state["annotations"])}/307', flush=True)
    for p in proposals:
        if not isinstance(p['parent_operation'], dict):
            continue
        n, target = p['number'], p['parent_operation']['add_if_unparented']
        live = read([n, target]); child = live[n]
        if child['parent']:
            assert child['parent']['number'] == target, n
        else:
            assert live[target]['subIssues']['totalCount'] < 100
            api(f'{REPO}/issues/{target}/sub_issues', {'sub_issue_id': int(child['fullDatabaseId'])}, 'POST')
            assert read([n])[n]['parent']['number'] == target
        state['parents'][str(n)] = target
        save('checkpoint.json', state)
        if len(state['parents']) % 10 == 0:
            print(f'Parent links verified: {len(state["parents"])}/80', flush=True)
    # Reconcile created issues by marker as well as checkpoint to prevent duplicates
    # following an interrupted create response.
    current = all_issues()
    for item in followups:
        key = item['key']; marker = f'<!-- issue-734-followup:{key} -->'
        found = [i for i in current.values() if marker in i['body']]
        assert len(found) <= 1
        if found:
            new = found[0]
        else:
            body = marker + '\n' + item['body'] + f'\n\nPublication approved under [D027]({APPROVAL}). The ordering above is now approved for scheduling; the identity/interface design and implementation remain separate work. Gate #738 remains pending.\n'
            result = api(f'{REPO}/issues', {'title': item['title'], 'body': body}, 'POST')
            new = read([result['number']])[result['number']]
            current[new['number']] = new
        n = new['number']
        state['followups'][key] = {'number': n, 'url': f'https://github.com/dhar174/tiny_village/issues/{n}'}
        save('checkpoint.json', state)
        live = read([n, item['parent']])
        if not live[n]['parent']:
            assert live[item['parent']]['subIssues']['totalCount'] < 100
            api(f'{REPO}/issues/{item["parent"]}/sub_issues', {'sub_issue_id': int(new['fullDatabaseId'])}, 'POST')
        assert read([n])[n]['parent']['number'] == item['parent']
        pairs = [(n,b) for b in item['blocked_by']] + [(b,n) for b in item['blocks']]
        for dependent, blocker in pairs:
            current = all_issues()
            edges = {v: [x['number'] for x in i['blockedBy']['nodes']] for v, i in current.items()}
            if blocker not in edges[dependent]:
                edges[dependent].append(blocker); no_cycles(edges)
                api(f'{REPO}/issues/{dependent}/dependencies/blocked_by',
                    {'issue_id': int(current[blocker]['fullDatabaseId'])}, 'POST')
            assert blocker in [x['number'] for x in read([dependent])[dependent]['blockedBy']['nodes']]
            state['edges'][f'{dependent}<-{blocker}'] = 'verified'
            save('checkpoint.json', state)
        print(f'{key} created/verified: #{n}', flush=True)
    for n in sorted(COMPLETED | set(DUPLICATES)):
        reason = 'completed' if n in COMPLETED else 'not_planned'
        live = api(f'{REPO}/issues/{n}')
        if live['state'] != 'closed' or live['state_reason'] != reason:
            api(f'{REPO}/issues/{n}', {'state': 'closed', 'state_reason': reason}, 'PATCH')
        after = api(f'{REPO}/issues/{n}')
        assert after['state'] == 'closed' and after['state_reason'] == reason
        state['closures'][str(n)] = reason
        save('checkpoint.json', state)
    after = all_issues(); before = json.loads((OUT / 'before.json').read_text())
    for p in proposals:
        n = p['number']; now = after[n]; old = before[str(n)]
        assert digest(now['body']) == state['annotations'][str(n)]['after_sha256'], n
        assert now['body'].endswith('\n\n' + old['body']), n
        assert now['assignees'] == old['assignees'], n
        expected_parent = state['parents'].get(str(n))
        assert now['parent'] == ({'number': expected_parent} if expected_parent else old['parent']), n
        # Existing dependencies must all survive; only approved remediation edges are added.
        expected_edges = {x['number'] for x in old['blockedBy']['nodes']}
        expected_edges.update(int(k.split('<-')[1]) for k in state['edges'] if k.startswith(f'{n}<-'))
        assert {x['number'] for x in now['blockedBy']['nodes']} == expected_edges, n
        if n not in COMPLETED | set(DUPLICATES):
            assert (now['state'], now['stateReason']) == (old['state'], old['stateReason']), n
    for n in (602,609,733,738):
        assert after[n]['state'] == 'OPEN'
    no_cycles({n: [i['parent']['number']] if i['parent'] else [] for n,i in after.items()})
    no_cycles({n: [x['number'] for x in i['blockedBy']['nodes']] for n,i in after.items()})
    save('after.json', after)
    save('verification.json', {'status': 'publication_verified', 'annotations': len(state['annotations']),
         'parent_links': len(state['parents']), 'closures': state['closures'], 'followups': state['followups'],
         'remediation_dependency_edges': len(state['edges']), 'original_bodies_preserved': True,
         'assignees_preserved': True, 'existing_parents_preserved': True, 'existing_dependencies_preserved': True,
         'relationship_cycles': 0, 'phase_gate_approvals': 0, 'project_mutations': 0,
         'issue_734_closure': 'pending final completion comment and readback', 'approval': APPROVAL})
    print('Publication independently verified; #734 completion comment/closure remains.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    run(args.apply)
