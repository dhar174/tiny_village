"""Recompute inspection evidence; receipts are records, not attestations or approval."""
import copy
import datetime as dt

from sync_project import canonical, digest, expected_readback, operation_value, prepare, replace_expected


PROJECT_URL = "https://github.com/users/dhar174/projects/10"


def timestamp(value):
    try:
        parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError('Receipt/closure timestamp is invalid') from error
    if parsed.utcoffset() is None:
        raise ValueError('Receipt/closure timestamp requires a timezone')
    return parsed


def validate_plan(desired, plan):
    if desired.get("project_url") != PROJECT_URL:
        raise ValueError("Inspection evidence must target TinyVillage Project 10")
    if plan.get('desired_digest') != digest(desired) or plan.get('source_guard') != desired['source_guard']:
        raise ValueError('Inspection plan does not match desired digest/source guard')
    baseline = plan['baseline']
    if plan.get('project_url') != desired['project_url'] or baseline['url'] != desired['project_url']:
        raise ValueError('Inspection plan targets a different project')
    if prepare(desired, baseline) != plan:
        raise ValueError('Inspection plan operations differ from the retained baseline/desired plan')


def approved_result_matches(plan, project):
    """Account for assigned IDs/defaults on new rows; preserve everything else."""
    expected = copy.deepcopy(plan['baseline'])
    for operation in plan['operations']:
        actual = operation_value(project, operation)
        if actual != expected_readback(operation, actual):
            return False
        replace_expected(expected, operation, actual)
    return canonical(expected) == canonical(project)


def critical_readiness_matches(desired, project):
    current = {i['number']: i for i in project['items'] if i.get('repository') == 'dhar174/tiny_village'}
    for row in desired['rows']:
        if row['number'] in (737, 738, 806):
            actual = current.get(row['number'], {}).get('values', {})
            if any(actual.get(name) != row['values'].get(name) for name in ('Readiness', 'Decision')):
                return False
    return True


def inspect_publication(desired, plan, service):
    validate_plan(desired, plan)
    before = service.sources(desired)
    project = service.project()
    after = service.sources(desired)
    delta = prepare(desired, project)
    sources_match = before == after == desired['source_guard']
    approved_match = approved_result_matches(plan, project) and critical_readiness_matches(desired, project)
    verified = sources_match and not delta['operations'] and approved_match
    return {'schema_version': 1, 'status': 'verified' if verified else 'incomplete',
            'observed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'project_id': project['id'], 'project_url': project['url'],
            'desired_digest': digest(desired), 'plan_digest': digest(plan),
            'project_digest': digest(canonical(project)),
            'source_matches': sources_match, 'approved_result_matches': approved_match,
            'remaining_operations': len(delta['operations']),
            'sources_before': before, 'sources_after': after, 'project': project,
            'github_mutations_performed': 0}


def verify_receipt(receipt, desired, plan):
    """Verify recorded observations offline; humans still establish their provenance."""
    validate_plan(desired, plan)
    timestamp(receipt.get('observed_at'))
    if receipt.get('schema_version') != 1 or receipt.get('status') != 'verified':
        raise ValueError('A verified inspection receipt is required')
    if receipt.get('desired_digest') != digest(desired) or receipt.get('plan_digest') != digest(plan):
        raise ValueError('Receipt desired/plan digest mismatch')
    project = receipt['project']
    if (receipt.get('project_id') != project['id'] or project['id'] != plan['baseline']['id']
            or receipt.get('project_url') != project['url'] or project['url'] != desired['project_url']):
        raise ValueError('Receipt project identity mismatch')
    if receipt.get('project_digest') != digest(canonical(project)):
        raise ValueError('Receipt project digest mismatch')
    if (receipt.get('sources_before') != desired['source_guard']
            or receipt.get('sources_after') != desired['source_guard'] or receipt.get('source_matches') is not True):
        raise ValueError('Receipt source guard changed before/during inspection')
    if (prepare(desired, project)['operations'] or receipt.get('remaining_operations') != 0
            or not approved_result_matches(plan, project) or not critical_readiness_matches(desired, project)
            or receipt.get('approved_result_matches') is not True
            or receipt.get('github_mutations_performed') != 0):
        raise ValueError('Receipt does not demonstrate complete, preserved publication')
    return receipt


def verify_prior_publication(evidence, scope, managed_numbers):
    if not evidence or set(evidence) != {'desired', 'plan', 'receipt'}:
        raise ValueError('Prior publication evidence (desired, plan, receipt) is required')
    desired, plan, receipt = (evidence[k] for k in ('desired', 'plan', 'receipt'))
    verify_receipt(receipt, desired, plan)
    if (scope.get('project_url') != PROJECT_URL or not scope.get('project_id')
            or scope['project_id'] != receipt['project_id']):
        raise ValueError('Prior publication project identity differs from the reviewed Project 10 identity')
    rows = {r['number']: r for r in desired['rows']}
    if len(rows) != len(desired['rows']) or set(rows) != set(managed_numbers) or len(rows) != scope['managed_issues']:
        raise ValueError('Prior publication receipt must cover the same managed cohort')
    expected_prs = {n: {'head': h, 'state': 'OPEN'} for n, h in scope['candidate_heads'].items()}
    if desired['source_guard']['prs'] != expected_prs:
        raise ValueError('Prior publication candidate cohort/heads differ')
    for n in (737, 738, 806):
        if rows[n].get('issue_state') != 'OPEN':
            raise ValueError('Prior publication must precede issue completion')
    if (rows[737]['values']['Readiness'] != 'Blocked' or rows[738]['values']['Readiness'] != 'Blocked'
            or rows[738]['values']['Decision'] != 'Pending'):
        raise ValueError('Prior publication must retain the Pending/Blocked human gate')
    return timestamp(receipt['observed_at'])
