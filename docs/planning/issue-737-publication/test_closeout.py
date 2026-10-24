"""Closeout scope, stage and portability regressions; no GitHub writes."""
from pathlib import Path, PureWindowsPath
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_package as validator
import sync_project as sync
from sync_project import load

ROOT = Path(__file__).resolve().parent

class CloseoutTests(unittest.TestCase):
    def setUp(self):
        self.scope = load(ROOT / 'review-scope.json')
        self.snapshot = {'pull_requests': [{'number': int(n), 'headRefOid': h} for n,h in self.scope['candidate_heads'].items()],
                         'issues': [{'number': n, 'state': 'OPEN'} for n in (737,738,741,774,799,800,801,802,803,806,807,808,809)]}
        self.rows = {737: {'readiness':'Blocked', 'blocking_issues':[806]},
                     738: {'decision':'Pending','readiness':'Blocked'}, 806: {'readiness':'Ready'}}
    def test_reviewed_24_candidates_pass_publication_stage(self):
        validator.validate_closeout(self.snapshot, self.rows, self.scope)
    def test_same_count_replacement_is_not_the_reviewed_cohort(self):
        self.snapshot['pull_requests'][0]['number'] = 999
        with self.assertRaisesRegex(ValueError, 'cohort'):
            validator.validate_closeout(self.snapshot, self.rows, self.scope)
    def test_changed_head_requires_new_review(self):
        self.snapshot['pull_requests'][0]['headRefOid'] = 'new'
        with self.assertRaisesRegex(ValueError, 'cohort'):
            validator.validate_closeout(self.snapshot, self.rows, self.scope)
    def test_unreviewed_stage_cannot_claim_completion(self):
        self.scope['stage'] = 'complete'
        with self.assertRaisesRegex(ValueError, 'completion|stage'):
            validator.validate_closeout(self.snapshot, self.rows, self.scope)
    def test_closed_737_rejected_before_publication(self):
        self.snapshot['issues'][0]['state'] = 'CLOSED'
        with self.assertRaisesRegex(ValueError, '737'):
            validator.validate_closeout(self.snapshot, self.rows, self.scope)
    def test_closed_runtime_requirement_rejected(self):
        self.snapshot['issues'][4]['state'] = 'CLOSED'
        with self.assertRaisesRegex(ValueError, '799'):
            validator.validate_closeout(self.snapshot, self.rows, self.scope)
    def test_human_gate_not_unlocked(self):
        self.rows[738]['readiness'] = 'Human review'
        with self.assertRaisesRegex(ValueError, 'gate'):
            validator.validate_closeout(self.snapshot, self.rows, self.scope)
    def test_windows_manifest_uses_posix_paths(self):
        self.assertEqual(validator.manifest_name(PureWindowsPath('C:/bundle/evidence/a.json'), PureWindowsPath('C:/bundle')), 'evidence/a.json')
    def test_live_inventory_keeps_unreviewed_pr_and_excludes_closeout(self):
        # Exercise the real source collector's GraphQL normalization without network.
        service = sync.GitHub()
        response = {'repository': {'defaultBranchRef': {'target': {'oid':'main'}},
            'p709': {'number':709,'headRefOid':'head','state':'OPEN'},
            'openInventory': {'nodes': [
                {'number':811,'headRefName':'codex/issue-737-closeout','headRefOid':'own','state':'OPEN','headRepository':{'nameWithOwner':'dhar174/tiny_village'}},
                {'number':812,'headRefName':'codex/issue-737-unreviewed','headRefOid':'new','state':'OPEN'}],
                'pageInfo': {'hasNextPage': False}}}}
        with patch.object(service, 'gql', return_value=response):
            result = service.sources({'source_guard': {'issues':{},'prs':{'709':{}}}})
        self.assertNotIn('811', result['prs'])
        self.assertIn('812', result['prs'])
        self.assertIn('709', result['prs'])
    def test_foreign_fork_with_same_coordinator_branch_stays_in_inventory(self):
        service = sync.GitHub()
        response = {'repository': {'defaultBranchRef': {'target': {'oid':'main'}},
            'p709': {'number':709,'headRefOid':'head','state':'OPEN'},
            'openInventory': {'nodes': [
                {'number':813,'headRefName':'codex/issue-737-closeout',
                 'headRepository':{'nameWithOwner':'other/tiny_village'},
                 'headRefOid':'foreign','state':'OPEN'}], 'pageInfo': {'hasNextPage': False}}}}
        with patch.object(service, 'gql', return_value=response):
            result = service.sources({'source_guard': {'issues':{},'prs':{'709':{}}}})
        self.assertIn('813', result['prs'])
    def test_only_explicit_coordinator_branches_excluded(self):
        for branch in ('codex/issue-737-closeout','codex/issue-737-coordination','codex/issue-737-publication'):
            pr = {'headRefName':branch,'headRepository':{'nameWithOwner':'dhar174/tiny_village'}}
            self.assertTrue(sync.excluded_coordinator(pr))
        pr['headRefName'] = 'codex/issue-737-unreviewed'
        self.assertFalse(sync.excluded_coordinator(pr))
        pr['headRefName'] = 'codex/issue-737-closeout'
        pr['headRepository'] = None
        self.assertFalse(sync.excluded_coordinator(pr))

if __name__ == '__main__':
    unittest.main(verbosity=2)
