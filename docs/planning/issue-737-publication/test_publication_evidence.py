"""Receipt and completion contracts using the real publisher's planning logic."""
import copy
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sync_project as sync
import validate_package as validator
from test_sync_project import fixture, MemoryService


def example():
    desired, project = fixture()
    desired['project_url'] = project['url']
    service = MemoryService(desired, project)
    plan = sync.prepare(desired, project)
    sync.apply_plan(plan, desired, service, {}, lambda: None, sync.digest(plan))
    return desired, plan, service


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        import publication_evidence
        self.evidence = publication_evidence
        self.desired, self.plan, self.service = example()

    def receipt(self):
        return self.evidence.inspect_publication(self.desired, self.plan, self.service)

    def test_independent_inspection_is_read_only_and_recomputable(self):
        writes = len(self.service.writes)
        receipt = self.receipt()
        self.evidence.verify_receipt(receipt, self.desired, self.plan)
        self.assertEqual(receipt['status'], 'verified')
        self.assertEqual(receipt['remaining_operations'], 0)
        self.assertEqual(len(self.service.writes), writes)

    def test_wrong_plan_is_rejected(self):
        self.plan['desired_digest'] = 'other'
        with self.assertRaisesRegex(ValueError, 'plan|digest'):
            self.receipt()

    def test_edited_plan_operations_are_rejected_even_with_new_digest(self):
        self.plan['operations'] = []
        with self.assertRaisesRegex(ValueError, 'plan'):
            self.receipt()

    def test_incomplete_board_cannot_be_claimed_verified(self):
        self.service.state['items'][0]['values']['Sequence'] = 123
        receipt = self.receipt()
        self.assertEqual(receipt['status'], 'incomplete')
        receipt.update(status='verified', remaining_operations=0)
        with self.assertRaises(ValueError):
            self.evidence.verify_receipt(receipt, self.desired, self.plan)

    def test_managed_value_tampering_is_rejected(self):
        receipt = self.receipt()
        receipt['project']['items'][0]['values']['Sequence'] = 999
        receipt['project_digest'] = sync.digest(sync.canonical(receipt['project']))
        with self.assertRaises(ValueError):
            self.evidence.verify_receipt(receipt, self.desired, self.plan)

    def test_unrelated_item_edits_do_not_count_as_preservation(self):
        self.service.state['items'][1]['values']['Notes'] = 'changed'
        with self.assertRaises(ValueError):
            self.evidence.verify_receipt(self.receipt(), self.desired, self.plan)

    def test_view_sorting_change_is_rejected(self):
        self.service.state['views'][0]['sort_by'] = [['Sequence', 'DESC']]
        with self.assertRaises(ValueError):
            self.evidence.verify_receipt(self.receipt(), self.desired, self.plan)

    def test_wrong_project_identity_is_rejected(self):
        self.service.state['id'] = 'another-project'
        with self.assertRaises(ValueError):
            self.evidence.verify_receipt(self.receipt(), self.desired, self.plan)

    def test_stale_sources_are_rejected(self):
        self.service.source['main'] = 'new-main'
        with self.assertRaises(ValueError):
            self.evidence.verify_receipt(self.receipt(), self.desired, self.plan)

    def test_source_change_during_inspection_is_rejected(self):
        real_project = self.service.project
        def change_source():
            self.service.source['main'] = 'new-main'
            return real_project()
        with patch.object(self.service, 'project', side_effect=change_source):
            receipt = self.receipt()
        with self.assertRaises(ValueError):
            self.evidence.verify_receipt(receipt, self.desired, self.plan)

    def test_receipt_digest_binding_is_required(self):
        receipt = self.receipt()
        receipt['plan_digest'] = 'other'
        with self.assertRaises(ValueError):
            self.evidence.verify_receipt(receipt, self.desired, self.plan)

    def test_timestamp_requires_timezone(self):
        receipt = self.receipt()
        receipt['observed_at'] = '2026-10-23T01:00:00'
        with self.assertRaises(ValueError):
            self.evidence.verify_receipt(receipt, self.desired, self.plan)


class StageTests(unittest.TestCase):
    def setUp(self):
        import publication_evidence
        self.evidence_module = publication_evidence
        self.scope = sync.load(Path(__file__).resolve().parent / 'review-scope.json')
        self.scope['stage'] = 'completed'
        self.scope['project_id'] = 'P10'
        self.scope['managed_issues'] = 3
        self.snapshot = {'pull_requests':[{'number':int(n),'headRefOid':h} for n,h in self.scope['candidate_heads'].items()],
            'issues':[{'number':n,'state':'OPEN','stateReason':None} for n in (737,738,741,774,799,800,801,802,803,806,807,808,809)]}
        for issue in self.snapshot['issues']:
            if issue['number'] in (737,806):
                issue.update(state='CLOSED',stateReason='COMPLETED',closedAt='2030-01-01T00:00:00Z')
        self.rows = {737:{'readiness':'Tracking'},806:{'readiness':'Tracking'},738:{'readiness':'Human review','decision':'Pending'}}
        d,p = fixture()
        d['project_url'] = p['url']
        p['fields'][2]['options'].update({'Tracking':'ot','Human review':'oh'})
        p['fields'][3]['options']['Not applicable'] = 'ona'
        d['source_guard']['prs'] = {n:{'head':h,'state':'OPEN'} for n,h in self.scope['candidate_heads'].items()}
        d['rows'] = [dict(d['rows'][0], number=n,node_id='i'+str(n),issue_state='OPEN',
            values=dict(d['rows'][0]['values'],Readiness=('Ready' if n==806 else 'Blocked'),Decision=('Pending' if n==738 else 'Not applicable'))) for n in (737,806,738)]
        d['source_guard']['issues'] = {str(n):'signature'+str(n) for n in (737,806,738)}
        p['items'] = []
        service=MemoryService(d,p); plan=sync.prepare(d,p)
        sync.apply_plan(plan,d,service,{},lambda:None,sync.digest(plan))
        receipt=self.evidence_module.inspect_publication(d,plan,service)
        self.evidence={'desired':d,'plan':plan,'receipt':receipt}

    def validate(self):
        return validator.validate_closeout(self.snapshot,self.rows,self.scope,self.evidence)

    def test_completed_stage_accepts_verified_prior_publication(self):
        self.validate()

    def test_completed_stage_requires_evidence(self):
        with self.assertRaisesRegex(ValueError,'evidence|receipt'):
            validator.validate_closeout(self.snapshot,self.rows,self.scope)

    def test_success_label_without_actual_readback_rejected(self):
        self.evidence['receipt']['project']['items'][0]['values']['Readiness']='Ready'
        with self.assertRaises(ValueError):self.validate()

    def test_gate_approval_is_never_inferred(self):
        self.rows[738]['decision']='Approved'
        with self.assertRaisesRegex(ValueError,'gate|Pending'):self.validate()

    def test_wrong_final_readiness_rejected(self):
        self.rows[738]['readiness']='Ready'
        with self.assertRaises(ValueError):self.validate()

    def test_unresolved_runtime_closure_rejected(self):
        self.snapshot['issues'][4]['state']='CLOSED'
        with self.assertRaisesRegex(ValueError,'799'):self.validate()

    def test_closure_before_receipt_rejected(self):
        self.snapshot['issues'][0]['closedAt']='2020-01-01T00:00:00Z'
        with self.assertRaisesRegex(ValueError,'before|receipt'):self.validate()

    def test_owner_must_close_after_access_blocker(self):
        self.snapshot['issues'][0]['closedAt']='2029-01-01T00:00:00Z'
        with self.assertRaisesRegex(ValueError,'806|order'):self.validate()

    def test_not_planned_closure_rejected(self):
        self.snapshot['issues'][0]['stateReason']='NOT_PLANNED'
        with self.assertRaises(ValueError):self.validate()

    def test_consistent_evidence_for_project11_cannot_complete_project10(self):
        d,plan,r=(self.evidence[k] for k in ('desired','plan','receipt'))
        url='https://github.com/users/dhar174/projects/11'
        d['project_url']=url;plan['project_url']=url;plan['baseline']['url']=url
        plan['desired_digest']=sync.digest(d)
        r['project_url']=url;r['project']['url']=url
        r['desired_digest']=sync.digest(d);r['plan_digest']=sync.digest(plan)
        r['project_digest']=sync.digest(sync.canonical(r['project']))
        with self.assertRaisesRegex(ValueError,'Project 10|project identity'):self.validate()

    def test_prior_project_node_must_match_reviewed_identity(self):
        self.scope['project_id']='another-project'
        with self.assertRaisesRegex(ValueError,'project identity'):self.validate()

    def test_changed_prior_candidate_heads_rejected(self):
        self.evidence['desired']['source_guard']['prs']['709']['head']='other'
        with self.assertRaises(ValueError):self.validate()

    def test_prior_receipt_must_cover_same_managed_cohort(self):
        self.scope['managed_issues']=4
        with self.assertRaises(ValueError):self.validate()

    def test_preserved_scope_cannot_hide_gate_readiness(self):
        self.evidence['receipt']['project']['items'][-1]['values'].update(Scope='Optional',Readiness='Deferred')
        with self.assertRaises(ValueError):self.validate()


class ReceiptCliTests(unittest.TestCase):
    def test_receipt_only_allowed_with_inspect(self):
        error = io.StringIO()
        with patch.object(sys,'argv',['sync_project.py','--prepare','--receipt','../proof.json']), patch.object(sync,'GitHub') as api, redirect_stderr(error):
            with self.assertRaises(SystemExit):sync.main()
            api.assert_not_called()
        self.assertIn('--receipt requires --inspect', error.getvalue())

    def test_receipt_inside_frozen_package_rejected_before_network(self):
        root=Path(__file__).resolve().parent
        error = io.StringIO()
        with patch.object(sys,'argv',['sync_project.py','--inspect','--receipt',str(root/'proof.json')]), patch.object(sync,'GitHub') as api, redirect_stderr(error):
            with self.assertRaises(SystemExit):sync.main()
            api.assert_not_called()
        self.assertIn('outside the frozen package', error.getvalue())

    def test_receipt_must_not_replace_the_approved_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plan.json'; path.write_text('preserve')
            error=io.StringIO()
            with patch.object(sys,'argv',['sync_project.py','--inspect','--plan',str(path),'--receipt',str(path)]), patch.object(sync,'GitHub') as api, redirect_stderr(error):
                with self.assertRaises(SystemExit):sync.main()
                api.assert_not_called()
            self.assertEqual(path.read_text(),'preserve')
            self.assertIn('distinct',error.getvalue())

    def test_receipt_temp_file_must_not_replace_the_approved_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt=Path(tmp)/'proof.json';plan=Path(tmp)/'proof.json.tmp'
            plan.write_text('preserve');error=io.StringIO()
            with patch.object(sys,'argv',['sync_project.py','--inspect','--plan',str(plan),'--receipt',str(receipt)]), patch.object(sync,'GitHub') as api, redirect_stderr(error):
                with self.assertRaises(SystemExit):sync.main()
                api.assert_not_called()
            self.assertEqual(plan.read_text(),'preserve')
            self.assertIn('distinct',error.getvalue())

    def test_receipt_temp_symlink_cannot_write_inside_frozen_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);frozen=folder/'frozen';frozen.mkdir()
            protected=frozen/'tracked.json';protected.write_text('preserve')
            receipt=folder/'proof.json'
            try: receipt.with_suffix('.json.tmp').symlink_to(protected)
            except (OSError,NotImplementedError):self.skipTest('Symlink creation unavailable on this platform')
            desired,plan,service=example()
            sync.save(frozen/'project-preview.json',desired);sync.save(folder/'plan.json',plan)
            with patch.object(sync,'ROOT',frozen), patch.object(sys,'argv',['sync_project.py','--inspect','--plan',str(folder/'plan.json'),'--receipt',str(receipt)]), patch.object(sync,'GitHub',return_value=service) as api, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):sync.main()
                api.assert_not_called()
            self.assertEqual(protected.read_text(),'preserve')

    def test_cli_retains_real_read_only_receipt(self):
        desired,plan,service=example()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);sync.save(folder/'desired.json',desired);sync.save(folder/'plan.json',plan)
            receipt_path=folder/'proof.json'
            argv=['sync_project.py','--inspect','--desired',str(folder/'desired.json'),'--plan',str(folder/'plan.json'),'--receipt',str(receipt_path)]
            with patch.object(sys,'argv',argv), patch.object(sync,'GitHub',return_value=service), redirect_stdout(io.StringIO()):
                self.assertEqual(sync.main(),0)
            import publication_evidence
            publication_evidence.verify_receipt(sync.load(receipt_path),desired,plan)

    def test_api_failure_writes_no_success_claim(self):
        desired,plan,service=example()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);sync.save(folder/'desired.json',desired);sync.save(folder/'plan.json',plan)
            receipt_path=folder/'proof.json'
            argv=['sync_project.py','--inspect','--desired',str(folder/'desired.json'),'--plan',str(folder/'plan.json'),'--receipt',str(receipt_path)]
            with patch.object(service,'project',side_effect=RuntimeError('FORBIDDEN')), patch.object(sys,'argv',argv), patch.object(sync,'GitHub',return_value=service), redirect_stderr(io.StringIO()):
                self.assertEqual(sync.main(),1)
            self.assertEqual(sync.load(receipt_path)['status'],'blocked_or_partial')

class FullCohortLifecycleTests(unittest.TestCase):
    def test_real_332_row_builder_supports_both_evidence_bound_stages(self):
        import datetime as dt
        import build_preview
        import publication_evidence
        root=Path(__file__).resolve().parent
        inputs=[sync.load(root/name) for name in ('snapshot.json','original-roadmap.json','triage-input.json','remediations.json','approval-records.json','original-pr-handoffs.json')]
        initial=build_preview.assemble(*inputs)
        desired=initial['preview']
        project={'id':'controlled-project10','url':desired['project_url'],
            'fields':[{'id':'f'+str(i),'name':f['name'],'type':f['data_type'],'options':{o:'o'+str(j) for j,o in enumerate(f.get('options',[]))}} for i,f in enumerate(desired['fields'])],
            'views':[dict(v,id='v'+str(i),number=i+1) for i,v in enumerate(desired['views'])],
            'items':[{'id':'item'+str(r['number']),'number':r['number'],'node_id':r['node_id'],'repository':sync.REPO,'archived':False,'values':copy.deepcopy(r['values'])} for r in desired['rows']]}
        project['items'].append({'id':'foreign','number':99,'node_id':'foreign99','repository':'another/repo','archived':False,'values':{'Notes':'preserve'}})
        first_plan=sync.prepare(desired,project)
        service=MemoryService(desired,project)
        sync.apply_plan(first_plan,desired,service,{},lambda:None,sync.digest(first_plan))
        first_receipt=publication_evidence.inspect_publication(desired,first_plan,service)
        publication_evidence.verify_receipt(first_receipt,desired,first_plan)
        close_time=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=1)).isoformat()
        for issue in inputs[0]['issues']:
            if issue['number'] in (737,806):issue.update(state='CLOSED',stateReason='COMPLETED',closedAt=close_time)
            for blocker in issue['blockedBy']['nodes']:
                if blocker['number'] in (737,806):blocker['state']='CLOSED'
        final=build_preview.assemble(*inputs)
        scope=sync.load(root/'review-scope.json');scope['stage']='completed';scope['project_id']='controlled-project10'
        validator.validate_closeout(inputs[0],final['rows_by_number'],scope,{'desired':desired,'plan':first_plan,'receipt':first_receipt})
        self.assertEqual(final['rows_by_number'][738]['readiness'],'Human review')
        self.assertEqual(final['rows_by_number'][738]['decision'],'Pending')
        final_desired=final['preview'];service.source=copy.deepcopy(final_desired['source_guard'])
        final_plan=sync.prepare(final_desired,service.project())
        sync.apply_plan(final_plan,final_desired,service,{},lambda:None,sync.digest(final_plan))
        receipt=publication_evidence.inspect_publication(final_desired,final_plan,service)
        publication_evidence.verify_receipt(receipt,final_desired,final_plan)
        self.assertEqual(len(final_desired['rows']),332)
        self.assertEqual(service.state['items'][-1]['values']['Notes'],'preserve')

    def test_completed_evidence_cannot_escape_frozen_package(self):
        with self.assertRaisesRegex(ValueError,'inside the frozen package'):
            validator.load_stage_evidence({'stage':'completed','publication_evidence':dict.fromkeys(('desired','plan','receipt'),'../outside.json')})

if __name__=='__main__':unittest.main(verbosity=2)
