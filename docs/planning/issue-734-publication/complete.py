"""Finish #734 only after the approved publication receipt exists and passes."""
import importlib.util,json
from pathlib import Path
root=Path(__file__).resolve().parents[3];out=root/'docs/planning/issue-734-publication'
if (out/'completion.json').exists():
 raise SystemExit('Completion is finalized; inspect completion.json and final-readback.json. No further writes attempted.')
spec=importlib.util.spec_from_file_location('publication',out/'publish.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
verification=json.loads((out/'verification.json').read_text())
assert verification['status']=='publication_verified'
assert verification['annotations']==307 and verification['parent_links']==80
assert len(verification['closures'])==7 and len(verification['followups'])==3
assert verification['remediation_dependency_edges']==7
merge=p.api(p.REPO+'/pulls/797'); assert merge['merged'] and merge['merge_commit_sha']=='2c3bc5753b15aad4a1e345c020c5872d6ba8b25c'
followups=verification['followups'];refs=', '.join(f'{k}: #{v["number"]}' for k,v in followups.items())
comment=('<!-- issue-734-publication-complete:v1 -->\n'
 'Approved under [D027]('+p.APPROVAL+'); #797 is verified merged at 2c3bc5753b15aad4a1e345c020c5872d6ba8b25c.\n\n'
 '**Publication complete and independently read back:** 307 marked annotations with original bodies and assignees preserved; '
 '80 missing native parent links added without replacing existing parents; five scoped completions (#355, #482, #594, #596, #597); '
 'two approved duplicates (#351 → #350; #382 → #380), closed as not planned.\n\n'
 'Three separate remediation issues were created and their parents/seven blocker edges verified: '+refs+'. '
 'These remain after human gate #738; #602/#609 stay open and their failed evidence is retained. '
 'No relationship cycle found and no existing dependency removed. The frozen review input remains historical; '
 '[current publication receipts](https://github.com/dhar174/tiny_village/pull/798) are separate.\n\n'
 'Closing #734 as completed. Phase parent #733 and human gate #738 remain open. '
 'Project 10 synchronization is still the disclosed #737 handoff; no Project mutations were performed. '
 'Next #735 is prepared in the maintainer-selected #703; #714 is superseded under D028. '
 '#735 closes only after #703 merges. #736/#737 and #738 remain their own ordered work/review steps.')
comments=p.api(p.REPO+'/issues/734/comments?per_page=100')
found=[c for c in comments if '<!-- issue-734-publication-complete:v1 -->' in c['body']]
assert len(found)<=1
if not found:
 found=[p.api(p.REPO+'/issues/734/comments',{'body':comment},'POST')]
else:assert found[0]['body']==comment
p.api(p.REPO+'/issues/734',{'state':'closed','state_reason':'completed'},'PATCH')
actual=p.all_issues();state=json.loads((out/'checkpoint.json').read_text());before=json.loads((out/'before.json').read_text())
for n,rec in state['annotations'].items():
 issue=actual[int(n)];old=before[n]
 assert p.digest(issue['body'])==rec['after_sha256'],n
 assert issue['body'].endswith('\n\n'+old['body']),n
 assert issue['assignees']==old['assignees'],n
 assert issue['parent']==({'number':state['parents'][n]} if n in state['parents'] else old['parent']),n
 expected={x['number'] for x in old['blockedBy']['nodes']}
 expected.update(int(k.split('<-')[1])for k in state['edges'] if k.startswith(n+'<-'))
 assert {x['number']for x in issue['blockedBy']['nodes']}==expected,n
 if int(n) not in p.COMPLETED|set(p.DUPLICATES)|{734}:
  assert (issue['state'],issue['stateReason'])==(old['state'],old['stateReason']),n
for n in p.COMPLETED|{734}:assert actual[n]['state']=='CLOSED' and actual[n]['stateReason']=='COMPLETED',n
for n in p.DUPLICATES:assert actual[n]['state']=='CLOSED' and actual[n]['stateReason']=='NOT_PLANNED',n
for n in [602,609,733,738]:assert actual[n]['state']=='OPEN',n
for draft in json.loads((p.BUNDLE/'proposed-followups.json').read_text())['items']:
 n=state['followups'][draft['key']]['number'];i=actual[n]
 assert i['title']==draft['title'] and draft['body'] in i['body'] and i['state']=='OPEN'
 assert i['parent']['number']==draft['parent']
 assert {x['number']for x in i['blockedBy']['nodes']}==set(draft['blocked_by'])
 for dependent in draft['blocks']:assert n in {x['number']for x in actual[dependent]['blockedBy']['nodes']}
p.no_cycles({n:[i['parent']['number']]if i['parent']else[]for n,i in actual.items()})
p.no_cycles({n:[x['number']for x in i['blockedBy']['nodes']]for n,i in actual.items()})
p.save('completion.json',{'status':'issue_734_completed_and_verified','state':'CLOSED','state_reason':'COMPLETED',
 'completion_comment':found[0]['html_url'],'followups':followups,'annotation_original_body_and_assignment_checks':307,
 'parent_checks':80,'seven_scoped_closures_verified':True,'remediation_dependency_checks':7,'cycles':0,
 'phase_733_state':actual[733]['state'],'phase_gate_738_state':actual[738]['state'],
 'retained_open_failures':[602,609],'project_mutations':0})
p.save('final-readback.json',actual)
print(json.dumps(json.loads((out/'completion.json').read_text()),indent=2))
