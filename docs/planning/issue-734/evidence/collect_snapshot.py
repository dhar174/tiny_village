import json, subprocess, hashlib, datetime
from pathlib import Path
root=Path(__file__).resolve().parents[1]
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def api(q,**variables):
    cmd=['gh','api','graphql','-f','query='+q]
    for k,v in variables.items():
        if v is not None: cmd+=['-f',f'{k}={v}']
    r=json.loads(subprocess.check_output(cmd,text=True))
    if r.get('errors'): raise RuntimeError(r['errors'])
    return r['data']
manifest={'repository':'dhar174/tiny_village','started_at':now(),'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'historical_cutoff':'2026-10-13T07:30:00Z','github_mutations_performed':0}
(root/'snapshot-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
issueq='''query($after:String){repository(owner:"dhar174",name:"tiny_village"){issues(first:50,after:$after,orderBy:{field:CREATED_AT,direction:ASC}){totalCount pageInfo{hasNextPage endCursor} nodes{number id url title body state stateReason createdAt updatedAt closedAt author{login} assignees(first:100){nodes{login} pageInfo{hasNextPage}} labels(first:100){nodes{name} pageInfo{hasNextPage}} parent{number title url} blockedBy(first:100){nodes{number title state url} pageInfo{hasNextPage}} comments(first:100){nodes{url body createdAt updatedAt author{login}} pageInfo{hasNextPage endCursor}} timelineItems(first:100,itemTypes:[CLOSED_EVENT,REOPENED_EVENT]){nodes{__typename ... on ClosedEvent{createdAt} ... on ReopenedEvent{createdAt}} pageInfo{hasNextPage endCursor}}}}}}'''
allissues=[];after=None;pages=[]
while True:
    d=api(issueq,after=after)['repository']['issues'];pages.append({'page':len(pages)+1,'pageInfo':d['pageInfo'],'totalCount':d['totalCount']});allissues+=d['nodes'];print('issue page',len(pages),'retained',len(allissues),flush=True)
    if not d['pageInfo']['hasNextPage']: break
    after=d['pageInfo']['endCursor']
for n in allissues:
    for name in ['assignees','labels','blockedBy']:
        if n[name]['pageInfo']['hasNextPage']:raise RuntimeError(f'Overflow {n["number"]} {name}')
    for name,field in [('comments','nodes{url body createdAt updatedAt author{login}}'),('timelineItems','nodes{__typename ... on ClosedEvent{createdAt} ... on ReopenedEvent{createdAt}}')]:
        while n[name]['pageInfo']['hasNextPage']:
            arg=',itemTypes:[CLOSED_EVENT,REOPENED_EVENT]' if name=='timelineItems' else ''
            q='query($after:String){repository(owner:"dhar174",name:"tiny_village"){issue(number:'+str(n['number'])+'){'+name+'(first:100,after:$after'+arg+'){'+field+' pageInfo{hasNextPage endCursor}}}}}'
            d=api(q,after=n[name]['pageInfo']['endCursor'])['repository']['issue'][name];n[name]['nodes']+=d['nodes'];n[name]['pageInfo']=d['pageInfo']
(root/'issues-snapshot.json').write_text(json.dumps({'pages':pages,'issues':allissues},indent=2,ensure_ascii=False)+'\n')
prq='''query($after:String){repository(owner:"dhar174",name:"tiny_village"){pullRequests(first:50,after:$after,states:OPEN,orderBy:{field:CREATED_AT,direction:ASC}){totalCount pageInfo{hasNextPage endCursor} nodes{number url title body createdAt updatedAt headRefOid headRefName baseRefName isDraft mergeable author{login} assignees(first:100){nodes{login}} closingIssuesReferences(first:100){nodes{number} pageInfo{hasNextPage}} files(first:100){nodes{path additions deletions} pageInfo{hasNextPage}} comments(first:100){nodes{url body author{login}} pageInfo{hasNextPage}} reviews(last:100){nodes{url state body author{login} submittedAt commit{oid}} pageInfo{hasPreviousPage}}}}}}'''
prs=[];after=None;pp=[]
while True:
    d=api(prq,after=after)['repository']['pullRequests'];pp.append({'pageInfo':d['pageInfo'],'totalCount':d['totalCount']});prs+=d['nodes'];print('PRs retained',len(prs),flush=True)
    if not d['pageInfo']['hasNextPage']:break
    after=d['pageInfo']['endCursor']
(root/'prs-snapshot.json').write_text(json.dumps({'pages':pp,'pull_requests':prs},indent=2,ensure_ascii=False)+'\n')
cut=manifest['historical_cutoff'];cohort=[]
for n in allissues:
    if n['createdAt']>=cut:continue
    state='OPEN'
    for ev in sorted(n['timelineItems']['nodes'],key=lambda e:e['createdAt']):
        if ev['createdAt']>=cut:break
        state='CLOSED' if ev['__typename']=='ClosedEvent' else 'OPEN'
    if state=='OPEN':cohort.append(n['number'])
assert len(cohort)==261, len(cohort)
manifest.update({'completed_at':now(),'issue_count':len(allissues),'open_issues':sum(i['state']=='OPEN' for i in allissues),'open_prs':len(prs),'historical_open_issues':cohort,'comments_retained':sum(len(i['comments']['nodes']) for i in allissues),'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'issues-snapshot.json',root/'prs-snapshot.json']},'pr_nested_overflow':[{'number':p['number'],'field':f} for p in prs for f in ['closingIssuesReferences','files','comments','reviews'] if p[f]['pageInfo'].get('hasNextPage',False) or p[f]['pageInfo'].get('hasPreviousPage',False)]})
(root/'snapshot-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k not in ['historical_open_issues','hashes']},indent=2))
