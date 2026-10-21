import json,re,subprocess,concurrent.futures
from pathlib import Path
p=Path(__file__).resolve().parents[1];m=json.loads((p/'snapshot-manifest.json').read_text());nums=set(m['historical_open_issues']);issues=json.loads((p/'issues-snapshot.json').read_text())['issues'];refs={}
for i in issues:
 if i['number'] in nums:
  for pr,comment in re.findall(r'/pull/(\d+)#discussion_r(\d+)',i['body']):refs.setdefault(comment,{'pr':int(pr),'issues':[]})['issues'].append(i['number'])
def get(item):
 cid,context=item;r=subprocess.run(['gh','api',f'repos/dhar174/tiny_village/pulls/comments/{cid}'],capture_output=True,text=True)
 return {'comment_id':int(cid),**context,'data':json.loads(r.stdout) if r.returncode==0 else None,'error':None if r.returncode==0 else r.stderr.strip()}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: rows=list(pool.map(get,refs.items()))
(p/'review-origin-snapshot.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
print('review origins',len(rows),'errors',sum(r['error'] is not None for r in rows))
for r in rows:print(r['issues'],(r['data'] or {}).get('path'),r['error'] or '')
