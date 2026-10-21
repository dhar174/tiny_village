import json, os, subprocess, sys
from pathlib import Path
root=Path('/workspace/tiny_village');sys.path.insert(0,str(root))
from tests.subprocess_capture import capture_python, CapturedTimeout
out=root/'docs/planning/issue-734/evidence'
results=[]
for name,profile in [('weather','visual'),('queue','core'),('named_target','core'),('live_timing','visual'),('frame_analysis','core'),('social','core'),('existing_regressions','core')]:
    env=os.environ.copy();env.update({'TINY_TEST_ARTIFACT_DIR':str(out/'captures'), 'SDL_VIDEODRIVER':'dummy','SDL_AUDIODRIVER':'dummy','PYGAME_HIDE_SUPPORT_PROMPT':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','MPLBACKEND':'Agg','XDG_CACHE_HOME':'/workspace/scratch/issue-734/cache'})
    if name=='existing_regressions':
        source="import pytest; raise SystemExit(pytest.main(['-q','tests/test_action_named_target_resolution.py','tests/test_goap_planner_integration_real.py','--tb=short']))"
    else:
        source=f"import runpy,sys; sys.argv=['audit_probes.py',{name!r}]; runpy.run_path({str(out/'audit_probes.py')!r},run_name='__main__')"
    payload=out/f'{name}.source.txt';payload.write_text(source+'\n')
    code='''import json,os,sys
from pathlib import Path
sys.path.insert(0,'/workspace/tiny_village')
from tests.subprocess_capture import capture_python,CapturedTimeout
try:
 r=capture_python(Path(sys.argv[1]).read_text(),cwd=Path('/workspace/tiny_village'),env=os.environ.copy(),test_id=sys.argv[2],fatal_timeout=20,timeout=25)
 print(json.dumps({'returncode':r.completed.returncode,'artifact_dir':str(r.artifact_dir)}))
except CapturedTimeout as e:
 print(json.dumps({'timeout':True,'artifact_dir':str(e.artifact_dir)}))
'''
    interpreter=f'/workspace/.venvs/issue726-{profile}/bin/python'
    run=subprocess.run([interpreter,'-c',code,str(payload),name],env=env,text=True,capture_output=True,timeout=40)
    result={'name':name,'profile':profile,'interpreter':interpreter,'parent_returncode':run.returncode,'capture':json.loads(run.stdout.strip().splitlines()[-1]) if run.stdout.strip() else {},'parent_stderr':run.stderr}
    results.append(result);(out/'audit-results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(result),flush=True)
