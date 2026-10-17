"""EXP028 pretraining amendment: same budgets, explicitly verified OOM outcome.
Unknown 137, wall guard and cleanup failures remain indeterminate. No retries.
"""
from __future__ import annotations
import json,subprocess,uuid
from coderl_lab.train.exp027_guarded_probe import SandboxIndeterminate,parse_verdict
from coderl_lab.train.exp024_reward import DRIVER
from coderl_lab.train.train014b_reward_sandbox import IMAGE_ID,validate_tests


def resource_exit(code,state):
    if (code==137 and state.get('OOMKilled') is True and state.get('ExitCode')==137 and state.get('Running') is False):
        return {'loaded':False,'load_error':'memory_budget_exceeded','cases':[],
                'resource_limited':True,'oom_verified_by_engine':True}
    raise SandboxIndeterminate('unclassified_container_exit_'+str(code))


def docker_probe(code,tests,wall_seconds=60.):
    if not isinstance(code,str) or not code or len(code)>16000:raise ValueError('invalid source')
    tests=validate_tests(tests)
    if not 0<wall_seconds<=60:raise ValueError('invalid wall guard')
    name='coderl-guard028-'+uuid.uuid4().hex[:16]
    cmd=['docker','run','--name',name,'-i','--network','none','--read-only','--cap-drop','ALL',
         '--security-opt','no-new-privileges','--user','65534:65534','--memory','256m','--cpus','1','--pids-limit','32',
         '--tmpfs','/tmp:rw,nosuid,size=16m','--workdir','/tmp','-e','PYTHONDONTWRITEBYTECODE=1','--entrypoint','python',IMAGE_ID,'-I','-c',DRIVER]
    try:
        try:r=subprocess.run(cmd,input=json.dumps({'code':code,'tests':tests}),capture_output=True,text=True,timeout=wall_seconds)
        except subprocess.TimeoutExpired as e:raise SandboxIndeterminate('wall_guard_expired_no_reward') from e
        if r.returncode==0:return parse_verdict(r.stdout,tests)
        if r.returncode==137:
            try:
                q=subprocess.run(['docker','inspect','--format','{{json .State}}',name],capture_output=True,text=True,timeout=10,check=True)
                state=json.loads(q.stdout)
            except (OSError,ValueError,subprocess.SubprocessError) as e:raise SandboxIndeterminate('OOM_evidence_unavailable') from e
            return resource_exit(r.returncode,state)
        raise SandboxIndeterminate('unclassified_container_exit_'+str(r.returncode))
    finally:
        try:c=subprocess.run(['docker','rm','-f',name],capture_output=True,text=True,timeout=15)
        except (OSError,subprocess.SubprocessError) as e:raise SandboxIndeterminate('container_cleanup_unconfirmed') from e
        if c.returncode and 'No such container' not in c.stderr:raise SandboxIndeterminate('container_cleanup_failed')
