"""EXP028 v3 resource protocol: bounded CPU time, persistent container evidence.

For future registered runs, not retroactive scoring. Kernel CPU exhaustion and
engine-confirmed OOM are explicit budget failures. Host wall timeout, malformed
results and unidentified exits remain indeterminate. Never retries a candidate.
"""
from __future__ import annotations
import json,subprocess,uuid
from coderl_lab.train.exp024_reward import DRIVER as BASE_DRIVER
from coderl_lab.train.exp027_guarded_probe import SandboxIndeterminate,parse_verdict
from coderl_lab.train.exp028_resource_guard import resource_exit
from coderl_lab.train.train014b_reward_sandbox import IMAGE_ID,validate_tests

CPU_SECONDS=15
CPU_HARD_SECONDS=16
CPU_MARKER='__EXP028_CPU_BUDGET__'
CPU_PREFIX='''import resource,signal,os

def _exp028_cpu_limit(signum,frame):
    os.write(1,b"__EXP028_CPU_BUDGET__\\n")
    os._exit(124)
signal.signal(signal.SIGXCPU,_exp028_cpu_limit)
resource.setrlimit(resource.RLIMIT_CPU,({cpu},{hard}))
'''
DRIVER=CPU_PREFIX.format(cpu=CPU_SECONDS,hard=CPU_HARD_SECONDS)+BASE_DRIVER


def decode_result(rc,stdout,state,tests):
    markers=stdout.splitlines().count(CPU_MARKER)
    if rc==124:
        if (markers==1 and state.get('ExitCode')==124 and state.get('Running') is False
            and state.get('OOMKilled') is False and '__EXP024__' not in stdout):
            return {'loaded':False,'load_error':'cpu_budget_exceeded','cases':[],
                    'resource_limited':True,'cpu_exit_verified':True}
        raise SandboxIndeterminate('CPU_budget_evidence_inconsistent')
    if markers:raise SandboxIndeterminate('unexpected_CPU_budget_marker')
    if rc==137:return resource_exit(rc,state)
    if rc!=0:raise SandboxIndeterminate('unclassified_container_exit_'+str(rc))
    return parse_verdict(stdout,tests)


def docker_probe(code,tests,wall_seconds=60.,cpu_seconds=CPU_SECONDS):
    if not isinstance(code,str) or not code or len(code)>16000:raise ValueError('invalid source')
    tests=validate_tests(tests)
    if not 0<wall_seconds<=60 or type(cpu_seconds) is not int or not 1<=cpu_seconds<=CPU_SECONDS:
        raise ValueError('invalid resource budget')
    driver=CPU_PREFIX.format(cpu=cpu_seconds,hard=cpu_seconds+1)+BASE_DRIVER
    name='coderl-guard028v3-'+uuid.uuid4().hex[:16]
    cmd=['docker','run','--name',name,'-i','--network','none','--read-only','--cap-drop','ALL',
         '--security-opt','no-new-privileges','--user','65534:65534','--memory','256m','--cpus','1','--pids-limit','32',
         '--tmpfs','/tmp:rw,nosuid,size=16m','--workdir','/tmp','-e','PYTHONDONTWRITEBYTECODE=1',
         '--entrypoint','python',IMAGE_ID,'-I','-c',driver]
    try:
        try:r=subprocess.run(cmd,input=json.dumps({'code':code,'tests':tests}),capture_output=True,text=True,timeout=wall_seconds)
        except subprocess.TimeoutExpired as e:raise SandboxIndeterminate('wall_guard_expired_no_reward') from e
        state={}
        if r.returncode in (124,137):
            try:
                q=subprocess.run(['docker','inspect','--format','{{json .State}}',name],capture_output=True,text=True,timeout=10,check=True)
                state=json.loads(q.stdout)
            except (OSError,ValueError,subprocess.SubprocessError) as e:raise SandboxIndeterminate('resource_exit_evidence_unavailable') from e
        return decode_result(r.returncode,r.stdout,state,tests)
    finally:
        try:c=subprocess.run(['docker','rm','-f',name],capture_output=True,text=True,timeout=15)
        except (OSError,subprocess.SubprocessError) as e:raise SandboxIndeterminate('container_cleanup_unconfirmed') from e
        if c.returncode and 'No such container' not in c.stderr:raise SandboxIndeterminate('container_cleanup_failed')
