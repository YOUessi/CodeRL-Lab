"""Guarded probe for subsequent training protocols, not a historical-score patch.

Host wall timeout is INDETERMINATE, never a wrong-answer label. A caller must
abort/checkpoint its experiment rather than feed a fabricated zero into GRPO.
This does not claim deterministic runtime or implement a universal CPU oracle.
"""
from __future__ import annotations
import json,subprocess,uuid
from coderl_lab.train.exp024_reward import DRIVER
from coderl_lab.train.train014b_reward_sandbox import IMAGE_ID,validate_tests
from coderl_lab.analysis.exp026_core import features

class SandboxIndeterminate(RuntimeError):
    """No trustworthy semantic reward is available for this execution."""


def parse_verdict(stdout,tests):
    marker='__EXP024__';messages=[s[len(marker):] for s in stdout.splitlines() if s.startswith(marker)]
    if len(messages)!=1:raise SandboxIndeterminate('missing or nonunique verdict')
    try:
        result=json.loads(messages[0]);features(result)
    except (ValueError,TypeError,KeyError) as exc:raise SandboxIndeterminate('malformed verdict') from exc
    if result['loaded'] and len(result['cases'])!=len(tests):raise SandboxIndeterminate('case count mismatch')
    return result


def docker_probe(code,tests,wall_seconds=8.):
    if not isinstance(code,str) or not code or len(code)>16000:raise ValueError('invalid candidate source')
    tests=validate_tests(tests)
    if not 0<wall_seconds<=60:raise ValueError('invalid safety wall guard')
    name='coderl-guard027-'+uuid.uuid4().hex[:16]
    cmd=['docker','run','--rm','--name',name,'-i','--network','none','--read-only',
         '--cap-drop','ALL','--security-opt','no-new-privileges','--user','65534:65534',
         '--memory','256m','--cpus','1','--pids-limit','32','--tmpfs','/tmp:rw,nosuid,size=16m',
         '--workdir','/tmp','-e','PYTHONDONTWRITEBYTECODE=1','--entrypoint','python',IMAGE_ID,'-I','-c',DRIVER]
    try:
        try:r=subprocess.run(cmd,input=json.dumps({'code':code,'tests':tests}),capture_output=True,text=True,timeout=wall_seconds)
        except subprocess.TimeoutExpired as exc:raise SandboxIndeterminate('wall_guard_expired_no_reward') from exc
        if r.returncode!=0:raise SandboxIndeterminate('unclassified_container_exit_'+str(r.returncode))
        return parse_verdict(r.stdout,tests)
    finally:
        try:clean=subprocess.run(['docker','rm','-f',name],capture_output=True,text=True,timeout=15)
        except (subprocess.SubprocessError,OSError) as exc:raise SandboxIndeterminate('container_cleanup_not_confirmed') from exc
        if clean.returncode!=0 and 'No such container' not in clean.stderr:
            raise SandboxIndeterminate('container_cleanup_failed')
