"""Conservatively audit/remove expired CodeRL candidate containers, never services.

Requires EXACT known driver code, no network, nonroot, read-only filesystem,
AutoRemove, all capabilities dropped, no bind mounts, explicit resource limits,
and age over 120 seconds. No images, volumes, repositories or other services
are deleted. Evidence is written before removals and after completion.
"""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,re,subprocess
from pathlib import Path
from coderl_lab.train.exp024_reward import DRIVER as STRICT_DRIVER
from coderl_lab.train.train014b_reward_sandbox import DRIVER as TRAIN_DRIVER
from coderl_lab.analysis.exp021_seed42_docker_score import DRIVER as EVAL_DRIVER

KNOWN={hashlib.sha256(x.encode()).hexdigest() for x in (STRICT_DRIVER,TRAIN_DRIVER,EVAL_DRIVER)}

def parse_docker_time(value):
    value=value.replace('Z','+00:00')
    value=re.sub(r'\.(\d+)(?=[+-])',lambda m:'.'+m.group(1)[:6].ljust(6,'0'),value)
    return dt.datetime.fromisoformat(value)

def recognized(d,now):
    cfg=d['Config'];host=d['HostConfig'];state=d['State']
    args=cfg.get('Cmd') or []
    matches=[hashlib.sha256(x.encode()).hexdigest() for x in args if isinstance(x,str) and hashlib.sha256(x.encode()).hexdigest() in KNOWN]
    started=parse_docker_time(state['StartedAt']);age=(now-started).total_seconds()
    checks={'exact_known_driver':len(matches)==1,'network_none':host.get('NetworkMode')=='none',
            'auto_remove':host.get('AutoRemove') is True,'read_only':host.get('ReadonlyRootfs') is True,
            'nonroot':cfg.get('User')=='65534:65534','drop_all':'ALL' in (host.get('CapDrop') or []),
            'memory_limit':0<host.get('Memory',0)<=1024**3,'pids_limit':0<(host.get('PidsLimit') or 0)<=64,
            'no_bind_or_volume_mount':all(m.get('Type')=='tmpfs' for m in d.get('Mounts',[])),
            'expired':age>121,'running':state.get('Running') is True}
    return {'id':d['Id'],'name':d.get('Name'),'image_id':d['Image'],'age_seconds':round(age,2),
            'driver_sha256':matches[0] if len(matches)==1 else None,'checks':checks,'eligible':all(checks.values())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    ids=subprocess.run(['docker','ps','-q'],capture_output=True,text=True,check=True,timeout=15).stdout.split()
    details=json.loads(subprocess.run(['docker','inspect',*ids],capture_output=True,text=True,check=True,timeout=20).stdout) if ids else []
    now=dt.datetime.now(dt.timezone.utc);audit=[recognized(x,now) for x in details]
    record={'experiment':'EXP026-expired-sandbox-cleanup','time':now.isoformat(),'apply':a.apply,'audited':audit,'removed':[]}
    a.out.parent.mkdir(parents=True,exist_ok=True);pre=a.out.with_suffix('.before.json')
    if pre.exists():raise FileExistsError(pre)
    pre.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    if a.apply:
        for item in audit:
            if not item['eligible']:continue
            r=subprocess.run(['docker','inspect',item['id']],capture_output=True,text=True,timeout=10)
            if r.returncode!=0:continue
            check=recognized(json.loads(r.stdout)[0],dt.datetime.now(dt.timezone.utc))
            if not check['eligible']:raise RuntimeError('container contract changed; abort')
            subprocess.run(['docker','rm','-f',item['id']],capture_output=True,text=True,check=True,timeout=15)
            record['removed'].append(item['id'])
    a.out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'audited':len(audit),'eligible':sum(x['eligible'] for x in audit),'removed':len(record['removed']),'skipped':sum(not x['eligible'] for x in audit)}))
if __name__=='__main__':main()
