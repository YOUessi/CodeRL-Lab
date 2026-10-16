import json
from coderl_lab.analysis.exp021_seed42_docker_score import extract_code,run_docker,verify_image
verify_image()
specs=[
 (42,'MBPPPlus/100',3,'next_smallest_palindrome','for n in range(1,401):\n expected=next(k for k in range(n+1,2000) if str(k)==str(k)[::-1])\n assert next_smallest_palindrome(n)==expected'),
 (44,'MBPPPlus/162',1,'sum_series','for n in range(0,301):\n expected=sum(n-2*i for i in range(n//2+1))\n assert sum_series(n)==expected'),
 (44,'MBPPPlus/166',0,'find_even_pair','import itertools\nfor arr in ([],[1],[2],[1,2],[2,4,6],[1,3,5],[1,2,3,4],[-3,-1,0,2],list(range(12))):\n expected=sum(((a^b)%2)==0 for a,b in itertools.combinations(arr,2))\n assert find_even_pair(arr)==expected'),
 (42,'MBPPPlus/618',2,'div_list','for a,b in [([2,4,9],[2,2,3]),([1,5],[2,4]),([-3,2],[2,4]),([5],[2])]:\n expected=[x/y for x,y in zip(a,b)]\n assert div_list(a,b)==expected'),
 (44,'MBPPPlus/9',0,'find_Rotations','for s in ["a","abc","abab","aaaa","abcdabcd","xyzxyzxyz","aba"]:\n expected=next((i for i in range(1,len(s)+1) if s[i:]+s[:i]==s),len(s))\n assert find_Rotations(s)==expected'),
]
for seed,q,i,entry,tests in specs:
 p=f'artifacts/exp021/eval_seed{seed}_96/checkpoint.jsonl';rows={(r['task_id'],r['sample_id'],r['arm']):r for r in map(json.loads,open(p))}
 out={}
 for arm in ('sft','grpo'):
  code,_=extract_code(rows[q,i,arm]['raw_code']);out[arm]=run_docker(code,tests,entry)
 print(seed,q,i,out,flush=True)
