import copy
import math
from pathlib import Path
import tempfile
import unittest
from coderl_lab.analysis import reward_signal_audit as audit


def rows():
    result=[]
    for group in range(32):
        for i in range(4):
            kind=group%4
            e,d,s=((1,1,1) if kind==0 else (0,0,0) if kind==1 else
                   (1,1,0) if kind==2 else (1,1,i/3))
            r=round(.1*e+.1*d+.8*s,6)
            result.append({'attempt_id':len(result),'group_id':group,'task_id':f't{group%2}',
                           'variant':'corrected','reward':r,'corrected_reward':r,
                           'executable_fraction':e,'dependency_fraction':d,'semantic_fraction':s,
                           'code_sha256':'a'*64,'tests_sha256':'b'*64})
    return result


class AuditTests(unittest.TestCase):
    def test_four_categories_and_semantic_distinction(self):
        report=audit.audit(rows(),{'t0':'family0','t1':'family1'})
        self.assertEqual([report['summary'][k] for k in audit.CLASSES],[8]*4)
        self.assertEqual(report['summary']['semantic_all_wrong'],16)
        self.assertEqual(report['summary']['final_reward_zero_variance'],24)
        self.assertEqual(report['groups'][2]['classification'],'same_intermediate')
        self.assertTrue(report['groups'][2]['semantic_all_wrong'])
        self.assertAlmostEqual(report['groups'][2]['reward_mean'],.2)
        self.assertEqual(sum(r['groups'] for r in report['families']),32)

    def test_partial_correctness_varies_without_binary_pass_variance(self):
        data=rows()
        for i in range(4):
            r=data[12+i];r['semantic_fraction']=i/8
            r['reward']=r['corrected_reward']=round(.2+.8*i/8,6)
        g=audit.audit(data,{'t0':'a','t1':'b'})['groups'][3]
        self.assertFalse(g['final_reward_zero_variance'])
        self.assertFalse(g['semantic_fraction_zero_variance'])
        self.assertTrue(g['binary_correctness_zero_variance'])

    def test_reject_bad_records(self):
        mutations=[lambda r:r.pop(),lambda r:r.append(copy.deepcopy(r[0])),
                   lambda r:r[0].pop('reward'),lambda r:r[0].update(reward=float('nan')),
                   lambda r:r[0].update(attempt_id=1),lambda r:r[0].update(group_id=3),
                   lambda r:r[0].update(variant='legacy'),lambda r:r[0].update(reward=.3),
                   lambda r:r[0].update(task_id='unknown'),lambda r:r[1].update(task_id='t1'),
                   lambda r:r[1].update(tests_sha256='c'*64),
                   lambda r:r[0].update(executable_fraction=.5),lambda r:r[0].update(code_sha256='bad')]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                data=rows();mutate(data)
                with self.assertRaises(ValueError):audit.audit(data,{'t0':'a','t1':'b'})

    def test_strict_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.jsonl'
            for value in ('{"a":1}\n{"a":','{"a":NaN}\n','{"a":1,"a":2}\n','{}\n\n','[]\n',''):
                path.write_text(value)
                with self.subTest(value=value),self.assertRaises(ValueError):audit.read_jsonl(path)
            path.write_text('{"a":1}\n');self.assertEqual(audit.read_jsonl(path),[{'a':1}])

    def test_family_mapping_rejects_duplicates_and_missing(self):
        data=[{'task_id':f't{i}','task_kind':f'f{i%6}'} for i in range(48)]
        self.assertEqual(len(audit.task_families(data)),48)
        data[1]['task_id']='t0'
        with self.assertRaises(ValueError):audit.task_families(data)
        with self.assertRaises(ValueError):audit.task_families([{'task_id':'t0'}])

    def test_no_candidate_content_exported(self):
        data=rows()
        for row in data:row.update(code='SECRET',test_cases=['SECRET'],prompt='SECRET')
        import json
        report=json.dumps(audit.audit(data,{'t0':'a','t1':'b'}))
        self.assertNotIn('SECRET',report)

if __name__=='__main__':unittest.main()
