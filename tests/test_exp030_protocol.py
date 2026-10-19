import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from coderl_lab.analysis.exp030_protocol import (
    canonical, once, postprocess, snippet_payload, validate_generations,
    task_seed, generation_protocol, sha,
)
from coderl_lab.datasets.exp030_ds1000 import eligible, freeze_source, validate_dataset


def problem(q=291, kind='Origin', library='Numpy', dependency='numpy'):
    return {'metadata': {'problem_id': q, 'library': library,
                         'perturbation_type': kind, 'perturbation_origin_id': q},
            'prompt': 'Problem: return result\nBEGIN SOLUTION\n<code>\n',
            'reference_code': 'result = 3',
            'code_context': f'import {dependency}\ndef test_execution(code):\n    pass\n'}


class ProtocolTests(unittest.TestCase):
    def test_origin_dependency_filter_is_fixed_before_scores(self):
        self.assertTrue(eligible(problem()))
        self.assertFalse(eligible(problem(kind='Surface')))
        self.assertFalse(eligible(problem(library='Pandas')))
        self.assertFalse(eligible(problem(dependency='scipy')))
        r=problem();r['code_context']='import tokenize, io\n'+r['code_context']
        self.assertTrue(eligible(r))
        r=problem();r['code_context']+='\nexec_context = """\nimport torch\n"""\n'
        self.assertFalse(eligible(r))

    def test_public_data_never_contains_reference_or_checker(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            m=freeze_source([problem()],root,'a'*64,expected_count=1)
            public=json.loads((root/'public.jsonl').read_text())
            self.assertEqual(set(public), {'task_id','source_task_id','messages'})
            self.assertEqual(public['messages'][-1]['content'],problem()['prompt'])
            self.assertNotIn('reference_code',canonical(public))
            self.assertEqual(validate_dataset(root,private=True)['task_count'],1)
            (root/'public.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):validate_dataset(root,private=False)

    def test_immutable_writes_reject_changed_content(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'one.json';once(p,{'x':1});once(p,{'x':1})
            with self.assertRaises(ValueError):once(p,{'x':2})

    def test_official_postprocess_preserves_solution_text(self):
        self.assertEqual(postprocess('```python\nresult = 2\n```\nprose'),'\nresult = 2\n')
        self.assertEqual(postprocess('<code>x=1</code>ignored'),'x=1')
        self.assertEqual(postprocess('x=1\nEND SOLUTION\nignored'),'x=1')

    def test_snippet_payload_is_inert_until_container_execution(self):
        raw='__import__("os").system("false")\nresult = "triple \\\" quote"'
        code,oracle=snippet_payload(raw,{'code_context':'def test_execution(code):\n    pass\n'})
        tree=ast.parse(code)
        self.assertEqual(len(tree.body),1)
        self.assertIsInstance(tree.body[0],ast.FunctionDef)
        value=tree.body[0].body[0].value
        self.assertIsInstance(value,ast.Constant)
        self.assertEqual(value.value,raw)
        self.assertIn('test_execution(__exp030_code)',oracle['test'])
        self.assertNotIn('test_string(__exp030_code)',oracle['test'])
        _,oracle2=snippet_payload('x=1',{'code_context':'def test_string(code):\n    pass\n'})
        self.assertIn('test_string(__exp030_code)',oracle2['test'])

    def test_draw_integrity_detects_duplicate_and_changed_code(self):
        data=[{'task_id':'DS1000/291','source_task_id':291,'messages':[]}]
        rr=[{'task_id':'DS1000/291','sample_id':i,'seed':task_seed('DS1000/291'),
             'raw_code':'x=1','code_sha256':hashlib.sha256(b'x=1').hexdigest(),
             'generated_tokens':3,'input_tokens':20,'reached_token_limit':False} for i in range(4)]
        self.assertEqual(len(validate_generations(rr,data,require_complete=True)),4)
        with self.assertRaises(ValueError):validate_generations(rr+[rr[0]],data)
        rr[0]={**rr[0],'raw_code':'x=2'}
        with self.assertRaises(ValueError):validate_generations(rr,data)

    def test_generation_scheme_is_same_for_all_models(self):
        p=generation_protocol()
        self.assertEqual(p['draws'],4)
        self.assertEqual(p['max_new_tokens'],256)
        self.assertEqual(p['sampling'],'per_task_seed_num_return_sequences_4')
        self.assertEqual(task_seed('DS1000/291'),task_seed('DS1000/291'))
        self.assertNotEqual(task_seed('DS1000/291'),task_seed('DS1000/292'))


if __name__=='__main__':unittest.main()
