"""No model code is executed by these pure dataset tests."""
import importlib.util
import itertools
import pytest


def module():
    assert importlib.util.find_spec('coderl_lab.datasets.exp028_coverage'), 'EXP028 data implementation missing'
    from coderl_lab.datasets import exp028_coverage
    return exp028_coverage


@pytest.mark.parametrize('kind,k,x,want',[
 ('prefix_threshold',2,[2,-1,3],2),
 ('prefix_threshold',1,[],0),
 ('sorted_gap',2,[7,1,1,4],2),
 ('subset_mod',3,[1,2,3],3),
 ('subset_mod',2,[],0),
 ('pair_distance',2,[-2,0,2],2),
 ('repeated_block',2,'ababa',2),
 ('repeated_block',5,'aa',0),
 ('window_minimum',2,[3,-1,2,4],0),
 ('window_minimum',3,[],0),
])
def test_small_oracles(kind,k,x,want):
    assert module().new_oracle(kind,k,x)==want


def test_subset_oracle_against_independent_dp():
    m=module()
    for n in range(5):
        for a in itertools.product((-2,0,1),repeat=n):
            for k in (1,2,3):
                counts=[1]+[0]*(k-1)
                for v in a:
                    before=counts[:]
                    for j,c in enumerate(before):counts[(j+v)%k]+=c
                assert m.new_oracle('subset_mod',k,list(a))==counts[0]-1


def test_new_rows_reproducible_and_reference_separate():
    m=module();rows,refs=m.make_new_rows()
    assert (rows,refs)==m.make_new_rows()
    assert len(rows)==24 and len(refs)==24
    assert len({r['task_kind'] for r in rows})==6
    for r in rows:
        assert set(r)=={'task_id','prompt','test_cases','task_kind'}
        assert len(r['test_cases'])==32
        assert len({m.canonical(t['x']) for t in r['test_cases']})==32
        assert 'test_cases' not in r['prompt'][0]['content']
        assert 'def solve' in refs[r['task_id']]


def fake_narrow():
    return [{'task_id':f'train/f{j}/{i}','task_kind':f'f{j}',
             'prompt':[{'role':'user','content':f'solve family {j} parameter {i}'}],
             'test_cases':[{'x':str(n),'expected':n} for n in range(10)]}
            for j in range(6) for i in range(8)]


def test_balanced_arms_keep_exactly_24_original_rows():
    m=module();old=fake_narrow();a,refs=m.build_arms(old)
    assert a['narrow']==old
    assert len(a['broad'])==48
    assert len({r['task_kind'] for r in a['broad']})==12
    oldids={r['task_id'] for r in old}
    reused=[r for r in a['broad'] if r['task_id'] in oldids]
    assert len(reused)==24
    assert all(r in old for r in reused)
    assert len({r['task_id'] for r in a['narrow']+a['broad']})==72


def test_duplicate_training_id_rejected():
    rows=fake_narrow();rows[-1]=rows[0]
    with pytest.raises(ValueError):module().build_arms(rows)


def test_unknown_family_rejected():
    with pytest.raises(ValueError):module().new_oracle('unknown',1,[])
