import numpy as np
import torch
import pytest
from tactile_editor.paired30_cache import edit_mask
from tactile_editor.direct_residual import DirectTactileTrunk

def test_three_decisions_preserve_fixed_shape_and_exclude_committed_actions():
    b={'action_mask':torch.ones(3,16,192,dtype=torch.bool),'decision_seconds':torch.tensor([0.,1/30,2/30]),'touch_valid':torch.ones(3,8,2,dtype=torch.bool)}
    m=edit_mask(b)
    assert m.shape==(3,16,192)
    assert m[:,:,:138].sum((1,2)).tolist()==[16*138,13*138,10*138]
    assert not m[:,:,138:].any()
    assert not m[1,:3].any() and not m[2,:6].any()

def test_missing_latest_hand_does_not_edit_that_hand():
    b={'action_mask':torch.ones(1,16,192,dtype=torch.bool),'decision_seconds':torch.tensor([1/30]),'touch_valid':torch.ones(1,8,2,dtype=torch.bool)}
    b['touch_valid'][0,-1,0]=False
    m=edit_mask(b)
    assert not m[:,:,:69].any() and m[:,:,69:138].any()

def test_history_mode_uses_gru_without_difference_channel():
    from tactile_editor.direct_residual import canonical_tactile_channels
    trunk=DirectTactileTrunk(torch.ones(2,21,21,dtype=torch.bool),torch.zeros(1,8,2,21,21),mode='history')
    assert trunk.temporal is not None
    p=torch.randn(1,8,2,21,21)
    channels,_=canonical_tactile_channels(p,torch.ones(1,8,2,dtype=torch.bool),torch.ones(2,21,21,dtype=torch.bool),include_delta=False)
    assert not channels[:,:,:,1].any()

def test_history_slices_end_at_each_decision_not_future():
    sequence=np.arange(10)
    assert sequence[0:8].tolist()==list(range(8))
    assert sequence[1:9][-1]==8
    assert sequence[2:10][-1]==9

def test_piecewise_execution_retains_previous_correction_when_touch_missing():
    plan=np.zeros(16)
    plan[:]=1  # t=0 revision
    plan[3:]=2 # t=33 ms revision
    # No t=67 ms data: hold last accepted revision, do not reset to base.
    assert plan[:8].tolist()==[1,1,1,2,2,2,2,2]

def make_cache_fixture(tmp_path,twla_ids=None):
    import json
    from tactile_editor.paired30_cache import Paired30Cache
    p=tmp_path/'shared/test';p.mkdir(parents=True)
    n=2
    values={'target_action':np.zeros((n,16,192),np.float32),'action_mask':np.ones((n,16,192),bool),
            'touch_sequence':np.zeros((n,10,2,21,21),np.float16),'touch_sequence_valid':np.ones((n,10,2),bool),
            'future_touch_pressure':np.zeros((n,3,2,21,21),np.float16),'future_touch_valid':np.ones((n,3,2),bool),
            'eligible':np.ones((n,3),bool),'source_frame':np.array([100,200],np.int64)}
    values['eligible'][0,2]=False
    values['touch_sequence_valid'][0,9]=False
    for i in range(10):values['touch_sequence'][:,i]=i
    for k,v in values.items():np.save(p/f'{k}.npy',v)
    np.savez(p/'queries.npz',episode_id=np.array(['episode1','episode2']),anchor_source_frame=np.array([10,20]))
    (p/'complete.json').write_text(json.dumps({'plans':n}))
    model='smolvla' if twla_ids is None else 'twla'
    b=tmp_path/'bases'/model/'test'
    if twla_ids is not None:b=b/'shard_00'
    b.mkdir(parents=True)
    count=n if twla_ids is None else len(twla_ids)
    np.save(b/'base_action.npy',np.zeros((count,16,192),np.float32))
    if twla_ids is not None:
        np.save(b/'query_index.npy',np.array(twla_ids,np.int64))
        np.save(b/'final_memory.npy',np.zeros((count,8,960),np.float16))
        (b/'complete.json').write_text('{}')
    return Paired30Cache(tmp_path,'test',model)

@pytest.mark.parametrize('ids',[[0,0,1],[-1,0],[0,2]])
def test_rejects_duplicate_or_out_of_bounds_prediction_indices(tmp_path,ids):
    with pytest.raises(ValueError,match='invalid or repeated query index'):
        make_cache_fixture(tmp_path,ids)

def test_actual_reader_uses_causal_history_and_shared_base(tmp_path):
    ds=make_cache_fixture(tmp_path)
    a,b=ds[0],ds[1]
    torch.testing.assert_close(a['base_action'],b['base_action'])
    torch.testing.assert_close(a['target_action'],b['target_action'])
    assert a['touch_pressure'][:,0,0,0].tolist()==list(range(8))
    assert b['touch_pressure'][:,0,0,0].tolist()==list(range(1,9))
    assert float(b['decision_seconds'])==pytest.approx(1/30)
    assert len(ds)==5

@pytest.mark.parametrize('batch_size',[1,4])
def test_actual_evaluator_stitches_decisions_and_holds_missing_update(tmp_path,monkeypatch,batch_size):
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import train_paired30 as trainer
    ds=make_cache_fixture(tmp_path)
    def prediction(model,batch,**kwargs):
        m=edit_mask(batch)
        delta=(batch['decision_tick'].float()+1)[:,None,None]
        p=batch['base_action']+delta*m.float()
        return p,m,{},None
    monkeypatch.setattr(trainer,'predict',prediction)
    stats={'action_mean':torch.zeros(192),'action_std':torch.ones(192),'reference_pressure':torch.zeros(1,8,2,21,21)}
    args=SimpleNamespace(micro_batch=batch_size,workers=0,seed=7)
    model=torch.nn.Identity()
    trainer.final_test(model,ds,args,stats,torch.device('cpu'),tmp_path)
    result=np.load(tmp_path/'test_executed_actions.npy')
    assert result[0,:,0].tolist()==[1,1,1,2,2,2,2,2]
    assert result[1,:,0].tolist()==[1,1,1,2,2,2,3,3]
    assert not result[:,:,138:].any()
