"""One common (query, decision) row set for all six editor experiments."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class Paired30Cache(Dataset):
    def __init__(self,root,split,model,*,include_rgb=False,statistics=None,require_complete=True):
        self.root=Path(root);self.split=split;self.model=model;self.include_rgb=include_rgb
        common=self.root/'shared'/split
        if require_complete and not (common/'complete.json').is_file():raise RuntimeError(f'incomplete common cache {common}')
        self.common={p.stem:np.load(p,mmap_mode='r') for p in common.glob('*.npy')}
        self.rows=np.argwhere(self.common['eligible'])
        self.queries=dict(np.load(common/'queries.npz',allow_pickle=False))
        n=len(self.queries['episode_id'])
        shapes={'target_action':(n,16,192),'action_mask':(n,16,192),
                'touch_sequence':(n,10,2,21,21),'touch_sequence_valid':(n,10,2),
                'future_touch_pressure':(n,3,2,21,21),'future_touch_valid':(n,3,2),
                'eligible':(n,3),'source_frame':(n,)}
        for name,shape in shapes.items():
            if name not in self.common or self.common[name].shape!=shape:
                raise ValueError(f'invalid common field {name}: expected {shape}')
        self.stats=statistics
        self.bases=[]
        if model=='twla':
            paths=sorted((self.root/'bases/twla'/split).glob('shard_*'))
            for p in paths:
                if require_complete and not (p/'complete.json').is_file():raise RuntimeError(f'incomplete predictions {p}')
                names=['base_action','final_memory','query_index']+(['image_tokens','clean_latent','rgb_valid','future_roi'] if include_rgb else [])
                self.bases.append({k:np.load(p/f'{k}.npy',mmap_mode='r') for k in names})
            n=len(self.common['target_action']);self.locations=np.full((n,2),-1,dtype=np.int64)
            for si,b in enumerate(self.bases):
                ids=b['query_index']
                if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer):
                    raise ValueError('query indices must be a one-dimensional integer array')
                if np.any(ids<0) or np.any(ids>=n) or len(np.unique(ids))!=len(ids):
                    raise ValueError('invalid or repeated query index within shard')
                if any(len(value)!=len(ids) for value in b.values()):
                    raise ValueError('prediction arrays have different row counts')
                if np.any(self.locations[ids,0]>=0):raise ValueError('duplicate query predictions')
                self.locations[ids,0]=si;self.locations[ids,1]=np.arange(len(ids))
            if np.any(self.locations<0):raise ValueError('missing canonical predictions')
        else:
            if model in ('trex','memoryvla') and require_complete:
                marker=self.root/'bases'/model/split/'complete.json'
                if not marker.is_file() or not json.loads(marker.read_text()).get('full_split'):
                    label='T-ReX' if model=='trex' else 'MemoryVLA'
                    raise RuntimeError(f'incomplete {label} predictions {marker.parent}')
            self.bases=[{'base_action':np.load(self.root/'bases'/model/split/'base_action.npy',mmap_mode='r')}]
            if self.bases[0]['base_action'].shape!=(n,16,192):
                raise ValueError('baseline prediction shape does not match common queries')
        self.identity=json.loads((common/'complete.json').read_text()) if (common/'complete.json').exists() else {}

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        qi,tick=map(int,self.rows[index]);c=self.common
        if self.model=='twla':si,bi=self.locations[qi]
        else:si,bi=0,qi
        b=self.bases[si]
        out={k:torch.from_numpy(np.array(v[bi],copy=True)) for k,v in b.items() if k!='query_index'}
        out.update(target_action=torch.from_numpy(np.array(c['target_action'][qi],copy=True)),action_mask=torch.from_numpy(np.array(c['action_mask'][qi],copy=True)),touch_pressure=torch.from_numpy(np.array(c['touch_sequence'][qi,tick:tick+8],copy=True)).float(),touch_valid=torch.from_numpy(np.array(c['touch_sequence_valid'][qi,tick:tick+8],copy=True)),future_touch_pressure=torch.from_numpy(np.array(c['future_touch_pressure'][qi,tick],copy=True)).float(),future_touch_valid=torch.from_numpy(np.array(c['future_touch_valid'][qi,tick],copy=True)),decision_seconds=torch.tensor(tick/30,dtype=torch.float32),query_index=torch.tensor(qi),decision_tick=torch.tensor(tick))
        if self.stats is not None:
            for k in ('base_action','target_action'):out[k]=(out[k].float()-self.stats['action_mean'])/self.stats['action_std']
        return out

def edit_mask(batch):
    from .training import build_future_action_mask
    mask=build_future_action_mask(batch['action_mask'],batch['decision_seconds'])
    mask[...,138:]=False
    latest=batch['touch_valid'][:,-1].bool()
    mask[...,:69]&=latest[:,0,None,None]
    mask[...,69:138]&=latest[:,1,None,None]
    return mask
