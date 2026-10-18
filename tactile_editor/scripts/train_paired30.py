"""Train comparable editors on the same offline 10 Hz / 30 Hz replay contract."""
import argparse, math, time
from paired30_common import *
import torch
from tactile_editor.paired30_cache import Paired30Cache,edit_mask
from tactile_editor.direct_residual import DirectTactileResidualEditor
from tactile_editor.encoder import DualHandTactileEncoder,load_formal_t2p_encoder
from tactile_editor.unified_world import UnifiedWorldTactileEditor,WorldMode
from tactile_editor.action_objectives import per_sample_action_loss
from tactile_editor.action_layout import FULL_ACTION_GROUPS
from tactile_editor.rgb_world_editor import roi_weighted_world_loss
from tactile_editor.single_stage_world import masked_tactile_flow_loss

STANDARD_VARIANTS=('smolvla','vitra','trex','memoryvla')
PUBLIC_VARIANTS=('w3',*STANDARD_VARIANTS)

def cache_model(variant):
    if variant not in PUBLIC_VARIANTS:raise ValueError(f'Unsupported editor: {variant}')
    return variant if variant in STANDARD_VARIANTS else 'twla'

def make_model(variant,stats,device,encoder_checkpoint=ENCODER):
    cache_model(variant)
    common=dict(action_mean=torch.zeros(192),action_std=torch.ones(192),pressure_masks=stats['pressure_masks'],reference_pressure=stats['reference_pressure'],articulation_only=False)
    if variant in STANDARD_VARIANTS:
        model=DirectTactileResidualEditor(**common,mode='history',residual_bases=8)
    else:
        enc=DualHandTactileEncoder(load_formal_t2p_encoder(encoder_checkpoint,history_frames=8),freeze=False)
        model=UnifiedWorldTactileEditor(enc,**common,world_mode=WorldMode.W3_BOTH,pressure_scale=stats['pressure_scale'])
    return model.to(device)

def predict(model,batch,*,world_head=None,aux=False,noise_generator=None,compute_rgb=True,compute_tactile=True):
    mask=edit_mask(batch)
    if isinstance(model,DirectTactileResidualEditor):
        pred,detail=model(batch['base_action'],mask,batch['touch_pressure'],batch['touch_valid'],batch['decision_seconds'])
        return pred,mask,detail,None
    kwargs={}
    if aux and compute_tactile and model.world_mode.uses_tactile:
        n=len(mask)
        kwargs.update(future_pressure=batch['future_touch_pressure'],future_valid=batch['future_touch_valid'],tactile_tau=torch.rand(n,device=mask.device,generator=noise_generator),tactile_noise=torch.randn(n,2,441,device=mask.device,generator=noise_generator))
    if aux and compute_rgb and model.world_mode.uses_rgb:
        latent=batch['clean_latent'].float()
        kwargs.update(rgb_world_head=world_head,image_tokens=batch['image_tokens'].float(),clean_latent=latent,rgb_valid=batch['rgb_valid'],rgb_tau=torch.rand(latent.shape[:2],device=mask.device,generator=noise_generator),rgb_noise=torch.randn(latent.shape,device=mask.device,generator=noise_generator))
    out=model(base_action=batch['base_action'],edit_mask=mask,pressure=batch['touch_pressure'],temporal_valid=batch['touch_valid'],decision_seconds=batch['decision_seconds'],final_memory=batch['final_memory'].float(),compute_auxiliary=aux,compute_rgb=compute_rgb,compute_tactile=compute_tactile,**kwargs)
    return out.final_action,mask,{'effective_delta':out.delta_action},out

def objective(pred,mask,detail,out,target,weight,*,return_terms=False):
    action=per_sample_action_loss(pred,target,mask,groups=FULL_ACTION_GROUPS).mean()
    delta=detail['effective_delta'].float().square().masked_fill(~mask,0).sum()/mask.sum().clamp_min(1)
    rgb=action.new_zeros(());touch=action.new_zeros(())
    if out is not None and out.tactile_velocity is not None:
        touch=masked_tactile_flow_loss(out.tactile_velocity,out.tactile_target,out.tactile_valid)
    # Missing future labels do not remove valid action supervision.
    if out is not None and out.rgb is not None and out.rgb.valid.any():
        rgb=roi_weighted_world_loss(out.rgb,weight['roi'],background_weight=weight.get('background_weight',.1))
    coeff=detail.get('coefficients')
    loss=action+1e-4*delta+weight['rgb']*rgb+weight['tactile']*touch
    if coeff is not None:loss=loss+1e-5*coeff.float().square().mean()
    parts={'action':float(action.detach()),'rgb':float(rgb.detach()),'tactile':float(touch.detach()),'delta_l2':float(delta.detach())}
    if return_terms:return loss,parts,{'action':action,'rgb':rgb,'tactile':touch}
    return loss,parts

def resolve_world_weights(variant,rgb=None,tactile=None):
    cache_model(variant)
    uses_rgb=uses_touch=variant=='w3'
    rgb=(.1 if uses_rgb else 0.) if rgb is None else rgb
    tactile=(1e-4 if uses_touch else 0.) if tactile is None else tactile
    if any(not math.isfinite(x) or x<0 for x in (rgb,tactile)):
        raise ValueError('world weights must be finite and nonnegative')
    if (rgb and not uses_rgb) or (tactile and not uses_touch):
        raise ValueError('nonzero world weight requires the corresponding branch')
    return rgb,tactile

def feature_gradient_diagnostic(terms,features,weights):
    """Measure gradients at the shared tokens without accumulating parameter gradients."""
    gradients={}
    for name,term in terms.items():
        gradient=torch.autograd.grad(term,features,retain_graph=True,allow_unused=True)[0] if term.requires_grad else None
        gradients[name]=torch.zeros_like(features,dtype=torch.float32) if gradient is None else gradient.detach().float()
    action=gradients['action'];aux=weights['rgb']*gradients['rgb']+weights['tactile']*gradients['tactile']
    an=float(action.norm());wn=float(aux.norm())
    return {'action_norm':an,'rgb_norm':float(gradients['rgb'].norm()),
            'tactile_norm':float(gradients['tactile'].norm()),'weighted_aux_norm':wn,
            'aux_to_action_ratio':wn/an if an>0 else None,
            'action_aux_cosine':float((action*aux).sum())/(an*wn) if an>0 and wn>0 else None}

def validate_resume_config(saved,current):
    # These fields determine training semantics. Output location and diagnostics do not.
    keys=('variant','epochs','micro_batch','effective_batch','seed','lambda_rgb','lambda_tactile',
          'train_rows','validation_rows','query_identity','world_weight_warmup','action_lr','encoder_lr')
    for key in keys:
        saved_value=saved.get(key)
        if key=='variant' and saved_value=='w3_both':saved_value='w3'
        if saved_value!=current.get(key):raise ValueError(f'resume configuration mismatch: {key}')
    for key,default in dict(aux_delay_epochs=0,aux_ramp_steps=400,encoder_freeze_epochs=0,roi_background=.1,validation_only=False).items():
        if saved.get(key,default)!=current.get(key,default):raise ValueError(f'resume configuration mismatch: {key}')
    if saved.get('wm_action_gradient',0.)!=current.get('wm_action_gradient',0.):
        raise ValueError('resume configuration mismatch: wm_action_gradient')

def move(batch,device):return {k:v.to(device,non_blocking=True) for k,v in batch.items()}

@torch.no_grad()
def validate(model,dataset,batch_size,workers,device):
    model.eval();total=base_total=0.;count=0;by_tick=np.zeros((3,3))
    loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,num_workers=workers,pin_memory=True)
    for b in loader:
        b=move(b,device)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred,m,_,_=predict(model,b)
        losses=per_sample_action_loss(pred,b['target_action'],m,groups=FULL_ACTION_GROUPS)
        base=per_sample_action_loss(b['base_action'],b['target_action'],m,groups=FULL_ACTION_GROUPS)
        total+=float(losses.sum());base_total+=float(base.sum());count+=len(losses)
        for tick in range(3):
            sel=b['decision_tick']==tick;by_tick[tick]+=[float(losses[sel].sum()),float(base[sel].sum()),int(sel.sum())]
    return {'action_loss':total/max(count,1),'base_action_loss':base_total/max(count,1),'samples':count,'by_tick':{str(t):{'action_loss':float(x[0]/max(x[2],1)),'base_action_loss':float(x[1]/max(x[2],1)),'samples':int(x[2])} for t,x in enumerate(by_tick)}}

def base_plans(dataset):
    n=len(dataset.common['target_action']);result=np.empty((n,16,192),np.float32)
    if dataset.model!='twla':result[:]=dataset.bases[0]['base_action']
    else:
        for b in dataset.bases:result[b['query_index']]=b['base_action']
    return result

def physical_metrics(pred,true,mask):
    # Accumulated wrist position/rotation; absolute finger MPJPE excludes palm origin.
    from scipy.spatial.transform import Rotation
    sums=np.zeros(3);counts=np.zeros(3)
    for offset in (0,69):
        wrist_valid=mask[:,:,offset:offset+3].all(2)
        err=np.linalg.norm(np.cumsum(pred[:,:,offset:offset+3]-true[:,:,offset:offset+3],axis=1),axis=2)*100
        sums[0]+=err[wrist_valid].sum();counts[0]+=wrist_valid.sum()
        p=Rotation.from_euler('xyz',pred[:,:,offset+3:offset+6].reshape(-1,3)).as_matrix().reshape(len(pred),-1,3,3)
        t=Rotation.from_euler('xyz',true[:,:,offset+3:offset+6].reshape(-1,3)).as_matrix().reshape(len(pred),-1,3,3)
        ap=np.broadcast_to(np.eye(3),(len(pred),3,3)).copy();at=ap.copy()
        for k in range(pred.shape[1]):
            ap=p[:,k]@ap;at=t[:,k]@at
            angle=np.rad2deg(np.arccos(np.clip((np.trace(ap@at.transpose(0,2,1),axis1=1,axis2=2)-1)/2,-1,1)))
            ok=mask[:,k,offset+3:offset+6].all(1);sums[1]+=angle[ok].sum();counts[1]+=ok.sum()
        d=(pred[:,:,offset+6:offset+69]-true[:,:,offset+6:offset+69]).reshape(len(pred),pred.shape[1],21,3)[:,:,1:]
        e=np.linalg.norm(d,axis=-1).mean(-1)*100;ok=mask[:,:,offset+9:offset+69].all(-1)
        sums[2]+=e[ok].sum();counts[2]+=ok.sum()
    return dict(zip(('wrist_position_cm','wrist_rotation_deg','finger_mpjpe_cm'),(sums/np.maximum(counts,1)).tolist()))

@torch.no_grad()
def final_test(model,dataset,args,stats,device,out):
    model.eval();base=base_plans(dataset);executed=base[:,:8].copy();shuffled=executed.copy();reference=executed.copy()
    std=stats['action_std'].numpy();mean=stats['action_mean'].numpy()
    loader=torch.utils.data.DataLoader(dataset,batch_size=args.micro_batch,num_workers=args.workers,pin_memory=True)
    # Fixed global, hand-validity-stratified permutation, not a within-batch roll.
    eligible=np.asarray(dataset.common['eligible']);hv=dataset.common['touch_sequence_valid']
    permutation=np.tile(np.arange(len(base))[:,None],(1,3))
    rng=np.random.default_rng(args.seed+700)
    for tick in range(3):
        code=hv[:,7+tick,0].astype(int)+2*hv[:,7+tick,1].astype(int)
        for side in (1,2,3):
            ids=np.flatnonzero(eligible[:,tick]&(code==side));draw=ids.copy();rng.shuffle(draw);permutation[ids,tick]=draw
    for raw in loader:
        b=move(raw,device);qi=raw['query_index'].numpy();ticks=raw['decision_tick'].numpy()
        with torch.autocast('cuda',dtype=torch.bfloat16):p,m,_,_=predict(model,b)
        pp=p.float().cpu().numpy()*std+mean;mm=m.cpu().numpy()
        other=[];other_valid=[]
        for i,tick in zip(qi,ticks):
            j=permutation[i,tick];other.append(np.array(dataset.common['touch_sequence'][j,tick:tick+8]));other_valid.append(np.array(dataset.common['touch_sequence_valid'][j,tick:tick+8]))
        b['touch_pressure']=torch.from_numpy(np.stack(other)).to(device).float();b['touch_valid']=torch.from_numpy(np.stack(other_valid)).to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):sp,_,_,_=predict(model,b)
        sp=sp.float().cpu().numpy()*std+mean
        b['touch_pressure']=stats['reference_pressure'].to(device).expand(len(qi),-1,-1,-1,-1)
        b['touch_valid']=raw['touch_valid'].to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):rp,_,_,_=predict(model,b)
        rp=rp.float().cpu().numpy()*std+mean
        # Process ticks in order; later decisions replace only uncommitted positions.
        for tick in range(3):
            sel=ticks==tick;ids=qi[sel];mask=mm[sel,:8]
            executed[ids]=np.where(mask,pp[sel,:8],executed[ids])
            shuffled[ids]=np.where(mask,sp[sel,:8],shuffled[ids])
            reference[ids]=np.where(mask,rp[sel,:8],reference[ids])
    target=np.asarray(dataset.common['target_action'][:,:8]);mask=np.asarray(dataset.common['action_mask'][:,:8])
    metrics={name:physical_metrics(p,target,mask) for name,p in [('base',base[:,:8]),('editor',executed),('shuffled_touch',shuffled),('reference_touch',reference)]}
    np.save(out/'test_executed_actions.npy',executed);np.save(out/'test_shuffled_actions.npy',shuffled)
    return {'execution_horizon':'first 8 actions / 100ms; latest available correction; zero compute latency','plans':len(base),'metrics':metrics,'remaining_horizon_loss':validate(model,dataset,args.micro_batch,args.workers,device)}

def main():
    ap=argparse.ArgumentParser(description='Train W3 or a standard baseline editor. Set world weights to zero for ablations.');ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--variant',choices=PUBLIC_VARIANTS,default='w3');ap.add_argument('--epochs',type=int,default=5);ap.add_argument('--micro-batch',type=int,default=64);ap.add_argument('--effective-batch',type=int,default=256);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--seed',type=int,default=20260921);ap.add_argument('--smoke-steps',type=int,default=0);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--encoder-checkpoint',type=Path,default=ENCODER)
    ap.add_argument('--world-checkpoint',type=Path,default=WORLD)
    ap.add_argument('--world-config',type=Path,help='Configuration that matches the RGB world checkpoint')
    ap.add_argument('--lambda-rgb',type=float)
    ap.add_argument('--lambda-tactile',type=float)
    ap.add_argument('--diagnostic-every',type=int,default=0)
    ap.add_argument('--smoke-full-weights',action='store_true')
    ap.add_argument('--action-lr',type=float,default=1e-4)
    ap.add_argument('--encoder-lr',type=float,default=1e-5)
    ap.add_argument('--aux-delay-epochs',type=int,default=0)
    ap.add_argument('--aux-ramp-steps',type=int,default=400)
    ap.add_argument('--encoder-freeze-epochs',type=int,default=0)
    ap.add_argument('--roi-background',type=float,default=.1)
    ap.add_argument('--wm-action-gradient',type=float,default=0.,help='Backward scale from tactile WM to corrected actions; forward values are unchanged')
    ap.add_argument('--validation-only',action='store_true',help='Do not open test data or write test predictions')
    a=ap.parse_args()
    if a.smoke_full_weights and not a.smoke_steps:raise ValueError('Full-weight smoke mode requires --smoke-steps')
    lambda_rgb,lambda_tactile=resolve_world_weights(a.variant,a.lambda_rgb,a.lambda_tactile)
    if not 0<=a.wm_action_gradient<=1:raise ValueError('wm-action-gradient must be in [0,1]')
    if a.wm_action_gradient and (a.variant!='w3' or lambda_tactile==0):
        raise ValueError('action gradient requires a tactile WM')
    if not (0<=a.roi_background<=1) or a.aux_delay_epochs<0 or a.encoder_freeze_epochs<0 or a.aux_ramp_steps<1:
        raise ValueError('invalid auxiliary training schedule')
    if not all(math.isfinite(v) and v>0 for v in (a.action_lr,a.encoder_lr)):
        raise ValueError('learning rates must be finite and positive')
    if a.epochs<1 or a.micro_batch<1 or a.effective_batch<1 or a.diagnostic_every<0:raise ValueError('invalid training budget')
    if a.effective_batch%a.micro_batch:raise ValueError('effective batch must divide into micro batches')
    torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_float32_matmul_precision('high');dev=torch.device('cuda')
    out=a.output_dir or a.root/('smoke_runs' if a.smoke_steps else 'runs')/a.variant
    if out.exists() and not a.resume:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    stats=torch.load(a.root/'statistics.pt',map_location='cpu',weights_only=False)
    base_model=cache_model(a.variant);rgb=lambda_rgb>0
    train=Paired30Cache(a.root,'train',base_model,include_rgb=rgb,statistics=stats)
    val=Paired30Cache(a.root,'val',base_model,statistics=stats)
    model=make_model(a.variant,stats,dev,a.encoder_checkpoint)
    if isinstance(model,UnifiedWorldTactileEditor) and model.tactile_world is not None:
        model.tactile_world.adapter.action_gradient_scale=a.wm_action_gradient
    if isinstance(model,UnifiedWorldTactileEditor):groups=model.optimizer_groups(action_lr=a.action_lr,encoder_lr=a.encoder_lr)
    else:groups=[{'params':model.parameters(),'lr':a.action_lr,'name':'standard_editor'}]
    optim=torch.optim.AdamW(groups,weight_decay=.01)
    head=None
    if rgb:
        from tactile_editor.public_backbone import load_rgb_world_head
        if a.world_config is None:ap.error('RGB supervision requires --world-config')
        head,_=load_rgb_world_head(a.world_checkpoint,a.world_config,dev)
    steps_per_epoch=math.ceil(len(train)/a.effective_batch);total=a.epochs*steps_per_epoch
    scheduler=torch.optim.lr_scheduler.LambdaLR(optim,lambda s:min((s+1)/400,1) if s<400 else .1+.45*(1+math.cos(math.pi*min((s-400)/max(total-400,1),1))))
    config={**vars(a),'train_rows':len(train),'validation_rows':len(val),'query_identity':train.identity,'action_scaling':'shared train-GT mean/std','lambda_rgb':lambda_rgb,'lambda_tactile':lambda_tactile,'world_weight_warmup':a.aux_ramp_steps,'seed_note':'same row shuffle and effective batch for runs with the same seed','test_exposure':'known upstream ten-domain exposure; development results'}
    if a.variant in ('trex','memoryvla'):
        config.update(test_exposure='See upstream_split_audit.json; cross-backbone comparison remains a development result.',
                      trainable_components=f'standard editor including randomly initialized CNN/GRU tactile trunk; cached {a.variant} frozen; no memory tokens enter editor',
                      pretrained_retrieval_encoder=False)
    step=0;first_epoch=0;best=float('inf')
    if a.resume and (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',map_location=dev,weights_only=False);validate_resume_config(saved['config'],config);model.load_state_dict(saved['model']);optim.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);step=saved['step'];first_epoch=saved['epoch'];best=saved['best']
    save_json(out/'config.json',config)
    generator=torch.Generator(device=dev).manual_seed(a.seed+123)
    if a.resume and (out/'last.pt').exists() and 'rng' in saved:
        torch.set_rng_state(saved['rng']['cpu'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in saved['rng']['cuda']]);generator.set_state(saved['rng']['noise'].cpu())
    started=time.time()
    for epoch in range(first_epoch,a.epochs):
        model.train();sums=np.zeros(4);seen=0
        loader=torch.utils.data.DataLoader(train,batch_size=a.micro_batch,shuffle=True,generator=torch.Generator().manual_seed(a.seed+epoch),num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)
        optim.zero_grad(set_to_none=True);window_count=0
        for bi,raw in enumerate(loader):
            b=move(raw,dev)
            ramp=1. if a.smoke_full_weights else min(max((step+1-a.aux_delay_epochs*steps_per_epoch)/a.aux_ramp_steps,0),1)
            weight={'rgb':config['lambda_rgb']*ramp,'tactile':config['lambda_tactile']*ramp,'roi':b.get('future_roi'),'background_weight':a.roi_background}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred,mask,detail,wm=predict(model,b,world_head=head,aux=True,noise_generator=generator,compute_rgb=weight['rgb']>0,compute_tactile=weight['tactile']>0)
                loss,parts,terms=objective(pred,mask,detail,wm,b['target_action'],weight,return_terms=True)
            if a.diagnostic_every and step%a.diagnostic_every==0 and window_count==0 and wm is not None:
                diagnostic={'step':step,'epoch':epoch+1,'weights':{k:weight[k] for k in ('rgb','tactile')},
                            **feature_gradient_diagnostic(terms,wm.tactile_features,weight)}
                if terms['tactile'].requires_grad:
                    action_gradient=torch.autograd.grad(terms['tactile'],pred,retain_graph=True,allow_unused=True)[0]
                    diagnostic['tactile_to_action_norm']=0. if action_gradient is None else float(action_gradient.float().norm())
                with (out/'feature_gradients.jsonl').open('a') as f:f.write(json.dumps(diagnostic)+'\n')
            if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite step {step}')
            count=len(mask);(loss*count/a.effective_batch).backward();window_count+=count;seen+=count;sums+=np.array(list(parts.values()))*count
            if (bi+1)%(a.effective_batch//a.micro_batch)==0 or bi+1==len(loader):
                if window_count!=a.effective_batch:
                    for p in model.parameters():
                        if p.grad is not None:p.grad.mul_(a.effective_batch/window_count)
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                if epoch<a.encoder_freeze_epochs:
                    for group in optim.param_groups:
                        if group.get('name')=='tactile.encoder':group['lr']=0.
                optim.step();scheduler.step();optim.zero_grad(set_to_none=True);window_count=0;step+=1
                if step%100==0 or a.smoke_steps:log(variant=a.variant,epoch=epoch+1,step=step,total_steps=total,loss=parts,grad=float(grad),seconds=time.time()-started)
                if a.smoke_steps and step>=a.smoke_steps:
                    save_json(out/'smoke.json',{'steps':step,'finite':True,'loss':parts,'grad':float(grad)});return
        validation=validate(model,val,a.micro_batch,a.workers,dev)
        improved=validation['action_loss']<best
        if improved:best=validation['action_loss']
        checkpoint={'model':model.state_dict(),'optimizer':optim.state_dict(),'scheduler':scheduler.state_dict(),'step':step,'epoch':epoch+1,'best':best,'config':config,'statistics':stats,
                    'rng':{'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'noise':generator.get_state()}}
        tmp=out/'last.tmp.pt';torch.save(checkpoint,tmp);tmp.replace(out/'last.pt')
        if improved:
            torch.save({'model':model.state_dict(),'step':step,'epoch':epoch+1,'validation':validation,'config':config,'statistics':stats},out/'best.pt')
        record={'epoch':epoch+1,'step':step,'train':dict(zip(('action','rgb','tactile','delta_l2'),(sums/max(seen,1)).tolist())),'val':validation,'seconds':time.time()-started}
        with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        log(variant=a.variant,**record)
    saved=torch.load(out/'best.pt',map_location=dev,weights_only=False);model.load_state_dict(saved['model'])
    if a.validation_only:
        save_json(out/'validation_summary.json',{'variant':a.variant,'selected_epoch':saved['epoch'],'selected_step':saved['step'],'validation':saved['validation'],'config':config,'seconds':time.time()-started,'test_evaluated':False})
        return
    test=Paired30Cache(a.root,'test',base_model,statistics=stats)
    results=final_test(model,test,a,stats,dev,out)
    save_json(out/'summary.json',{'variant':a.variant,'selected_epoch':saved['epoch'],'selected_step':saved['step'],'validation':saved['validation'],'test':results,'seconds':time.time()-started,'config':config})
    log(complete=a.variant,test=results)

if __name__=='__main__':main()
