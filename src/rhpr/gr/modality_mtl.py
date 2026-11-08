"""Reconstruction-task UW, GradNorm (Chen et al. 2018), PCGrad (Yu et al. 2020).

PCGrad uses random-order pair projections, sums shared projected gradients
and sums task-private gradients. Quantization gradients are added unprojected.
GradNorm reuses our MIT LibMTL adaptation; UW uses exp(-s)*MSE+s.
"""
import random
import torch
from torch import nn
from rhpr.gr.vendor.libmtl_balancing import gradnorm_objective


class ModalityMTL(nn.Module):
    def __init__(self, mode, tasks, seed=42):
        super().__init__()
        if mode not in {'sum', 'uw', 'gradnorm', 'pcgrad'}:
            raise ValueError('unknown modality MTL mode')
        self.mode=mode; self.tasks=tuple(tasks); self.rng=random.Random(seed)
        self.values=nn.Parameter(torch.zeros(len(tasks)), requires_grad=mode in {'uw','gradnorm'})
        self.register_buffer('reference',torch.zeros(len(tasks)))
        self.diagnostics={}; self.calls=0; self.phase="ae"; self.history=[]; self.total_projections=0; self.projected_updates=0

    def weights(self):
        if self.mode=='uw':return torch.exp(-self.values)
        if self.mode=='gradnorm':return len(self.tasks)*torch.softmax(self.values,dim=0)
        return torch.ones_like(self.values)

    def objective(self, losses):
        v=torch.stack([losses[k] for k in self.tasks])
        if self.mode=='uw':return (self.weights()*v+self.values).sum()
        return (self.weights().detach()*v).sum()

    def reset_phase(self):
        self.reference.zero_(); self.phase="rq"

    def backward_tasks(self, losses, total, parameters, shared_weight):
        parameters=list(parameters)
        probe={}
        if (self.calls+1)%500==0:
            gs=torch.stack([torch.autograd.grad(losses[k],shared_weight,retain_graph=True)[0].detach().flatten() for k in self.tasks])
            norms=gs.norm(dim=1); cos=(gs@gs.T)/(norms[:,None]*norms[None,:]).clamp_min(1e-12)
            probe={'shared_encoder_gradient_norms':norms.tolist(),'shared_encoder_gradient_cosines':cos.tolist()}

        if self.mode=='gradnorm':
            vals=torch.stack([losses[k] for k in self.tasks])
            if not bool((self.reference>0).all()):self.reference.copy_(vals.detach().clamp_min(1e-12))
            norms=torch.stack([torch.autograd.grad(v,shared_weight,retain_graph=True)[0].detach().norm() for v in vals])
            aux,_=gradnorm_objective(self.weights(),norms,vals/self.reference,1.5)
            wg=torch.autograd.grad(aux,self.values)[0]
            total.backward();self.values.grad=wg
            self.diagnostics={'gradient_norms':norms.tolist()}
        elif self.mode=='pcgrad':
            task=[losses[k] for k in self.tasks]
            # The remainder includes only the fixed-weight quantization objective.
            rest=total-sum(task)
            gradients=[torch.autograd.grad(v,parameters,retain_graph=True,allow_unused=True) for v in task]
            extra=torch.autograd.grad(rest,parameters,allow_unused=True)
            masks=torch.stack([torch.cat([torch.full_like(p,g is not None,dtype=torch.bool).flatten() for p,g in zip(parameters,gs)]) for gs in gradients])
            raw=torch.stack([torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(parameters,gs)]) for gs in gradients])
            shared=masks.all(0); projected=raw.clone(); conflicts=0
            for i in range(len(task)):
                order=list(range(len(task)));self.rng.shuffle(order)
                for j in order:
                    if i==j:continue
                    dot=(projected[i,shared]*raw[j,shared]).sum()
                    if dot<0:
                        projected[i,shared]-=dot/raw[j,shared].square().sum().clamp_min(1e-12)*raw[j,shared]
                        conflicts+=1
            merged=raw.sum(0);merged[shared]=projected[:,shared].sum(0)
            start=0
            for p,g in zip(parameters,extra):
                value=merged[start:start+p.numel()].view_as(p);start+=p.numel()
                p.grad=value.clone() if g is None else value+g
            self.total_projections+=conflicts; self.projected_updates+=int(conflicts>0)
            self.diagnostics={'negative_projection_count':conflicts,'total_projections':self.total_projections,'projected_updates':self.projected_updates}
        else:total.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters) or (self.values.grad is not None and not torch.isfinite(self.values.grad).all()):
            raise FloatingPointError('nonfinite modality MTL gradient')
        self.calls+=1
        if self.calls%500==0:
            self.diagnostics.update(probe)
            self.diagnostics.update(training_stage=self.phase,step=self.calls,mode=self.mode,weights=self.weights().detach().tolist(),losses={k:float(v.detach()) for k,v in losses.items()})
            self.history.append(dict(self.diagnostics))
            from rhpr.gr.experiment import _event
            _event('modality_mtl',**self.diagnostics)

    def get_extra_state(self):return {'rng':self.rng.getstate(),'calls':self.calls,'total_projections':self.total_projections,'projected_updates':self.projected_updates}
    def set_extra_state(self,state):self.rng.setstate(state['rng']);self.calls=state['calls'];self.total_projections=state.get('total_projections',0);self.projected_updates=state.get('projected_updates',0)
