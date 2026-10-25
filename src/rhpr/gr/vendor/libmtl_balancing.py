"""DWA/GradNorm adapted from LibMTL (MIT), revision4336804847eaa5e0b924b743d76beec7ac3fdc97.

See vendor/libmtl/NOTICE.md and LICENSE. Adaptation uses catalogue passes,
observed-item means, native Adagrad and the last shared decoder weight.
"""
import torch
from torch import nn


def dwa_weights(previous, recent, temperature=2.0):
    ratio = recent / previous.clamp_min(1e-12)
    return len(ratio) * torch.softmax(ratio / temperature, dim=-1)


def gradnorm_objective(weights, norms, loss_ratios, alpha=1.5):
    weighted_norms = weights * norms.detach()
    rates = loss_ratios.detach() / loss_ratios.detach().mean().clamp_min(1e-12)
    targets = (weighted_norms.mean() * rates.pow(alpha)).detach()
    return (weighted_norms - targets).abs().sum(), targets


class TaskBalancer(nn.Module):
    def __init__(self, tasks, policy):
        super().__init__()
        self.policy = dict(policy)
        if policy['mode'] == 'gradnorm':
            self.logits = nn.Parameter(torch.ones(tasks))
        for name in ['weights', 'reference', 'previous', 'recent', 'sums', 'counts']:
            self.register_buffer(name, torch.ones(tasks) if name == 'weights' else torch.zeros(tasks))
        self.register_buffer('passes', torch.tensor(0, dtype=torch.long))
        self.diagnostics = {}

    def multipliers(self):
        if self.policy['mode'] == 'gradnorm' and int(self.passes) > 0:
            return len(self.logits) * torch.softmax(self.logits, dim=-1)
        return self.weights

    def backward_weights(self, normalized_losses, shared_weight, counts):
        if self.policy['mode'] != 'gradnorm' or int(self.passes) == 0:
            return
        # Never infer a learning rate from a missing target. All four tasks are
        # observed in the actual large batches; skip only the weighting update
        # for an incomplete set. Neural loss still uses its observed-task mask.
        if not all(counts):
            self.diagnostics = {'weight_update_skipped_missing_task': True}
            return
        norms = torch.stack([torch.autograd.grad(loss, shared_weight, retain_graph=True)[0].detach().norm()
                             for loss in normalized_losses])
        ratios = torch.stack(tuple(normalized_losses)).detach() / self.reference.clamp_min(1e-12)
        auxiliary, targets = gradnorm_objective(self.multipliers(), norms, ratios, self.policy.get('alpha', 1.5))
        self.logits.grad = torch.autograd.grad(auxiliary, self.logits)[0]
        if not torch.isfinite(self.logits.grad).all():
            raise FloatingPointError('nonfinite GradNorm weight gradient')
        self.diagnostics = {'gradient_norms': norms.cpu().tolist(), 'target_norms': targets.cpu().tolist(),
                            'auxiliary_loss': float(auxiliary.detach())}

    @torch.no_grad()
    def observe(self, normalized_losses, counts):
        counts = self.counts.new_tensor(counts)
        self.sums.add_(torch.stack(tuple(normalized_losses)).detach() * counts)
        self.counts.add_(counts)

    @torch.no_grad()
    def finish_pass(self):
        if not bool((self.counts > 0).all()):
            raise ValueError('adaptive balancing needs every task observed in a catalogue pass')
        means = self.sums / self.counts
        if not torch.isfinite(means).all():
            raise FloatingPointError('nonfinite adaptive balancing pass means')
        if int(self.passes) == 0:
            self.reference.copy_(means)
        self.previous.copy_(self.recent)
        self.recent.copy_(means)
        self.passes.add_(1)
        if self.policy['mode'] == 'dwa' and int(self.passes) >= 2:
            self.weights.copy_(dwa_weights(self.previous, self.recent, self.policy.get('temperature', 2.0)))
        self.sums.zero_();self.counts.zero_()
