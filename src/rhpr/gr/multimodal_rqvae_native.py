"""Per-modality MLP heads around the pinned GRID residual quantizer."""
import math
import torch
from torch import nn
from lightning.pytorch.trainer.states import TrainerFn
from src.modules.clustering.residual_quantization import ResidualQuantization


class MultimodalResidualQuantization(ResidualQuantization):
    def __init__(self, modality_schema, active_modalities, head_dim=128, head_hidden=256, reconstruction_weights=None, reconstruction_columns=None, **kwargs):
        super().__init__(**kwargs)
        self.modality_schema = dict(modality_schema)
        self.active_modalities = tuple(active_modalities)
        weights = dict(reconstruction_weights or {})
        if (set(weights)-set(modality_schema) or any(isinstance(v,bool) or not isinstance(v,(int,float))
                or not math.isfinite(v) or v<0 for v in weights.values())):
            raise ValueError('modality reconstruction weights must be finite nonnegative known-channel values')
        self.reconstruction_weights = {name:float(weights.get(name,1.)) for name in modality_schema}
        if not any(self.reconstruction_weights[name]>0 for name in self.active_modalities):
            raise ValueError('at least one active reconstruction weight must be positive')
        self.reconstruction_columns = dict(reconstruction_columns or {})
        for name, columns in self.reconstruction_columns.items():
            if (name not in modality_schema or not columns or len(set(columns)) != len(columns)
                    or any(type(c) is not int or c < 0 or c >= modality_schema[name] for c in columns)):
                raise ValueError("invalid reconstruction columns")
        self.encoder_heads = nn.ModuleDict({name: nn.Sequential(
            nn.Linear(dim, head_hidden), nn.ReLU(), nn.Linear(head_hidden, head_dim))
            for name, dim in modality_schema.items()})
        self.decoder_heads = nn.ModuleDict({name: nn.Sequential(
            nn.Linear(self.decoder.output_dim, head_hidden), nn.ReLU(), nn.Linear(head_hidden, dim))
            for name, dim in modality_schema.items()})
        self.head_dim = head_dim
        self.last_head_losses = {}
        self.warmup_transferred = False
        self.reconstruction_updates = 0
        self.initialization_batches = 0
        self.initial_latent_audit = None

    def on_train_batch_start(self, batch, batch_idx):
        if self.warmup_transferred:
            from rhpr.gr.multimodal_warmup import NEURAL_MODULES
            for name in NEURAL_MODULES:
                getattr(self,name).train(self.training and self.reconstruction_updates>0)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.last_head_losses:self.reconstruction_updates+=1
        else:self.initialization_batches+=1

    def encode_modalities(self, fields):
        first = fields[next(iter(self.modality_schema))]
        blocks = []
        for name in self.modality_schema:
            if name in self.active_modalities:
                # Mask after the MLP as well: missing inputs must not introduce
                # modality-specific head biases, nor access inactive channels.
                block = self.encoder_heads[name](fields[name])
                block = block * fields[name + '_observed'][:, None]
            else:
                block = first.new_zeros((len(first), self.head_dim))
            blocks.append(block)
        return self.encoder(self.normalization_layer(torch.cat(blocks, dim=-1)))

    def reconstruction(self, quantized, fields):
        shared = self.decoder(quantized)
        losses = {}
        for name in self.active_modalities:
            valid = fields[name + '_observed'].bool()
            prediction = self.decoder_heads[name](shared)
            target = fields[name]
            if name in self.reconstruction_columns:
                columns = self.reconstruction_columns[name]
                prediction, target = prediction[:, columns], target[:, columns]
            losses[name] = ((prediction[valid] - target[valid]).square().mean()
                            if valid.any() else prediction.sum() * 0.)
        # Preserve the historical operation order for the equal-weight control.
        total = (sum(losses.values()) if all(self.reconstruction_weights[k]==1 for k in losses)
                 else sum(self.reconstruction_weights[k]*v for k,v in losses.items()))
        if hasattr(self, 'mtl'):
            total = self.mtl.objective(losses)
            if self.training and torch.is_grad_enabled():self.mtl_losses = losses
        return total, losses

    def model_step(self, model_input):
        if hasattr(self,'mtl_losses'):del self.mtl_losses
        fields = {k: v.to(self.device) for k, v in model_input.transformed_features.items()}
        encoded = self.encode_modalities(fields)
        if self.warmup_transferred and self.initial_latent_audit is None:
            anchor=self.ae_anchor_latents[fields['_item_indices'].long()]
            drift=float((encoded.detach()-anchor).abs().max())
            if drift>1e-5:raise RuntimeError('warm-up latents changed before native book initialization')
            self.initial_latent_audit={'max_absolute_difference':drift,'items':len(encoded)}
        codes, residuals, quantized, qloss = self(encoded)
        self.last_head_losses = {}
        reconstruction = qloss.new_zeros(())
        if self.trainer.state.fn != TrainerFn.PREDICTING and self.quantization_layer_list[-1].is_initialized:
            reconstruction, losses = self.reconstruction(quantized, fields)
            self.last_head_losses = {k: float(v.detach()) for k, v in losses.items()}
        self.last_quantization_loss = float(qloss.detach())
        self.last_reconstruction_loss = float(reconstruction.detach())
        return codes, residuals, qloss, reconstruction

    def backward(self, loss, *args, **kwargs):
        if not hasattr(self, 'mtl') or not hasattr(self, 'mtl_losses'):
            return super().backward(loss, *args, **kwargs)
        shared = [m.weight for m in self.encoder.modules() if isinstance(m, nn.Linear)][-1]
        parameters = [p for n,p in self.named_parameters() if p.requires_grad and not n.startswith('mtl.')]
        self.mtl.backward_tasks({k:v*self.reconstruction_loss_weight for k,v in self.mtl_losses.items()}, loss, parameters, shared)
