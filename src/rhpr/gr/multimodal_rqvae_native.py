"""Per-modality MLP heads around the pinned GRID residual quantizer."""
import torch
from torch import nn
from lightning.pytorch.trainer.states import TrainerFn
from src.modules.clustering.residual_quantization import ResidualQuantization


class MultimodalResidualQuantization(ResidualQuantization):
    def __init__(self, modality_schema, active_modalities, head_dim=128, head_hidden=256, **kwargs):
        super().__init__(**kwargs)
        self.modality_schema = dict(modality_schema)
        self.active_modalities = tuple(active_modalities)
        self.encoder_heads = nn.ModuleDict({name: nn.Sequential(
            nn.Linear(dim, head_hidden), nn.ReLU(), nn.Linear(head_hidden, head_dim))
            for name, dim in modality_schema.items()})
        self.decoder_heads = nn.ModuleDict({name: nn.Sequential(
            nn.Linear(self.decoder.output_dim, head_hidden), nn.ReLU(), nn.Linear(head_hidden, dim))
            for name, dim in modality_schema.items()})
        self.head_dim = head_dim
        self.last_head_losses = {}

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
            losses[name] = ((prediction[valid] - fields[name][valid]).square().mean()
                            if valid.any() else prediction.sum() * 0.)
        return sum(losses.values()), losses

    def model_step(self, model_input):
        fields = {k: v.to(self.device) for k, v in model_input.transformed_features.items()}
        encoded = self.encode_modalities(fields)
        codes, residuals, quantized, qloss = self(encoded)
        self.last_head_losses = {}
        reconstruction = qloss.new_zeros(())
        if self.trainer.state.fn != TrainerFn.PREDICTING and self.quantization_layer_list[-1].is_initialized:
            reconstruction, losses = self.reconstruction(quantized, fields)
            self.last_head_losses = {k: float(v.detach()) for k, v in losses.items()}
        self.last_quantization_loss = float(qloss.detach())
        self.last_reconstruction_loss = float(reconstruction.detach())
        return codes, residuals, qloss, reconstruction
