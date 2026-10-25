"""Small field-reconstruction adapter; import after load_grid(source_root)."""
import torch
from lightning.pytorch.trainer.states import TrainerFn

from src.modules.clustering.residual_quantization import ResidualQuantization
from rhpr.gr.fields import FIELD_NAMES, FieldEncoder, FieldHeads
from rhpr.gr.vendor.libmtl_uw import UncertaintyWeighting


class FieldResidualQuantization(ResidualQuantization):
    def __init__(self, feature_schema, field_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.feature_encoder = FieldEncoder(feature_schema, field_dim)
        self.field_heads = FieldHeads(feature_schema, self.decoder.output_dim)
        self.uncertainty = UncertaintyWeighting(len(FIELD_NAMES))
        self.last_field_losses, self.last_field_counts = {}, {}
        self.last_loss_components = {}

    def encode_fields(self, fields):
        features = self.feature_encoder(fields)
        return features, self.encoder(self.normalization_layer(features))

    def model_step(self, model_input):
        fields = {key: value.to(self.device) for key, value in model_input.transformed_features.items()}
        _, encoded = self.encode_fields(fields)
        codes, residuals, quantized, quantization_loss = self(encoded)
        # Only the next periodic log needs these scalar graph values. Detach so
        # diagnostics never retain the preceding optimization graph.
        self.last_field_losses, self.last_field_counts = {}, {}
        reconstruction = quantization_loss.new_zeros(())
        # Preserve GRID's initialization-only updates and subsequent STE path.
        if self.trainer.state.fn != TrainerFn.PREDICTING and self.quantization_layer_list[-1].is_initialized:
            predictions = self.field_heads(self.decoder(quantized))
            losses, counts = self.field_heads.losses(predictions, fields)
            reconstruction = self.uncertainty(torch.stack([losses[name] for name in FIELD_NAMES]),
                torch.tensor([counts[name] > 0 for name in FIELD_NAMES], device=self.device))
            self.last_field_losses = {name: float(loss.detach()) for name, loss in losses.items()}
            self.last_field_counts = counts
        self.last_loss_components = {
            "quantization": quantization_loss.detach(), "weighted_reconstruction": reconstruction.detach(),
            "quantization_weight": self.quantization_loss_weight,
            "reconstruction_weight": self.reconstruction_loss_weight}
        return codes, residuals, quantization_loss, reconstruction
