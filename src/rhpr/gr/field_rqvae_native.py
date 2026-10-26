"""Small field-reconstruction adapter; import after load_grid(source_root)."""
import torch
from lightning.pytorch.trainer.states import TrainerFn

from src.modules.clustering.residual_quantization import ResidualQuantization
from rhpr.gr.fields import FIELD_NAMES, FieldEncoder, FieldHeads
from rhpr.gr.vendor.libmtl_uw import UncertaintyWeighting
from rhpr.gr.field_autoencoder import FieldReconstructionObjective, make_autoencoder_optimizer


class FieldResidualQuantization(FieldReconstructionObjective, ResidualQuantization):
    def __init__(self, feature_schema, field_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.feature_encoder = FieldEncoder(feature_schema, field_dim)
        self.field_heads = FieldHeads(feature_schema, self.decoder.output_dim)
        self.field_names = tuple(feature_schema["fields"])
        self.uncertainty = UncertaintyWeighting(len(self.field_names))
        self.last_field_losses, self.last_field_counts = {}, {}
        self.last_loss_components = {}
        self.loss_policy = {"mode": "uncertainty"}
        self.loss_normalizers = {name: 1.0 for name in self.field_names}
        self._balancing_losses = None
        self.reconstruction_updates = 0
        self.initialization_batches = 0

    def configure_optimizers(self):
        if self.loss_policy["mode"] == "uncertainty" and "uw_learning_rate" not in self.loss_policy:
            return super().configure_optimizers()
        return {"optimizer": make_autoencoder_optimizer(self, self.optimizer)}

    def on_before_backward(self, loss):
        # Native GRID calls opt.zero_grad() before manual_backward(). This hook
        # preserves the GradNorm gradient until the same optimizer step.
        if self._balancing_losses is not None:
            shared = [m.weight for m in self.decoder.modules() if isinstance(m, torch.nn.Linear)][-1]
            self.task_balancer.backward_weights(self._balancing_losses, shared, list(self.last_field_counts.values()))

    def on_train_epoch_start(self):
        self._full_reconstruction_epoch = all(layer.is_initialized for layer in self.quantization_layer_list)
        self._balancing_batches = 0

    def on_train_batch_start(self, batch, batch_idx):
        from rhpr.gr.ae_staging import enforce_stage_modes
        enforce_stage_modes(self)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.last_field_losses:
            self.reconstruction_updates += 1
        else:
            self.initialization_batches += 1
        if self._balancing_losses is not None:
            if self._full_reconstruction_epoch:
                self.task_balancer.observe(self._balancing_losses, list(self.last_field_counts.values()))
                self._balancing_batches += 1
            self._balancing_losses = None

    def on_train_epoch_end(self):
        if (hasattr(self, "task_balancer") and self._full_reconstruction_epoch
                and self._balancing_batches == self.trainer.num_training_batches):
            self.task_balancer.finish_pass()

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
        self._balancing_losses = None
        reconstruction = quantization_loss.new_zeros(())
        # Preserve GRID's initialization-only updates and subsequent STE path.
        if self.trainer.state.fn != TrainerFn.PREDICTING and self.quantization_layer_list[-1].is_initialized:
            predictions = self.field_heads(self.decoder(quantized))
            reconstruction, losses, counts = self.objective(predictions, fields)
            if hasattr(self, "task_balancer"):
                self._balancing_losses = [losses[name] / self.loss_normalizers[name] for name in self.field_names]
            self.last_field_losses = {name: float(loss.detach()) for name, loss in losses.items()}
            self.last_field_counts = counts
        self.last_loss_components = {
            "quantization": quantization_loss.detach(), "weighted_reconstruction": reconstruction.detach(),
            "quantization_weight": self.quantization_loss_weight,
            "reconstruction_weight": self.reconstruction_loss_weight}
        return codes, residuals, quantization_loss, reconstruction
