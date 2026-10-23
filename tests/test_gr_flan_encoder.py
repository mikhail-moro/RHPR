"""Source verification, resumable ranges and byte-preserving encoder extraction."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_gr_flan_encoder import download_notice, download_verified, extract_encoder, sha256, validate_encoder


class Response(io.BytesIO):
    def __init__(self, content, start, end, size, status=206):
        super().__init__(content)
        self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{size}"}


class PublishedDownloadTests(unittest.TestCase):
    def test_notice_accepts_full_response_but_rejects_substitution_and_reuses_verified_copy(self):
        data = b"Original licensed model notice."
        digest = hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "LICENSE"
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b"changed")):
                with self.assertRaisesRegex(ValueError, "notice digest mismatch"):
                    download_notice("https://example.invalid/LICENSE", path, len(data), digest)
            self.assertFalse(path.exists())
            with patch("urllib.request.urlopen", return_value=io.BytesIO(data)):
                download_notice("https://example.invalid/LICENSE", path, len(data), digest)
            self.assertEqual(path.read_bytes(), data)
            with patch("urllib.request.urlopen") as network:
                download_notice("https://example.invalid/LICENSE", path, len(data), digest)
            network.assert_not_called()

    def test_resume_skips_completed_ranges_and_checks_whole_published_hash(self):
        content = b"pinned model data"
        digest = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "model.bin"
            partial = target.with_name("model.bin.partial")
            partial.write_bytes(content[:4] + b"\0" * (len(content) - 4))
            target.with_name("model.bin.chunks.json").write_text(json.dumps({
                "size_bytes": len(content), "sha256": digest, "chunk_size": 4, "completed": [0],
            }))
            starts = []

            def request(request, timeout):
                start, end = [int(x) for x in request.headers["Range"].split("=")[1].split("-")]
                starts.append(start)
                return Response(content[start:end + 1], start, end, len(content))

            with patch("urllib.request.urlopen", side_effect=request):
                download_verified("https://example.invalid/pinned", target, len(content), digest,
                                  chunk_size=4, workers=2)
            self.assertEqual(target.read_bytes(), content)
            self.assertEqual(sorted(starts), [4, 8, 12, 16])
            self.assertFalse(partial.exists())
            self.assertFalse(target.with_name("model.bin.chunks.json").exists())
            with patch("urllib.request.urlopen") as network:
                download_verified("https://example.invalid/pinned", target, len(content), digest)
            network.assert_not_called()

    def test_range_rejection_and_digest_substitution_never_publish_source(self):
        content = b"expected"
        digest = hashlib.sha256(content).hexdigest()
        for wrong_status in (True, False):
            with self.subTest(wrong_status=wrong_status), tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / "model.bin"

                def request(request, timeout):
                    body = content if wrong_status else b"replaced"
                    return Response(body, 0, len(body) - 1, len(body), status=200 if wrong_status else 206)

                with patch("urllib.request.urlopen", side_effect=request), self.assertRaises(ValueError):
                    download_verified("https://example.invalid/pinned", target, len(content), digest,
                                      chunk_size=len(content), workers=1)
                self.assertFalse(target.exists())

    def test_partial_identity_cannot_be_reused_for_another_model_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "model.bin"
            target.with_name("model.bin.partial").write_bytes(b"1234")
            target.with_name("model.bin.chunks.json").write_text(json.dumps({
                "size_bytes": 4, "sha256": "old", "chunk_size": 4, "completed": [0],
            }))
            with patch("urllib.request.urlopen") as network, self.assertRaisesRegex(ValueError, "different"):
                download_verified("https://example.invalid/new", target, 4, "new", chunk_size=4)
            network.assert_not_called()


class EncoderExtractionTests(unittest.TestCase):
    def test_streamed_subset_is_readable_and_preserves_each_selected_value(self):
        import torch
        from safetensors.torch import load_file, save_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensors = {"shared.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
                       "encoder.block.0.weight": torch.randn(4, 3),
                       "decoder.block.0.weight": torch.randn(3, 2)}
            source, output = root / "source.safetensors", root / "model.safetensors"
            save_file(tensors, source, metadata={"format": "pt"})
            report = extract_encoder(source, output)
            loaded = load_file(output)
            self.assertEqual(set(loaded), {"shared.weight", "encoder.block.0.weight"})
            self.assertEqual(report["tensor_count"], 2)
            self.assertEqual(report["parameters"], 36)
            self.assertEqual(report["derived_sha256"], sha256(output))
            for name, tensor in loaded.items():
                self.assertTrue(torch.equal(tensor, tensors[name]))

    def test_transformers_meta_validation_restores_shared_alias_without_random_weights(self):
        import torch
        from safetensors.torch import save_file
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast, T5Config, T5EncoderModel

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration = T5Config(d_model=8, d_ff=16, d_kv=4, num_heads=2,
                                     num_layers=1, vocab_size=16)
            model = T5EncoderModel(configuration)
            expected_parameters = sum(x.numel() for x in model.parameters())
            state = {name: tensor for name, tensor in model.state_dict().items()
                     if name != "encoder.embed_tokens.weight"}
            save_file(state, root / "model.safetensors", metadata={"format": "pt"})
            configuration.save_pretrained(root)
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
                {"<pad>": 0, "<unk>": 1, "</s>": 2}, unk_token="<unk>")),
                unk_token="<unk>", pad_token="<pad>", eos_token="</s>")
            tokenizer.save_pretrained(root)
            index = root / "index.json"
            index.write_text(json.dumps({"weight_map": {name: "source" for name in state}}))
            result = validate_encoder(root, index)
            self.assertEqual(result["missing_keys"], [])
            self.assertEqual(result["unexpected_keys"], [])
            self.assertEqual(result["parameters"], expected_parameters)
            self.assertEqual(result["dtype"], "torch.float32")

    def test_published_index_alias_requires_source_metadata_and_real_layers_remain_mandatory(self):
        import torch
        from safetensors.torch import save_file
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast, T5Config, T5EncoderModel

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration = T5Config(d_model=8, d_ff=16, d_kv=4, num_heads=2,
                                     num_layers=1, vocab_size=16)
            model = T5EncoderModel(configuration)
            configuration.save_pretrained(root)
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
                {"<pad>": 0, "<unk>": 1, "</s>": 2}, unk_token="<unk>")),
                unk_token="<unk>", pad_token="<pad>", eos_token="</s>")
            tokenizer.save_pretrained(root)
            original = model.state_dict()
            state = {name: value for name, value in original.items()
                     if name != "encoder.embed_tokens.weight"}
            source = root / "source.safetensors"
            save_file(state, source, metadata={"format": "pt",
                      "encoder.embed_tokens.weight": "shared.weight"})
            extract_encoder(source, root / "model.safetensors")
            index = root / "index.json"
            index.write_text(json.dumps({"weight_map": {name: "source" for name in original}}))
            result = validate_encoder(root, index)
            self.assertEqual(result["index_aliases_restored"], ["encoder.embed_tokens.weight"])
            self.assertEqual(result["missing_keys"], [])
            save_file(state, root / "model.safetensors", metadata={"format": "pt"})
            with self.assertRaisesRegex(ValueError, "tensor keys differ"):
                validate_encoder(root, index)
            incomplete = dict(state)
            del incomplete["encoder.block.0.layer.0.SelfAttention.q.weight"]
            save_file(incomplete, root / "model.safetensors", metadata={"format": "pt",
                      "encoder.embed_tokens.weight": "shared.weight"})
            with self.assertRaisesRegex(ValueError, "tensor keys differ"):
                validate_encoder(root, index)


if __name__ == "__main__":
    unittest.main()
