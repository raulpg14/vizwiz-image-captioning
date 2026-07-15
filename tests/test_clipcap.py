"""
Standalone smoke test for src/models/clipcap.py.

Verifies the reconstructed ClipCapModel is structurally correct: builds a
fresh model, runs a forward pass with dummy tensors, checks output shapes,
and confirms gradients flow back through both the mapping network and the
unfrozen GPT-2 layers.

This does NOT require the VizWiz dataset, a GPU, or any checkpoint file —
it only needs the packages in requirements.txt (transformers, torch) and
internet access the first time it runs (to download the pretrained gpt2
weights from Hugging Face, ~500MB, cached locally afterward).

Run from the repo root:
    python -m pytest tests/test_clipcap.py -v
or directly:
    python tests/test_clipcap.py
"""

import sys
from pathlib import Path

import torch

# Allow running this script directly (not just via pytest) from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.clipcap import (
    ClipCapModel,
    MappingNetwork,
    create_clipcap_model,
    set_gpt2_trainable,
)


def test_mapping_network_output_shape():
    """MappingNetwork should turn a (B, 512) CLIP embedding into (B, prefix_len, 768)."""
    device = torch.device('cpu')
    prefix_len = 10
    gpt2_dim = 768

    mapping_net = MappingNetwork(
        clip_dim=512, gpt2_dim=gpt2_dim, prefix_len=prefix_len, n_layers=2, n_heads=8
    ).to(device)

    dummy_clip = torch.randn(4, 512)
    out = mapping_net(dummy_clip)

    assert out.shape == (4, prefix_len, gpt2_dim), f"Shape error: {out.shape}"
    print(f"[PASS] MappingNetwork output shape: {tuple(out.shape)}")


def test_clipcap_model_forward_pass():
    """
    Full ClipCapModel forward pass: dummy CLIP embeddings + dummy caption
    token ids should produce logits of shape (B, prefix_len + seq_len, vocab_size).
    """
    device = torch.device('cpu')
    batch_size = 2
    seq_len = 12
    prefix_length = 10
    vocab_size = 50257  # GPT-2 base vocab size

    model = create_clipcap_model(
        device=device,
        clip_dim=512,
        gpt2_dim=768,
        prefix_length=prefix_length,
        n_layers=2,       # smaller than the real 4 layers, just for a fast test
        n_heads=8,
        freeze_gpt2_layers=6,
    )
    model.train()

    clip_embeds = torch.randn(batch_size, 512)
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    logits = model(clip_embeds=clip_embeds, input_ids=input_ids, attention_mask=attention_mask)

    expected_shape = (batch_size, prefix_length + seq_len, vocab_size)
    assert logits.shape == expected_shape, f"Shape error: got {tuple(logits.shape)}, expected {expected_shape}"
    print(f"[PASS] ClipCapModel forward pass output shape: {tuple(logits.shape)}")

    return model, logits, clip_embeds, input_ids, attention_mask


def test_gradients_flow():
    """
    Confirms backprop reaches both the mapping network and GPT-2's unfrozen
    layers — if the reconstructed forward pass had a bug (e.g. wrong tensor
    disconnected from the graph), gradients would be None or all-zero here.
    """
    device = torch.device('cpu')
    model = create_clipcap_model(device=device, n_layers=2, freeze_gpt2_layers=6)
    model.train()

    batch_size, seq_len = 2, 8
    clip_embeds = torch.randn(batch_size, 512, requires_grad=False)
    input_ids = torch.randint(low=0, high=50257, size=(batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = input_ids.clone()

    logits = model(clip_embeds=clip_embeds, input_ids=input_ids, attention_mask=attention_mask)

    # Only the text portion of logits has corresponding labels; prefix
    # positions are not supervised directly (same convention as training).
    text_logits = logits[:, model.prefix_length:, :]
    loss = torch.nn.functional.cross_entropy(
        text_logits.reshape(-1, text_logits.size(-1)),
        labels.reshape(-1),
    )
    loss.backward()

    mapping_grad = model.mapping_network.projection.weight.grad
    assert mapping_grad is not None, "No gradient reached MappingNetwork.projection"
    assert torch.any(mapping_grad != 0), "MappingNetwork.projection gradient is all-zero"
    print(f"[PASS] Gradient reached mapping_network.projection (norm={mapping_grad.norm():.4f})")

    # Layer 11 (last, unfrozen) should have gradients; layer 0 (frozen) should not.
    last_layer_grad = model.gpt.transformer.h[11].mlp.c_fc.weight.grad
    first_layer_grad = model.gpt.transformer.h[0].mlp.c_fc.weight.grad

    assert last_layer_grad is not None and torch.any(last_layer_grad != 0), \
        "No gradient reached unfrozen GPT-2 layer 11"
    assert first_layer_grad is None or torch.all(first_layer_grad == 0), \
        "Frozen GPT-2 layer 0 unexpectedly received a gradient"

    print(f"[PASS] Gradient reached unfrozen GPT-2 layer 11 (norm={last_layer_grad.norm():.4f})")
    print("[PASS] Frozen GPT-2 layer 0 correctly received no gradient")


def test_set_gpt2_trainable_toggle():
    """set_gpt2_trainable should flip requires_grad on layers >= freeze_layers and the LM head."""
    device = torch.device('cpu')
    model = create_clipcap_model(device=device, n_layers=2, freeze_gpt2_layers=6)

    # After creation, layer 6+ should already be trainable (default), layer 0 frozen.
    assert model.gpt.transformer.h[6].mlp.c_fc.weight.requires_grad is True
    assert model.gpt.transformer.h[0].mlp.c_fc.weight.requires_grad is False

    set_gpt2_trainable(model, trainable=False, freeze_layers=6)
    assert model.gpt.transformer.h[6].mlp.c_fc.weight.requires_grad is False
    assert model.gpt.lm_head.weight.requires_grad is False
    print("[PASS] set_gpt2_trainable(False) correctly freezes layers 6-11 + LM head")

    set_gpt2_trainable(model, trainable=True, freeze_layers=6)
    assert model.gpt.transformer.h[6].mlp.c_fc.weight.requires_grad is True
    assert model.gpt.lm_head.weight.requires_grad is True
    print("[PASS] set_gpt2_trainable(True) correctly unfreezes layers 6-11 + LM head")


if __name__ == '__main__':
    print("Running clipcap.py smoke tests (CPU, dummy data, no dataset required)...\n")

    print("--- Test 1: MappingNetwork output shape ---")
    test_mapping_network_output_shape()

    print("\n--- Test 2: ClipCapModel forward pass ---")
    test_clipcap_model_forward_pass()

    print("\n--- Test 3: Gradient flow ---")
    test_gradients_flow()

    print("\n--- Test 4: set_gpt2_trainable toggle ---")
    test_set_gpt2_trainable_toggle()

    print("\nAll tests passed. clipcap.py is structurally and functionally correct.")