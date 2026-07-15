"""
Architecture 1: CLIP ViT-B/32 (frozen) + Transformer Mapping Network +
GPT-2 with selective fine-tuning.

This module is import-safe: importing it has no side effects. Model
construction (loading pretrained weights, freezing parameters) happens
inside `create_clipcap_model()`, which the caller invokes explicitly.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPVisionModelWithProjection, GPT2LMHeadModel
from tqdm import tqdm

from src.data.dataset import VizWizRawDataset


# --------------------------------------------------------------------------
# Vision encoder (frozen)
# --------------------------------------------------------------------------

def load_clip_encoder(device, model_name='openai/clip-vit-base-patch32'):
    """
    Loads the pretrained CLIP vision encoder and freezes all parameters —
    Architecture 1 never fine-tunes CLIP itself, only the mapping network
    that sits downstream of it.
    """
    encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_name, use_safetensors=True
    ).to(device)

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    print('CLIP parameters frozen.')
    print(f"CLIP parameter count: {sum(p.numel() for p in encoder.parameters()):,}")
    return encoder


def validate_feature_keys(features, split_name, splits):
    """
    Validates that cached CLIP feature keys exactly match the expected
    filenames for a split — catches stale/partial feature caches early.
    """
    expected = set(Path(x).name for x in splits[split_name])
    found = set(features.keys())
    missing = sorted(expected - found)
    extra = sorted(found - expected)

    if missing or extra:
        raise ValueError(
            f"{split_name} feature cache mismatch - "
            f"missing: {missing[:5]} ({len(missing)} total), "
            f"extra: {extra[:5]} ({len(extra)} total). "
            f"Delete the cached .pt file and re-extract."
        )
    print(f"  {split_name} feature cache validated - {len(found)} keys")


def extract_clip_features(split_name, split_dict, images_dir, processor, clip_model,
                           device, batch_size=64, seed_worker=None, generator=None):
    """
    Runs the frozen CLIP encoder once over every image in a split and
    returns a dict of {filename: 512-dim CPU tensor}. This is a one-time
    pre-processing pass — cache the result to disk rather than re-running
    per epoch.
    """
    t0 = time.time()

    raw_dataset = VizWizRawDataset(split_dict, split_name, images_dir, processor)
    raw_loader = DataLoader(
        raw_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    features_dict = {}
    clip_model.eval()

    with torch.inference_mode():
        for batch in tqdm(raw_loader, desc=f"Extracting {split_name} features"):
            pixel_values = batch['pixel_values'].to(device)
            embeddings = clip_model(pixel_values=pixel_values).image_embeds  # (B, 512)

            for img_id, embed in zip(batch['image_id'], embeddings.cpu()):
                features_dict[img_id] = embed

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    print(f"  {split_name} extraction complete: {len(features_dict)} images in {mins}m {secs:02d}s")

    return features_dict


# --------------------------------------------------------------------------
# Mapping network: CLIP embedding -> GPT-2 prefix tokens
# --------------------------------------------------------------------------

class MappingNetwork(nn.Module):
    """
    Projects a 512-dim CLIP embedding into `prefix_len` tokens of 768-dim,
    scale-matched to GPT-2's token embedding space.
    """

    def __init__(self, clip_dim=512, gpt2_dim=768, prefix_len=10, n_layers=4, n_heads=8):
        """
        Args:
            clip_dim: Dimensionality of CLIP image_embeds.
            gpt2_dim: GPT-2 hidden size.
            prefix_len: Number of learned visual prefix tokens.
            n_layers: Transformer encoder layer count.
            n_heads: Transformer attention head count.
        """
        super().__init__()
        self.prefix_len = prefix_len
        self.gpt2_dim = gpt2_dim

        # Input projection: 512 -> 768 * prefix_len
        self.projection = nn.Linear(clip_dim, gpt2_dim * prefix_len)
        self.norm = nn.LayerNorm(gpt2_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gpt2_dim,
            nhead=n_heads,
            dim_feedforward=gpt2_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_proj = nn.Linear(gpt2_dim, gpt2_dim)
        self.final_norm = nn.LayerNorm(gpt2_dim)

        # Learned scalar gate: initialized to 0.1 so prefix std is ~0.1 at
        # init, closely matching GPT-2 token embedding std (~0.11) and
        # preventing softmax saturation. Backprop adjusts this freely.
        self.output_scale = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        """Xavier-initializes mapper weights for stable optimisation."""
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        for module in self.transformer.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Small normal init (GPT-2 scheme) keeps the linear map close to
        # identity at start, before final_norm rescales it.
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, clip_embed):
        """
        Args:
            clip_embed: Tensor of shape (batch, 512).
        Returns:
            Tensor of shape (batch, prefix_len, gpt2_dim) — GPT-2 prefix embeddings.
        """
        x = self.projection(clip_embed)
        x = x.view(-1, self.prefix_len, self.gpt2_dim)

        x = self.norm(x)
        x = self.transformer(x)

        x = self.output_proj(x)
        x = self.final_norm(x)
        x = x * self.output_scale

        return x


# --------------------------------------------------------------------------
# Language decoder: GPT-2 with selective fine-tuning
# --------------------------------------------------------------------------

def load_gpt2_selective_finetune(device, freeze_layers=6, model_name='gpt2'):
    """
    Loads GPT-2 and freezes embeddings plus the first `freeze_layers`
    transformer blocks, leaving later blocks + LM head trainable. Also
    detaches the LM head from GPT-2's default weight-tying to `wte`, since
    freezing `wte` would otherwise silently freeze the LM head too.
    """
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    for param in model.transformer.wte.parameters():
        param.requires_grad = False
    for param in model.transformer.wpe.parameters():
        param.requires_grad = False

    for i in range(freeze_layers):
        for param in model.transformer.h[i].parameters():
            param.requires_grad = False

    print(f"GPT-2 embeddings and first {freeze_layers} layers frozen.")

    # Untie lm_head from wte so it stays trainable despite wte being frozen.
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())
    print(f"lm_head weight untied - trainable: {model.lm_head.weight.requires_grad}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nGPT-2 total params    : {total:,}")
    print(f"GPT-2 trainable params: {trainable:,}  (layers {freeze_layers}-11 + LM head)")
    print(f"GPT-2 frozen params   : {total - trainable:,}  (embeddings + layers 0-{freeze_layers - 1})")

    return model


def set_gpt2_trainable(model, trainable: bool, freeze_layers: int = 6):
    """Toggles fine-tuning on GPT-2's later layers + LM head for a ClipCapModel."""
    for i in range(freeze_layers, 12):
        for param in model.gpt.transformer.h[i].parameters():
            param.requires_grad = trainable
    model.gpt.lm_head.weight.requires_grad = trainable


# --------------------------------------------------------------------------
# ClipCap wrapper model
# --------------------------------------------------------------------------

class ClipCapModel(nn.Module):
    """
    End-to-end Architecture 1 model: projects a CLIP image embedding into
    GPT-2 prefix tokens via the mapping network, concatenates them with
    caption token embeddings, and runs the combined sequence through GPT-2.
    """

    def __init__(self, mapping_network, gpt_decoder, prefix_length=10):
        super().__init__()
        self.mapping_network = mapping_network
        self.gpt = gpt_decoder
        self.prefix_length = prefix_length

    def forward(self, clip_embeds, input_ids, attention_mask):
        """
        Returns GPT-2 vocabulary logits for the visual prefix + text tokens.

        Args:
            clip_embeds: (batch, 512) CLIP image embeddings.
            input_ids: (batch, seq_len) caption token ids.
            attention_mask: (batch, seq_len) caption attention mask.
        """
        prefix_tokens = self.mapping_network(clip_embeds)  # (B, prefix_length, 768)

        text_embeddings = self.gpt.transformer.wte(input_ids)
        prefix_tokens = prefix_tokens.to(dtype=text_embeddings.dtype)
        inputs_embeds = torch.cat([prefix_tokens, text_embeddings], dim=1)

        prefix_attention = torch.ones(
            attention_mask.size(0),
            self.prefix_length,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        extended_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

        outputs = self.gpt(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits


def create_clipcap_model(device, clip_dim=512, gpt2_dim=768, prefix_length=10,
                          n_layers=4, n_heads=8, freeze_gpt2_layers=6):
    """
    Builds a fresh ClipCapModel: a newly initialized MappingNetwork paired
    with a GPT-2 decoder using selective fine-tuning.
    """
    mapping_net = MappingNetwork(
        clip_dim=clip_dim,
        gpt2_dim=gpt2_dim,
        prefix_len=prefix_length,
        n_layers=n_layers,
        n_heads=n_heads,
    ).to(device)

    gpt_decoder = load_gpt2_selective_finetune(device, freeze_layers=freeze_gpt2_layers)

    model = ClipCapModel(
        mapping_network=mapping_net,
        gpt_decoder=gpt_decoder,
        prefix_length=prefix_length,
    ).to(device)

    return model