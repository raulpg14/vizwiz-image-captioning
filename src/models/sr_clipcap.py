"""
Architecture 2: SR-ClipCap (SigLIP + Perceiver Resampler + LoRA GPT-2).

This module is import-safe: importing it has no side effects. Model
construction (loading pretrained weights, configuring LoRA adapters) happens
inside `create_sr_clipcap_model()`, which the caller invokes explicitly.
"""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel, GPT2Tokenizer, GPT2LMHeadModel
from peft import LoraConfig, TaskType, get_peft_model


# --------------------------------------------------------------------------
# Vision encoder: SigLIP (frozen)
# --------------------------------------------------------------------------

def load_siglip_backbone(device="cpu", model_name="google/siglip-base-patch16-224"):
    """
    Loads the pretrained SigLIP vision encoder and freezes all parameters —
    Architecture 2 relies on a frozen vision backbone, extracting an unpooled
    grid of spatial features rather than a single global embedding.
    """
    load_kwargs = {}
    if str(device) == 'cuda':
        load_kwargs['dtype'] = torch.float16

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SiglipVisionModel.from_pretrained(
        model_name,
        use_safetensors=True,
        **load_kwargs,
    ).to(device)

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    print(f'SigLIP parameters frozen. Backbone: {model_name}')
    return processor, model


def load_rgb_image(image_path):
    """
    Loads one image as RGB and immediately closes the underlying file handle
    to prevent OS "too many open files" errors during large dataset iterations.
    """
    with Image.open(image_path) as image:
        return image.convert("RGB")


def extract_siglip_spatial_features(
    loader, 
    model, 
    processor, 
    device, 
    expected_seq_len=196,
    expected_hidden_dim=768, 
    save_dtype=torch.float16, 
    overwrite=False
):
    """
    Runs the frozen SigLIP encoder once over every image to extract unpooled
    patch-grid features (e.g., 14x14 = 196 tokens). Caches to disk to save
    compute during the iterative training phase.
    """
    model.eval()
    extracted = 0
    skipped = 0

    for batch in tqdm(loader, desc="Extracting SigLIP spatial features"):
        pending = []

        for file_name, image_path, feature_path in zip(batch["file_name"], batch["image_path"], batch["feature_path"]):
            feature_path = Path(feature_path)

            if feature_path.exists() and not overwrite:
                skipped += 1
                continue
            pending.append((file_name, Path(image_path), feature_path))

        if not pending:
            continue

        images = [load_rgb_image(image_path) for _, image_path, _ in pending]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device, non_blocking=True)

        with torch.inference_mode(), torch.autocast(device_type=str(device).split(":")[0], dtype=torch.float16, enabled=(str(device) == "cuda")):
            outputs = model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
      
        spatial_tokens = outputs.last_hidden_state  # (Batch, 196, 768)

        if spatial_tokens.ndim != 3:
            raise ValueError(f"Expected [batch, {expected_seq_len}, {expected_hidden_dim}], got {tuple(spatial_tokens.shape)}")

        if tuple(spatial_tokens.shape[1:]) != (expected_seq_len, expected_hidden_dim):
            raise ValueError(
                f"Unexpected SigLIP spatial token shape: {tuple(spatial_tokens.shape[1:])}; "
                f"expected {(expected_seq_len, expected_hidden_dim)}"
            )

        spatial_tokens = spatial_tokens.detach().to("cpu", dtype=save_dtype)

        # Atomic save: write to a temporary file, then rename to avoid corrupted partial files
        for tokens, (file_name, _, feature_path) in zip(spatial_tokens, pending):
            tmp_path = feature_path.with_suffix(feature_path.suffix + ".tmp")
            torch.save(tokens.clone(), tmp_path)
            tmp_path.replace(feature_path)
            extracted += 1

        del images, inputs, pixel_values, outputs, spatial_tokens


# --------------------------------------------------------------------------
# Mapping network: Perceiver Resampler
# --------------------------------------------------------------------------

class PerceiverResamplerLayer(nn.Module):
    """
    One cross-attention + FFN block. Uses Pre-LayerNorm architecture 
    to guarantee gradient stability during deep network training.
    """

    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.attn_norm  = nn.LayerNorm(embed_dim)
        self.kv_norm    = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, latents, spatial_tokens, return_attn=False):
        # ── Pre-LN cross-attention: Queries from latents, K/V from spatial grid ──
        normed = self.attn_norm(latents)
        normed_kv = self.kv_norm(spatial_tokens)
        
        attn_out, attn_weights = self.cross_attn(
            query=normed, key=normed_kv, value=normed_kv,
            need_weights=return_attn, average_attn_weights=False,
        )
        latents = latents + attn_out

        # ── Pre-LN Feed Forward Network ──
        latents = latents + self.mlp(self.final_norm(latents))

        if return_attn:
            return latents, attn_weights
        return latents


class PerceiverResampler(nn.Module):
    """
    Compresses SigLIP's 14x14 (196) spatial token grid down into a fixed 
    number of learned visual prefix tokens (default 16) via cross-attention.
    """

    def __init__(self):
        super().__init__()
        self.num_queries = 16
        self.embed_dim   = 768
        self.num_layers  = 3

        # Learned latent queries and spatial positional embeddings
        self.latents = nn.Parameter(torch.randn(1, self.num_queries, self.embed_dim) * 0.02)
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, 196, self.embed_dim) * 0.02)

        self.layers = nn.ModuleList([
            PerceiverResamplerLayer(self.embed_dim, num_heads=8)
            for _ in range(self.num_layers)
        ])

    def forward(self, spatial_tokens):
        """
        Args:
            spatial_tokens: (Batch, 196, 768) unpooled features from SigLIP.
        Returns:
            (Batch, 16, 768) compressed visual prefix tokens.
        """
        if spatial_tokens.size(1) != self.spatial_pos_embed.size(1):
            raise ValueError(f"Expected 196 spatial tokens, got {spatial_tokens.size(1)}")

        batch_size = spatial_tokens.size(0)
        
        # Inject spatial structure into the vision tokens
        spatial_tokens = spatial_tokens + self.spatial_pos_embed.to(dtype=spatial_tokens.dtype)
        
        # Expand learned latent queries for the whole batch
        latents = self.latents.expand(batch_size, -1, -1)

        # Iteratively refine latents by attending to the spatial grid
        for layer in self.layers:
            latents = layer(latents, spatial_tokens)

        return latents


@torch.inference_mode()
def get_resampler_attention_weights(spatial_tensor, model, device='cuda'):
    """
    Extracts per-layer cross-attention weights from the Perceiver Resampler
    for visualization/analysis purposes.
    """
    if spatial_tensor.ndim == 2:
        spatial_tensor = spatial_tensor.unsqueeze(0)
    spatial_tensor = spatial_tensor.to(device=device, dtype=model.resampler.latents.dtype)

    batch_size = spatial_tensor.size(0)
    tokens     = spatial_tensor + model.resampler.spatial_pos_embed.to(dtype=spatial_tensor.dtype)
    latents    = model.resampler.latents.expand(batch_size, -1, -1)

    all_attn_weights = []
    for layer in model.resampler.layers:
        latents, attn_weights = layer(latents, tokens, return_attn=True)
        all_attn_weights.append(attn_weights.cpu())

    return torch.stack(all_attn_weights, dim=0).squeeze(1)


# --------------------------------------------------------------------------
# Language decoder utilities
# --------------------------------------------------------------------------

def _get_gpt2_token_embedding_layer(gpt_model):
    """
    Helper function to reliably extract the word token embedding layer (`wte`)
    from GPT-2. This is necessary because PEFT/LoRA wraps the base model,
    changing the attribute path.
    """
    if hasattr(gpt_model, "base_model"):
        # PEFT model wrapper path
        return gpt_model.base_model.model.transformer.wte
    # Standard GPT2LMHeadModel path
    return gpt_model.transformer.wte


# --------------------------------------------------------------------------
# SR-ClipCap wrapper model & Factory
# --------------------------------------------------------------------------

class SRClipCapModel(nn.Module):
    """
    End-to-end Architecture 2 model: 
    SigLIP spatial grid -> Perceiver Resampler -> LoRA-adapted GPT-2.
    """

    def __init__(self, resampler, gpt_decoder, prefix_length=16):
        super().__init__()
        self.resampler = resampler
        self.gpt = gpt_decoder
        self.prefix_length = prefix_length
        
        # Learned scalar to match output std dev to GPT-2 token embeddings
        self.output_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, spatial_tokens, input_ids, attention_mask, context_dropout_prob=0.0):
        """
        Returns GPT-2 vocabulary logits for visual prefix + text tokens.

        Args:
            spatial_tokens: (Batch, 196, 768) Visual features from SigLIP.
            input_ids: (Batch, Seq_Len) Tokenized text captions.
            attention_mask: (Batch, Seq_Len) Mask for text tokens.
            context_dropout_prob: Probability of zeroing the visual prefix for
                                  each sample independently during training to 
                                  prevent over-reliance on the image. Text is never dropped.
        """
        spatial_tokens = spatial_tokens.to(dtype=self.resampler.latents.dtype)
        prefix_tokens  = self.resampler(spatial_tokens) * self.output_scale  # (Batch, 16, 768)

        # ── Per-sample context dropout: zero visual prefix, never text ───────
        self._last_context_dropout_stats = {'dropped': 0, 'total': prefix_tokens.size(0)}
        
        if self.training and context_dropout_prob > 0.0:
            drop_mask = torch.rand(
                prefix_tokens.size(0), 1, 1, device=prefix_tokens.device
            ) < context_dropout_prob
            
            prefix_tokens = prefix_tokens.masked_fill(drop_mask, 0.0)
            self._last_context_dropout_stats = {
                'dropped': int(drop_mask.sum().item()),
                'total': prefix_tokens.size(0),
            }

        text_embeddings = _get_gpt2_token_embedding_layer(self.gpt)(input_ids)
        prefix_tokens   = prefix_tokens.to(dtype=text_embeddings.dtype)
        
        # Concatenate visual prefix with the text embeddings
        inputs_embeds = torch.cat([prefix_tokens, text_embeddings], dim=1)

        # Extend attention mask to account for the new visual prefix tokens
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


def create_sr_clipcap_model(device="cpu", print_trainable=False):
    """
    Builds a fresh SR-ClipCap model: Initializes a Perceiver Resampler and 
    wraps a base GPT-2 decoder in LoRA adapters (targeting c_attn and c_proj).
    
    Args:
        device (str or torch.device): Target device for the model.
        print_trainable (bool): Whether to print PEFT parameter count statistics.
    """
    # Initialize tokenizer purely to get the padding token ID
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure LoRA for parameter-efficient adaptation
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=['c_attn', 'c_proj'],
        bias='none',
        fan_in_fan_out=True,
        inference_mode=False,
    )

    resampler = PerceiverResampler().to(device)
    gpt_decoder = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    
    gpt_decoder.config.use_cache = False
    gpt_decoder.config.pad_token_id = tokenizer.pad_token_id
    
    # Wrap GPT-2 decoder with LoRA adapters
    gpt_decoder = get_peft_model(gpt_decoder, lora_config)
    gpt_decoder.config.use_cache = False

    if print_trainable:
        gpt_decoder.print_trainable_parameters()

    fresh_model = SRClipCapModel(
        resampler=resampler,
        gpt_decoder=gpt_decoder,
        prefix_length=16,
    ).to(device)

    return fresh_model