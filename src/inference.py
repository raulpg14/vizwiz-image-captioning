"""
Inference and decoding logic for Architecture 1 (ClipCap) and 
Architecture 2 (SR-ClipCap).

Includes greedy decoding, length-normalized beam search, penalty applications
(repetition, structural tokens), and the shared CIDEr evaluation routine.
"""

import torch
from tqdm import tqdm
from pycocoevalcap.cider.cider import Cider

from src.data.preprocessing import normalise_ref
from src.training.metrics import assert_exact_key_match


DEFAULT_GENERATION_CONFIG = {
    'max_caption_len': 20,
    'beam_size_default': 5,
    'length_norm_beta': 0.7,
    'min_new_tokens': 3,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0,
    'no_repeat_ngram': 0,
    'blocked_token_ids': [],
}


def get_gpt2_token_embedding_layer(gpt_decoder):
    """Return GPT-2 token embeddings from a base or PEFT-wrapped decoder."""
    base_model = gpt_decoder.get_base_model() if hasattr(gpt_decoder, 'get_base_model') else gpt_decoder
    return base_model.transformer.wte


def build_clip_prefix(clip_feat, model, device='cuda'):
    """Build Architecture 1 visual prefix tokens from one CLIP embedding."""
    return model.mapping_network(clip_feat.unsqueeze(0).to(device))


def build_spatial_prefix(spatial_tensor, model, device='cuda'):
    """Build Architecture 2 visual prefix tokens from one SigLIP spatial tensor."""
    if spatial_tensor.ndim == 2:
        spatial_tensor = spatial_tensor.unsqueeze(0)
    elif spatial_tensor.ndim != 3:
        raise ValueError(f"Expected [196, 768] or [1, 196, 768], got {tuple(spatial_tensor.shape)}")

    spatial_tensor = spatial_tensor.to(device=device, dtype=model.resampler.latents.dtype)
    return model.resampler(spatial_tensor) * model.output_scale


def apply_generation_penalties(logits, generated_ids, config=None):
    """Apply centralized repetition and structural-token penalties."""
    config = config or DEFAULT_GENERATION_CONFIG
    if generated_ids:
        token_counts = {}
        for tok in generated_ids:
            token_counts[tok] = token_counts.get(tok, 0) + 1

        for tok_id, count in token_counts.items():
            logits[0, tok_id] -= config.get('frequency_penalty', 0.0) * count
            logits[0, tok_id] -= config.get('presence_penalty', 0.0)

        n = config.get('no_repeat_ngram', 0)
        if n > 0 and len(generated_ids) >= n - 1:
            ngram_prefix = tuple(generated_ids[-(n - 1):])
            for i in range(len(generated_ids) - (n - 1)):
                if tuple(generated_ids[i:i + n - 1]) == ngram_prefix:
                    blocked_token = generated_ids[i + n - 1]
                    logits[0, blocked_token] = float('-inf')

    for tok_id in config.get('blocked_token_ids', []):
        logits[0, tok_id] = float('-inf')

    return logits


def normalise_beam_score(seq, score, tokenizer, beta=None):
    """Normalize beam scores by sequence length to prevent short-sequence bias."""
    beta = DEFAULT_GENERATION_CONFIG['length_norm_beta'] if beta is None else beta
    gen_len = len(seq) - 1
    if seq[-1] == tokenizer.eos_token_id:
        gen_len -= 1
    return score / max(gen_len, 1) ** beta


def decode_with_prefix(feature, model, tokenizer, prefix_builder, *, strategy='beam', beam_size=None,
                       max_len=None, beta=None, min_new_tokens=None, device='cuda'):
    """Shared greedy/beam decoder parameterized by an architecture-specific prefix builder."""
    config = DEFAULT_GENERATION_CONFIG
    beam_size = beam_size or config['beam_size_default']
    max_len = max_len or config['max_caption_len']
    beta = beta if beta is not None else config['length_norm_beta']
    min_new_tokens = min_new_tokens if min_new_tokens is not None else config['min_new_tokens']

    model.eval()
    token_embedding_layer = get_gpt2_token_embedding_layer(model.gpt)

    with torch.inference_mode():
        prefix = prefix_builder(feature, model, device=device)

    if strategy == 'greedy':
        generated = []
        input_ids = torch.tensor([[tokenizer.eos_token_id]], device=device)

        with torch.inference_mode():
            for _ in range(max_len):
                token_embeds = token_embedding_layer(input_ids)
                full_seq = torch.cat([prefix.to(dtype=token_embeds.dtype), token_embeds], dim=1)
                full_attention_mask = torch.ones(full_seq.shape[:2], dtype=torch.long, device=device)
                
                logits = model.gpt(
                    inputs_embeds=full_seq,
                    attention_mask=full_attention_mask,
                    use_cache=False,
                    return_dict=True,
                ).logits[:, -1, :]
                
                logits = apply_generation_penalties(logits, generated, config=config)
                next_token = logits.argmax(dim=-1, keepdim=True)

                if next_token.item() == tokenizer.eos_token_id:
                    break

                input_ids = torch.cat([input_ids, next_token], dim=1)
                generated.append(next_token.item())

        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    if strategy != 'beam':
        raise ValueError(f"Unknown decoding strategy: {strategy}")

    beams = [([tokenizer.eos_token_id], 0.0)]
    completed = []

    for _ in range(max_len):
        candidates = []

        for seq, score in beams:
            input_ids = torch.tensor([seq], device=device)
            with torch.inference_mode():
                token_embeds = token_embedding_layer(input_ids)
                full_seq = torch.cat([prefix.to(dtype=token_embeds.dtype), token_embeds], dim=1)
                full_attention_mask = torch.ones(full_seq.shape[:2], dtype=torch.long, device=device)
                
                logits = model.gpt(
                    inputs_embeds=full_seq,
                    attention_mask=full_attention_mask,
                    use_cache=False,
                    return_dict=True,
                ).logits[:, -1, :]

            generated = seq[1:]
            logits = apply_generation_penalties(logits, generated, config=config)

            if len(generated) < min_new_tokens:
                logits[0, tokenizer.eos_token_id] = float('-inf')

            log_probs = torch.log_softmax(logits, dim=-1)
            top_log_probs, top_ids = log_probs.topk(beam_size)

            for log_p, tok_id in zip(top_log_probs[0], top_ids[0]):
                new_seq = seq + [tok_id.item()]
                new_score = score + log_p.item()

                if tok_id.item() == tokenizer.eos_token_id:
                    if len(new_seq) > 1 + min_new_tokens:
                        completed.append((new_seq, new_score))
                else:
                    candidates.append((new_seq, new_score))

        if not candidates:
            break

        beams = sorted(
            candidates,
            key=lambda x: normalise_beam_score(x[0], x[1], tokenizer, beta=beta),
            reverse=True,
        )[:beam_size]

    all_beams = beams + completed
    best_seq = max(all_beams, key=lambda x: normalise_beam_score(x[0], x[1], tokenizer, beta=beta))[0]
    
    return tokenizer.decode(best_seq, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------
# Architecture 1 Decoding Wrappers
# --------------------------------------------------------------------------

def beam_search(clip_feat, model, tokenizer, beam_size=None, max_len=None, beta=None, min_new_tokens=None, device='cuda'):
    """Run beam search decoding for Architecture 1 (ClipCap)."""
    return decode_with_prefix(
        clip_feat, model, tokenizer, build_clip_prefix,
        strategy='beam', beam_size=beam_size, max_len=max_len,
        beta=beta, min_new_tokens=min_new_tokens, device=device,
    )


def greedy_decode(clip_feat, model, tokenizer, max_len=None, device='cuda'):
    """Run greedy decoding for Architecture 1 (ClipCap)."""
    return decode_with_prefix(
        clip_feat, model, tokenizer, build_clip_prefix,
        strategy='greedy', max_len=max_len, device=device,
    )


# --------------------------------------------------------------------------
# Architecture 2 Decoding Wrappers
# --------------------------------------------------------------------------

def beam_search_sr(spatial_tensor, model, tokenizer, beam_size=None, max_len=None, beta=None, min_new_tokens=None, device='cuda'):
    """Run beam search decoding for Architecture 2 (SR-ClipCap)."""
    return decode_with_prefix(
        spatial_tensor, model, tokenizer, build_spatial_prefix,
        strategy='beam', beam_size=beam_size, max_len=max_len,
        beta=beta, min_new_tokens=min_new_tokens, device=device,
    )


def greedy_decode_sr(spatial_tensor, model, tokenizer, max_len=None, device='cuda'):
    """Run greedy decoding for Architecture 2 (SR-ClipCap)."""
    return decode_with_prefix(
        spatial_tensor, model, tokenizer, build_spatial_prefix,
        strategy='greedy', max_len=max_len, device=device,
    )


# --------------------------------------------------------------------------
# Evaluation Hooks
# --------------------------------------------------------------------------

def evaluate_cider_with_decoder(model, tokenizer, val_references, feature_loader, decoder_fn, *, device='cuda'):
    """
    Shared validation CIDEr routine. Loops over a validation set, decodes predictions, 
    and leverages pycocoevalcap to compute the corpus CIDEr score.
    """
    model.eval()
    predictions = {}

    for img_id in tqdm(val_references.keys(), desc='Validation greedy decode', leave=False):
        feature = feature_loader(img_id)
        predictions[str(img_id)] = decoder_fn(feature, model, tokenizer, device=device)

    gts = {str(k): [normalise_ref(r) for r in v] for k, v in val_references.items()}
    res = {str(k): [v] for k, v in predictions.items()}
    assert_exact_key_match(gts, res, 'val CIDEr (greedy)')

    score, _ = Cider().compute_score(gts, res)
    return float(score)