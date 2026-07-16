"""
Evaluation metrics and training-curve plotting, shared by both VizWiz
captioning architectures.

This module is import-safe: importing it has no side effects. Plotting
functions save to a caller-provided path rather than a hardcoded directory.
"""

import re

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

# Word-level tokenizer used consistently across BLEU scoring and failure
# classification, so both use identical tokenization.
WORD_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")

DEFAULT_BLEU_WEIGHTS = {
    'BLEU-1': (1.0, 0.0, 0.0, 0.0),
    'BLEU-2': (0.5, 0.5, 0.0, 0.0),
    'BLEU-3': (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
    'BLEU-4': (0.25, 0.25, 0.25, 0.25),
}
METRIC_ORDER = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'CIDEr', 'ROUGE-L']


def assert_exact_key_match(gts, res, context=''):
    """Raises ValueError if ground-truth and prediction dictionaries have mismatched keys."""
    gts_keys = set(map(str, gts.keys()))
    res_keys = set(map(str, res.keys()))

    if gts_keys == res_keys:
        return

    missing_in_res = sorted(gts_keys - res_keys)
    extra_in_res = sorted(res_keys - gts_keys)

    raise ValueError(
        f"Key mismatch in {context}:\n"
        f"  Missing in predictions : {missing_in_res[:5]} "
        f"({len(missing_in_res)} total)\n"
        f"  Extra in predictions   : {extra_in_res[:5]} "
        f"({len(extra_in_res)} total)\n"
        f"  GTS count: {len(gts_keys)} | RES count: {len(res_keys)}"
    )


def metric_text(value, tokenizer=None):
    """
    Converts a metric input (string, dict, list, or token ids) into plain
    caption text without sanitizing content.
    """
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        for key in ('caption', 'text', 'sentence', 'prediction'):
            if key in value:
                return metric_text(value[key], tokenizer=tokenizer)
        return ' '.join(metric_text(v, tokenizer=tokenizer) for v in value.values())

    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return metric_text(value[0], tokenizer=tokenizer)

        if all(isinstance(x, (int, np.integer)) for x in value):
            if tokenizer is None:
                raise ValueError('Token-id prediction requires a tokenizer.')
            return tokenizer.decode(list(map(int, value)), skip_special_tokens=True)

        return ' '.join(metric_text(x, tokenizer=tokenizer) for x in value)

    return str(value)


def word_tokens(text):
    """Tokenizes caption text using the project's word-level regex."""
    return WORD_TOKEN_RE.findall(metric_text(text).lower())


def compute_nltk_bleu_scores(gts, res, tokenizer, bleu_weights=None):
    """
    Computes BLEU-1 through BLEU-4 using NLTK's corpus_bleu.

    Args:
        gts: dict {image_id: [reference_caption, ...]}
        res: dict {image_id: predicted_caption}
        tokenizer: used only if predictions arrive as raw token ids.
        bleu_weights: optional override of the default n-gram weight dict.

    Raises:
        ValueError if keys mismatch, or if tokenization produces empty
        hypotheses/references, or if corpus-wide unigram overlap is zero
        (a strong signal of a tokenization/format bug rather than a
        genuinely bad model).
    """
    assert_exact_key_match(gts, res, 'BLEU')
    bleu_weights = bleu_weights or DEFAULT_BLEU_WEIGHTS

    list_of_references = []
    hypotheses = []
    debug_rows = []

    for img_id in sorted(gts.keys()):
        ref_values = gts[img_id]
        pred_value = res[img_id]

        if not isinstance(ref_values, (list, tuple)):
            ref_values = [ref_values]

        if isinstance(pred_value, (list, tuple)) and len(pred_value) == 1:
            pred_value = pred_value[0]

        ref_tokens = [word_tokens(metric_text(ref)) for ref in ref_values]
        hyp_tokens = word_tokens(metric_text(pred_value, tokenizer=tokenizer))

        list_of_references.append(ref_tokens)
        hypotheses.append(hyp_tokens)
        debug_rows.append((img_id, hyp_tokens, ref_tokens))

    empty_hypotheses = [img_id for img_id, hyp, _ in debug_rows if len(hyp) == 0]
    empty_references = [
        img_id for img_id, _, refs in debug_rows
        if not refs or all(len(ref) == 0 for ref in refs)
    ]

    if empty_hypotheses:
        raise ValueError(
            f"BLEU tokenization produced empty predictions for "
            f"{len(empty_hypotheses)} images. Examples: {empty_hypotheses[:5]}"
        )

    if empty_references:
        raise ValueError(
            f"BLEU tokenization produced empty references for "
            f"{len(empty_references)} images. Examples: {empty_references[:5]}"
        )

    unigram_overlap = 0
    for _, hyp, refs in debug_rows:
        hyp_set = set(hyp)
        ref_set = set(token for ref in refs for token in ref)
        unigram_overlap += len(hyp_set & ref_set)

    if unigram_overlap == 0:
        print('DEBUG: first 5 BLEU tokenization examples')
        for img_id, hyp, refs in debug_rows[:5]:
            print(f"\nImage: {img_id}")
            print(f"Prediction tokens: {hyp}")
            print(f"Reference tokens : {refs[:2]}")

        raise ValueError(
            'BLEU unigram overlap is zero across the corpus. '
            'This indicates a tokenization/input-format bug, not a normal metric result.'
        )

    smoothing = SmoothingFunction().method1

    return {
        metric: corpus_bleu(
            list_of_references,
            hypotheses,
            weights=tuple(weights),
            smoothing_function=smoothing,
        )
        for metric, weights in bleu_weights.items()
    }


def classify_caption_failure(pred, refs, tokenizer, config=None):
    """
    Classifies a predicted caption into one of: 'Truncated', 'Generic / Vague',
    'Hallucination', or 'Plausible', using fixed hardcoded thresholds (not
    dataset-derived percentiles, so results are consistent across runs).

    Args:
        config: dict with keys 'max_caption_tokens', 'generic_max_words',
            'hallucination_min_unmatched_words', 'hallucination_max_shared_words'.
            Falls back to sensible defaults if not provided.
    """
    config = config or {
        'max_caption_tokens': 20,
        'generic_max_words': 4,
        'hallucination_min_unmatched_words': 3,
        'hallucination_max_shared_words': 1,
    }

    pred_words = word_tokens(pred)
    pred_word_set = set(pred_words)
    ref_word_set = {token for ref in refs for token in word_tokens(ref)}

    unmatched_words = len(pred_word_set - ref_word_set)
    shared_words = len(pred_word_set & ref_word_set)

    if len(tokenizer.encode(pred)) >= config['max_caption_tokens']:
        return 'Truncated'

    if len(pred_words) <= config['generic_max_words']:
        return 'Generic / Vague'

    if (
        unmatched_words >= config['hallucination_min_unmatched_words']
        and shared_words <= config['hallucination_max_shared_words']
    ):
        return 'Hallucination'

    return 'Plausible'


# --------------------------------------------------------------------------
# Training-curve plotting
# --------------------------------------------------------------------------

def plot_training_curves(train_losses, val_losses, val_ciders, learning_rates,
                          best_epoch_idx, save_path, title=None, show=False):
    """
    Plots 4-panel training diagnostics (train loss, val loss, val CIDEr,
    learning rate) and saves to `save_path`.

    Args:
        save_path: where to save the figure (e.g. 'results/training_curves_arch1.png').
        title: figure suptitle; defaults to a generic caption-model title.
        show: if True, also calls plt.show() (only useful in a notebook/interactive session).
    """
    import matplotlib.pyplot as plt  # imported lazily so this module doesn't require
                                       # matplotlib unless plotting is actually used

    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        title or 'Training Diagnostics',
        fontsize=13, fontweight='bold',
    )

    axes[0].plot(epochs, train_losses, color='#1f77b4', marker='o', linewidth=2, markersize=4)
    axes[0].set_title('Training Loss', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(epochs, val_losses, color='#ff7f0e', marker='o', linewidth=2, markersize=4, label='Val loss')
    if best_epoch_idx is not None and 0 <= best_epoch_idx < len(val_losses):
        axes[1].plot(
            best_epoch_idx + 1, val_losses[best_epoch_idx],
            color='#d62728', marker='*', markersize=16, zorder=5,
            label=f'Best checkpoint ({best_epoch_idx + 1}, loss={val_losses[best_epoch_idx]:.4f})'
        )
    axes[1].set_title('Validation Loss', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Cross-Entropy Loss')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].plot(epochs, val_ciders, color='#2ca02c', marker='o', linewidth=2, markersize=4, label='Val CIDEr (greedy)')
    axes[2].set_title('Validation CIDEr', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('CIDEr Score')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, linestyle='--', alpha=0.6)

    axes[3].plot(epochs, learning_rates, color='#9467bd', linestyle='-', linewidth=2)
    axes[3].set_title('Learning Rate', fontsize=13, fontweight='bold')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Learning Rate')
    axes[3].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axes[3].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)