"""
Training loop infrastructure shared by both VizWiz captioning architectures,
plus Architecture 1 (ClipCap) specific training adapters.

This module is import-safe: importing it has no side effects. Training only
starts when `train_model()` is explicitly called with a fully-constructed
model, dataloaders, optimizer, etc. — see `run_clipcap_training()` at the
bottom of this file for the Architecture 1 entry point.
"""

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.training.metrics import word_tokens
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu


# --------------------------------------------------------------------------
# Shared, architecture-agnostic training loop
# --------------------------------------------------------------------------

def metric_min_delta_for(metric_min_deltas, criterion_name):
    """Returns the min_delta configured for the currently active stopping metric."""
    aliases = {'bleu4': 'val_bleu4'}

    if isinstance(metric_min_deltas, dict):
        if criterion_name in metric_min_deltas:
            return float(metric_min_deltas[criterion_name])
        alias = aliases.get(criterion_name)
        if alias in metric_min_deltas:
            return float(metric_min_deltas[alias])
        return 0.0

    return float(metric_min_deltas)


def default_checkpoint_payload(model, state, epoch, train_loss, val_loss, val_metric,
                                best_metric, best_metric_epoch, histories, hparams):
    """Default checkpoint payload for models exposing a single state_dict()."""
    active_monitor = state.get('stopping_criterion', 'val_loss')
    return {
        'model_state': model.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'cider': val_metric,
        'monitor': active_monitor,
        'active_monitor': active_monitor,
        'current_stage': state.get('current_stage'),
        'metric_mode': state.get('metric_mode', 'min'),
        'best_metric_name': active_monitor,
        'best_metric_value': state.get('best_tracked_metric'),
        'best_metric_epoch': state.get('best_tracked_metric_epoch'),
        'best_cider': best_metric,
        'best_cider_epoch': best_metric_epoch,
        'optimizer': state['optimizer'].state_dict(),
        'scheduler': state['scheduler'].state_dict() if state.get('scheduler') is not None else None,
        'scaler': state['scaler'].state_dict() if state.get('scaler') is not None else None,
        'hparams': hparams,
        'train_losses': histories['train_losses'],
        'val_losses': histories['val_losses'],
        'val_ciders': histories['val_ciders'],
        'val_bleu4s': histories['val_bleu4s'],
        'context_drop_rates': histories.get('context_drop_rates', []),
        'learning_rates': histories['learning_rates'],
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, *,
                 train_epoch_fn, validate_epoch_fn, val_metric_fns, checkpoint_path,
                 num_epochs, patience, eval_start_epoch, hparams, device,
                 run_name='model', stage_callback=None, checkpoint_builder=None,
                 lr_group_index=0, val_loss_min_delta=None,
                 early_stop_min_delta=None, grad_clip_norm=0.5, seed=42):
    """
    Reusable training loop with per-stage stopping criteria, used by both
    Architecture 1 and Architecture 2 via their own train/validate/checkpoint
    adapter functions.

    Args:
        train_epoch_fn(model, loader, criterion, optimizer, scaler, device, scheduler, epoch) -> float
        validate_epoch_fn(model, loader, criterion, device, epoch) -> float
        val_metric_fns: dict mapping criterion name -> callable(model) -> float,
            e.g. {'val_cider': fn, 'val_bleu4': fn}. The active criterion is
            read from state['stopping_criterion'], set by `stage_callback`.
        stage_callback(epoch, model, state): called at epoch 0 (setup) and at
            stage transitions, to freeze/unfreeze parameters or rebuild the
            optimizer/scheduler. May be None for single-stage training.
        checkpoint_builder: function producing the dict saved to disk on each
            new best checkpoint. Defaults to `default_checkpoint_payload`.
        val_loss_min_delta / early_stop_min_delta: float, or dict mapping
            criterion names to their own min_delta. Either name is accepted;
            they're kept as separate parameters because Architecture 1 and 2
            historically used different names for the same concept.
        seed: base random seed, re-derived per-epoch for reproducibility.

    Returns:
        dict with the trained model, histories, and best-metric bookkeeping.
    """
    min_delta_source = val_loss_min_delta if val_loss_min_delta is not None else early_stop_min_delta
    metric_min_deltas = min_delta_source if min_delta_source is not None else 0.0

    checkpoint_path = Path(checkpoint_path)
    checkpoint_builder = checkpoint_builder or default_checkpoint_payload
    val_metric_fns = val_metric_fns or {}

    state = {
        'optimizer': optimizer,
        'scheduler': scheduler,
        'scaler': scaler,
        'stopping_criterion': 'val_loss',
        'metric_mode': 'min',
        'best_tracked_metric': float('inf'),
        'best_tracked_metric_epoch': -1,
        'patience_counter': 0,
        'patience_active': True,
    }

    histories = {
        'train_losses': [],
        'val_losses': [],
        'val_ciders': [],
        'val_bleu4s': [],
        'context_drop_rates': [],
        'learning_rates': [],
        'output_scales': [],
    }

    best_epoch = -1
    best_cider = float('-inf')
    best_cider_epoch = -1
    epoch_times = []

    print('=' * 65)
    print(f'Starting training: {run_name}')
    print(f'  Epochs: {num_epochs} | Patience: {patience} | Eval starts: epoch {eval_start_epoch + 1}')
    print(f'  Checkpoint: {checkpoint_path}')
    print('=' * 65)

    training_start = time.time()

    if stage_callback is not None:
        stage_callback(None, model, state)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        stage1_hard_cap_epoch = state.get('stage1_hard_cap_epoch')
        if (
            stage_callback is not None
            and state.get('current_stage') == 1
            and stage1_hard_cap_epoch is not None
            and epoch >= stage1_hard_cap_epoch
        ):
            print(f"  Stage 1 hard cap reached at epoch {epoch + 1}; transitioning to Stage 2.")
            stage_callback(epoch, model, state)
            incoming_mode = state.get('metric_mode', 'min')
            state['patience_counter'] = 0
            state['best_tracked_metric'] = float('-inf') if incoming_mode == 'max' else float('inf')
            state['best_tracked_metric_epoch'] = -1

        epoch_seed = seed + int(epoch)
        random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(epoch_seed)

        train_loss = train_epoch_fn(
            model, train_loader, criterion, state['optimizer'], state['scaler'],
            device, state.get('scheduler'), epoch,
        )
        val_loss = validate_epoch_fn(model, val_loader, criterion, device, epoch)

        val_cider = 0.0
        val_bleu4 = 0.0

        if epoch >= eval_start_epoch:
            criterion_name = state.get('stopping_criterion', 'val_loss')

            if criterion_name == 'val_cider' and 'val_cider' in val_metric_fns:
                val_cider = val_metric_fns['val_cider'](model)
            elif criterion_name in {'bleu4', 'val_bleu4'}:
                bleu4_fn = val_metric_fns.get('val_bleu4') or val_metric_fns.get('bleu4')
                if bleu4_fn is not None:
                    val_bleu4 = bleu4_fn(model)
                if 'val_cider' in val_metric_fns:
                    val_cider = val_metric_fns['val_cider'](model)
            elif criterion_name == 'val_loss':
                pass

            if val_cider > best_cider:
                best_cider = val_cider
                best_cider_epoch = epoch

        criterion_name = state.get('stopping_criterion', 'val_loss')
        if criterion_name == 'val_loss':
            current_metric = val_loss
        elif criterion_name == 'val_cider':
            current_metric = val_cider
        elif criterion_name in {'bleu4', 'val_bleu4'}:
            current_metric = val_bleu4
        else:
            current_metric = val_loss

        metric_mode = state.get('metric_mode', 'min')

        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        eta_secs = int(avg_epoch_time * (num_epochs - epoch - 1))
        eta_mins, eta_s = divmod(eta_secs, 60)
        eta_str = f'{eta_mins}m {eta_s:02d}s' if num_epochs - epoch - 1 > 0 else 'complete'

        lr_group = min(lr_group_index, len(state['optimizer'].param_groups) - 1)
        current_lr = state['optimizer'].param_groups[lr_group]['lr']
        output_scale_value = None
        if hasattr(model, 'output_scale'):
            output_scale_value = float(model.output_scale.detach().item())

        dropout_stats = state.pop('context_dropout_stats', None)
        if dropout_stats is None:
            context_drop_rate = None
        else:
            drop_total = max(1, int(dropout_stats.get('total', 0)))
            context_drop_rate = float(dropout_stats.get('dropped', 0)) / drop_total

        histories['train_losses'].append(train_loss)
        histories['val_losses'].append(val_loss)
        histories['val_ciders'].append(val_cider)
        histories['val_bleu4s'].append(val_bleu4)
        histories['context_drop_rates'].append(context_drop_rate)
        histories['learning_rates'].append(current_lr)
        histories['output_scales'].append(output_scale_value)

        output_scale_text = (
            f" | Output Scale: {output_scale_value:.3f}" if output_scale_value is not None else ""
        )
        context_drop_text = (
            f" | Context Drop Rate: {context_drop_rate * 100:.1f}%" if context_drop_rate is not None else ""
        )

        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | ETA: {eta_str} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val CIDEr: {val_cider:.4f} | Val BLEU-4: {val_bleu4:.4f} | LR: {current_lr:.2e}"
            f"{output_scale_text} | Stop: {criterion_name}{context_drop_text}"
        )

        if epoch >= eval_start_epoch:
            patience_counter = state.get('patience_counter', 0)
            best_tracked_metric = state.get(
                'best_tracked_metric', float('inf') if metric_mode == 'min' else float('-inf')
            )
            current_min_delta = metric_min_delta_for(metric_min_deltas, criterion_name)

            improved = (
                current_metric < best_tracked_metric - current_min_delta
                if metric_mode == 'min'
                else current_metric > best_tracked_metric + current_min_delta
            )

            cider_floor_ok = True
            cider_floor = None
            stage2_best_cider = state.get('stage2_best_cider')
            if criterion_name in {'bleu4', 'val_bleu4'} and stage2_best_cider is not None:
                cider_floor_delta = float(state.get('cider_floor_delta', 0.05))
                cider_floor = float(stage2_best_cider) - cider_floor_delta
                cider_floor_ok = val_cider >= cider_floor

            if improved and cider_floor_ok:
                state['best_tracked_metric'] = current_metric
                state['patience_counter'] = 0
                state['best_tracked_metric_epoch'] = epoch
                best_epoch = epoch

                payload = checkpoint_builder(
                    model, state, epoch, train_loss, val_loss, val_cider,
                    best_cider, best_cider_epoch, histories, hparams,
                )
                tmp_path = checkpoint_path.with_name(checkpoint_path.name + '.tmp')
                torch.save(payload, tmp_path)
                os.replace(tmp_path, checkpoint_path)
                print(f"  New best {criterion_name} ({current_metric:.4f}) — checkpoint saved.")

            elif improved:
                print(
                    f"  WARNING: {criterion_name} improved to {current_metric:.4f}, "
                    f"but Val CIDEr {val_cider:.4f} is below Stage-2 floor {cider_floor:.4f}; "
                    f"checkpoint not saved; patience remains {patience_counter}/{patience}."
                )
                state['patience_counter'] = patience_counter + 1
                if state['patience_counter'] >= patience:
                    print(f"  Early stopping triggered (CIDEr floor breached for "
                          f"{state['patience_counter']} consecutive epochs).")
                    state['stop_training'] = True

            else:
                state['patience_counter'] = patience_counter + 1
                print(f"  No {criterion_name} improvement; "
                      f"patience: {state['patience_counter']}/{patience}")

                if state['patience_counter'] >= patience:
                    current_stage = state.get('current_stage', 3)

                    if current_stage < 3:
                        print(f"  Patience exhausted in Stage {current_stage} — "
                              f"transitioning to Stage {current_stage + 1}.")

                        if stage_callback is not None:
                            stage_callback(epoch + 1, model, state)

                        incoming_mode = state.get('metric_mode', 'min')
                        state['patience_counter'] = 0
                        state['best_tracked_metric'] = (
                            float('-inf') if incoming_mode == 'max' else float('inf')
                        )
                        state['best_tracked_metric_epoch'] = -1

                    else:
                        print(f"  Early stopping triggered at epoch {epoch + 1} "
                              f"[{criterion_name}] — Stage 3 complete.")
                        break

            if state.get('stop_training'):
                break

        else:
            state['patience_counter'] = 0
            print(f"  (Eval skipped — epoch {epoch+1} < {eval_start_epoch + 1})")

    total_elapsed = time.time() - training_start
    total_mins, tot_s = divmod(int(total_elapsed), 60)
    print('=' * 65)
    print(f'Training complete: {run_name}')
    print(f'  Total time : {total_mins}m {tot_s:02d}s | Epochs: {len(histories["train_losses"])}/{num_epochs}')
    best_metric_name = state.get('stopping_criterion', 'val_loss')
    best_metric_value = state.get('best_tracked_metric')
    best_metric_epoch = state.get('best_tracked_metric_epoch', -1)
    if best_metric_epoch >= 0:
        print(f'  Best Tracked Metric [{best_metric_name}]: {best_metric_value:.4f} at epoch {best_metric_epoch + 1}')
    else:
        print('  No tracked metric improvement for active monitor.')
    if best_cider_epoch >= 0:
        print(f'  Best Val CIDEr : {best_cider:.4f} at epoch {best_cider_epoch + 1}')
    print('=' * 65)

    return {
        'model': model,
        'checkpoint_path': checkpoint_path,
        'best_metric_name': best_metric_name,
        'best_metric_value': best_metric_value,
        'best_metric_epoch': best_metric_epoch,
        'best_epoch': best_epoch,
        'best_cider_val': best_cider,
        'best_cider_epoch': best_cider_epoch,
        'train_losses': histories['train_losses'],
        'val_losses': histories['val_losses'],
        'val_ciders': histories['val_ciders'],
        'val_bleu4s': histories['val_bleu4s'],
        'context_drop_rates': histories['context_drop_rates'],
        'learning_rates': histories['learning_rates'],
        'output_scales': histories['output_scales'],
        'optimizer': state['optimizer'],
        'scheduler': state['scheduler'],
        'scaler': state['scaler'],
    }


def evaluate_bleu4_with_decoder(model, tokenizer, val_references, feature_loader, decoder_fn, *, device='cuda'):
    """Shared greedy-decode validation BLEU-4 routine, used as a Stage-3 stopping criterion."""
    from src.data.preprocessing import normalise_ref  # local import avoids a hard dependency for callers who don't need it

    model.eval()
    hypotheses = []
    references_list = []

    for img_id in tqdm(val_references.keys(), desc='Validation BLEU-4 decode', leave=False):
        feature = feature_loader(img_id)
        caption = decoder_fn(feature, model, tokenizer, device=device)
        hyp_words = word_tokens(caption)
        ref_words = [word_tokens(normalise_ref(r)) for r in val_references[img_id]]
        hypotheses.append(hyp_words)
        references_list.append(ref_words)

    if not hypotheses:
        return 0.0

    return float(corpus_bleu(
        references_list, hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1,
    ))


# --------------------------------------------------------------------------
# Architecture 1 (ClipCap) training adapters
# --------------------------------------------------------------------------

A1_DEFAULT_HPARAMS = {
    'patience': 5,
    'eval_start_epoch': 4,
    'warmup_epochs': 2,
    'warmup_temp_epochs': 4,
    'warmup_temp': 2.0,
    'grad_clip_norm': 0.5,
}


def compute_loss(logits, labels, criterion, epoch, warmup_epochs=4, temp=2.0):
    """Computes shifted causal language-model loss, with optional temperature warmup."""
    if epoch < warmup_epochs:
        logits = logits / temp

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return criterion(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def train_one_epoch_a1(model, loader, criterion, optimizer, scaler, device, scheduler=None, epoch=0,
                        warmup_temp_epochs=4, warmup_temp=2.0, grad_clip_norm=0.5):
    """Architecture 1 batch-training adapter for the shared train_model loop."""
    model.train()
    total_loss = 0.0
    steps_run = 0
    device_type = device.type if hasattr(device, 'type') else str(device)
    batch_bar = tqdm(loader, desc='Training', leave=False, unit='batch')

    for batch in batch_bar:
        clip_embeds = batch['clip_embed'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(device_type == 'cuda')):
            logits = model(clip_embeds=clip_embeds, input_ids=input_ids, attention_mask=attention_mask)
            prefix_labels = torch.full(
                (clip_embeds.shape[0], model.prefix_length), -100,
                dtype=torch.long, device=device,
            )
            full_labels = torch.cat([prefix_labels, labels], dim=1)
            loss = compute_loss(logits, full_labels, criterion, epoch, warmup_temp_epochs, warmup_temp)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        steps_run += 1
        batch_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(steps_run, 1)


@torch.inference_mode()
def validate_one_epoch_a1(model, loader, criterion, device, epoch=0):
    """Architecture 1 validation adapter for the shared train_model loop."""
    model.eval()
    device_type = device.type if hasattr(device, 'type') else str(device)
    total_loss = 0.0
    steps_run = 0
    batch_bar = tqdm(loader, desc='Validation loss', leave=False, unit='batch')

    for batch in batch_bar:
        clip_embeds = batch['clip_embed'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.amp.autocast('cuda', enabled=(device_type == 'cuda')):
            logits = model(clip_embeds=clip_embeds, input_ids=input_ids, attention_mask=attention_mask)
            prefix_labels = torch.full(
                (clip_embeds.shape[0], model.prefix_length), -100,
                dtype=torch.long, device=device,
            )
            full_labels = torch.cat([prefix_labels, labels], dim=1)
            loss = compute_loss(logits, full_labels, criterion, epoch=0, warmup_epochs=0, temp=1.0)

        total_loss += loss.item()
        steps_run += 1
        batch_bar.set_postfix(val_loss=f"{loss.item():.4f}")

    return total_loss / max(steps_run, 1)


def make_a1_stage_callback(mapping_lr, gpt2_lr, warmup_epochs, train_loader, total_steps):
    """
    Builds the Architecture 1 stage callback: starts with GPT-2 frozen
    (mapping network trains alone), then unfreezes GPT-2 and rebuilds the
    optimizer/scheduler for joint fine-tuning at `warmup_epochs`.

    Args:
        train_loader: needed to compute remaining scheduler steps at the
            unfreeze transition.
        total_steps: total training steps planned across all epochs.
    """
    from src.models.clipcap import set_gpt2_trainable
    from transformers import get_cosine_schedule_with_warmup

    def stage_callback(epoch, model, state):
        if epoch == 0:
            set_gpt2_trainable(model, trainable=False)
            print('  GPT-2 fine-tune layers FROZEN; mapping network trains alone')

        elif epoch == warmup_epochs:
            set_gpt2_trainable(model, trainable=True)
            state['optimizer'] = torch.optim.AdamW([
                {'params': model.mapping_network.parameters(), 'lr': mapping_lr},
                {'params': [p for p in model.gpt.parameters() if p.requires_grad], 'lr': gpt2_lr},
            ], weight_decay=0.01)

            steps_done = epoch * len(train_loader)
            steps_remaining = total_steps - steps_done
            state['scheduler'] = get_cosine_schedule_with_warmup(
                state['optimizer'],
                num_warmup_steps=0,
                num_training_steps=steps_remaining,
            )
            print('  GPT-2 fine-tune layers UNFROZEN; optimiser rebuilt; joint training begins')

    return stage_callback


def build_a1_checkpoint(model, state, epoch, train_loss, val_loss, val_metric,
                         best_metric, best_metric_epoch, histories, hparams):
    """Architecture 1 checkpoint payload — saves mapping_net / gpt2 state dicts separately."""
    active_monitor = state.get('stopping_criterion', 'val_loss')
    return {
        'mapping_net': model.mapping_network.state_dict(),
        'gpt2': model.gpt.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'cider': val_metric,
        'monitor': active_monitor,
        'active_monitor': active_monitor,
        'current_stage': state.get('current_stage'),
        'metric_mode': state.get('metric_mode', 'min'),
        'best_metric_name': active_monitor,
        'best_metric_value': state.get('best_tracked_metric'),
        'best_metric_epoch': state.get('best_tracked_metric_epoch'),
        'best_cider': best_metric,
        'best_cider_epoch': best_metric_epoch,
        'optimizer': state['optimizer'].state_dict(),
        'scheduler': state['scheduler'].state_dict() if state.get('scheduler') is not None else None,
        'scaler': state['scaler'].state_dict() if state.get('scaler') is not None else None,
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        'hparams': hparams,
        'train_losses': histories['train_losses'],
        'val_losses': histories['val_losses'],
        'val_ciders': histories['val_ciders'],
        'learning_rates': histories['learning_rates'],
    }


def run_clipcap_training(model, train_loader, val_loader, tokenizer, val_features, val_references, *,
                          checkpoint_path, device, greedy_decode_fn, evaluate_cider_fn,
                          mapping_lr=1e-4, gpt2_lr=5e-5,
                          epochs=30, batch_size=64, prefix_length=10, freeze_gpt2_layers=6,
                          warmup_steps=500, patience=5, eval_start_epoch=4,
                          warmup_epochs=2, warmup_temp_epochs=4, warmup_temp=2.0,
                          max_caption_len=20, val_loss_min_delta=None):
    """
    Architecture 1 (ClipCap) training entry point. Constructs the criterion,
    optimizer, scheduler, stage callback, and hparams dict, then delegates
    to the shared `train_model()` loop.

    This is the function equivalent of the notebook's Architecture 1
    training cell — call this instead of copy-pasting the setup each time.

    Args:
        greedy_decode_fn: decoding function from src/inference.py (built in
            Step 5), e.g. `inference.greedy_decode`. Passed in explicitly
            rather than imported here, to avoid a circular import between
            src/training/train.py and src/inference.py.
        evaluate_cider_fn: validation CIDEr routine from src/inference.py,
            e.g. `inference.evaluate_cider_with_decoder`.

    Returns the same dict as `train_model()`.
    """
    from transformers import get_cosine_schedule_with_warmup

    total_steps = epochs * len(train_loader)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.mapping_network.parameters(), lr=mapping_lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    stage_callback = make_a1_stage_callback(mapping_lr, gpt2_lr, warmup_epochs, train_loader, total_steps)

    hparams = {
        'mapping_lr': mapping_lr,
        'gpt2_lr': gpt2_lr,
        'epochs': epochs,
        'warmup_steps': warmup_steps,
        'batch_size': batch_size,
        'prefix_length': prefix_length,
        'freeze_gpt2_layers': freeze_gpt2_layers,
        'warmup_epochs': warmup_epochs,
        'warmup_temp_epochs': warmup_temp_epochs,
        'warmup_temp': warmup_temp,
        'max_caption_len': max_caption_len,
        'monitor': 'val_loss',
    }

    print('=' * 65)
    print('Architecture 1 training configuration')
    print(f'  Epochs: {epochs} | Batch size: {batch_size} | Patience: {patience}')
    print(f'  Warmup steps: {warmup_steps} | Total steps: {total_steps:,}')
    print(f'  Mapping LR: {mapping_lr:.0e} | GPT-2 LR: {gpt2_lr:.0e}')
    print(f'  Max caption length: {max_caption_len}')
    print('=' * 65)

    def train_epoch_fn(m, loader, crit, opt, scl, dev, sched, ep):
        return train_one_epoch_a1(
            m, loader, crit, opt, scl, dev, sched, ep,
            warmup_temp_epochs=warmup_temp_epochs, warmup_temp=warmup_temp,
        )

    result = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler, scaler,
        train_epoch_fn=train_epoch_fn,
        validate_epoch_fn=validate_one_epoch_a1,
        val_metric_fns={
            'val_cider': lambda m: evaluate_cider_fn(
                m, tokenizer, val_references,
                feature_loader=lambda img_id: val_features[img_id],
                decoder_fn=greedy_decode_fn, device=device,
            ),
        },
        checkpoint_path=checkpoint_path,
        num_epochs=epochs,
        patience=patience,
        eval_start_epoch=eval_start_epoch,
        hparams=hparams,
        device=device,
        run_name='Architecture 1 ClipCap',
        stage_callback=stage_callback,
        checkpoint_builder=build_a1_checkpoint,
        lr_group_index=0,
        val_loss_min_delta=val_loss_min_delta,
    )

    return result