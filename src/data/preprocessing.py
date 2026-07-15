"""
Data staging, vocabulary/reference loading, and tokenizer/processor setup
for the VizWiz image captioning pipeline.

This module is import-safe: importing it has no side effects. All file I/O,
Drive staging, and environment detection happen only when the relevant
function is explicitly called (e.g. from a notebook or `src/training/train.py`).
"""

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, CLIPProcessor


# --------------------------------------------------------------------------
# Environment & configuration
# --------------------------------------------------------------------------

def detect_colab():
    """Returns True if running inside Google Colab, False otherwise."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def build_config(project_dir=None):
    """
    Builds the central environment/path configuration dictionary.

    Args:
        project_dir: Root directory for local (non-Colab) runs. Defaults to
            the current working directory if not provided.

    Returns:
        A dict with 'in_colab', 'paths', and pipeline hyperparameters
        (generation, metrics, classification thresholds, training settings).
    """
    in_colab = detect_colab()
    project_dir = Path(project_dir or Path.cwd()).resolve()

    colab_drive_dir = Path(
        '/content/drive/MyDrive/UTS/01.MDSI/05.DeepLearning/02.Assignments/03.AT3'
    )

    config = {
        'in_colab': in_colab,
        'data': {
            'phase1_artifacts': [
                'word2idx.json',
                'splits.json',
                'captions_clean.parquet',
                'references_by_image.json',
            ],
            'min_expected_images': 7750,
        },
        'generation': {
            'max_caption_len': 20,
            'beam_size_default': 5,
            'beam_size_test': 7,
            'length_norm_beta': 0.7,
            'min_new_tokens': 3,
            'no_repeat_ngram': 3,
            'frequency_penalty': 0.15,
            'presence_penalty': 0.6,
            'blocked_token_ids': [198, 628],
        },
        'metrics': {
            'bleu_weights': {
                'BLEU-1': (1.0, 0.0, 0.0, 0.0),
                'BLEU-2': (0.5, 0.5, 0.0, 0.0),
                'BLEU-3': (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
                'BLEU-4': (0.25, 0.25, 0.25, 0.25),
            },
            'metric_order': ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'CIDEr', 'ROUGE-L'],
        },
        'classification': {
            'max_caption_tokens': 20,
            'generic_max_words': 4,
            'hallucination_min_unmatched_words': 3,
            'hallucination_max_shared_words': 1,
            'severity_cutoffs': {
                'generic_vague': 0.08,
                'truncated': 0.15,
                'hallucination': 0.25,
            },
        },
        'features': {
            'tensor_cache_max_items': 2048,
            'manifest_hash_bytes': 65536,
        },
        'training': {
            'early_stop_min_delta': {
                'val_loss': 0.005,
                'val_cider': 0.01,
                'val_bleu4': 0.001,
            },
            'grad_clip_norm': 0.5,
        },
    }

    if in_colab:
        runtime_root = Path('/content/vizwiz')
        artifacts_dir = Path('/content/artifacts')
        spatial_features_dir = Path('/content/spatial_features')
        checkpoints_dir = colab_drive_dir / 'checkpoints'
        drive_val_zip = colab_drive_dir / 'val.zip'
        drive_artifacts_dir = colab_drive_dir / 'artifacts'
    else:
        runtime_root = project_dir
        artifacts_dir = project_dir / 'artifacts'
        spatial_features_dir = project_dir / 'spatial_features'
        checkpoints_dir = project_dir / 'checkpoints'
        drive_val_zip = None
        drive_artifacts_dir = None

    config['paths'] = {
        'project_dir': project_dir,
        'drive_dir': colab_drive_dir if in_colab else None,
        'drive_val_zip': drive_val_zip,
        'drive_artifacts_dir': drive_artifacts_dir,
        'runtime_root': runtime_root,
        'local_images_dir': runtime_root / 'images',
        'artifacts_dir': artifacts_dir,
        'checkpoints_dir': checkpoints_dir,
        'spatial_features_dir': spatial_features_dir,
    }

    return config


def ensure_directories(config):
    """Creates the runtime directories referenced in `config['paths']` if missing."""
    for key in ('local_images_dir', 'artifacts_dir', 'checkpoints_dir', 'spatial_features_dir'):
        Path(config['paths'][key]).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Data staging (Colab Drive -> local runtime, or local passthrough)
# --------------------------------------------------------------------------

def copy_phase1_artifacts(source_dir, target_dir, artifacts):
    """Copies required Phase 1 artifacts (vocab, splits, captions, references) to local storage."""
    print('Copying Phase 1 artifacts to local runtime storage...')
    for artifact in artifacts:
        src = Path(source_dir) / artifact
        dst = Path(target_dir) / artifact
        if not src.exists():
            raise FileNotFoundError(f'Missing required Phase 1 artifact: {src}')
        shutil.copy2(src, dst)
    print('Artifacts ready.')


def stage_images_from_zip(zip_path, target_dir, min_expected_images, staging_zip_path):
    """
    Extracts VizWiz images into local runtime storage. Skips extraction if
    enough images are already present (avoids re-staging on repeated runs).
    """
    target_dir = Path(target_dir)
    existing_jpgs = list(target_dir.glob('**/*.jpg'))
    if len(existing_jpgs) >= min_expected_images:
        print(f'Images already present ({len(existing_jpgs)} files); skipping extraction.')
        return

    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f'Missing required image archive: {zip_path}')

    print('Copying and extracting images to local runtime storage...')
    staging_zip_path = Path(staging_zip_path)
    shutil.copy2(zip_path, staging_zip_path)

    with zipfile.ZipFile(staging_zip_path, 'r') as zf:
        zf.extractall(target_dir)

    staging_zip_path.unlink(missing_ok=True)
    print('Image extraction complete.')


def find_images_dir(base_dir):
    """Finds the directory that directly contains VizWiz JPG files under `base_dir`."""
    base_dir = Path(base_dir)
    if list(base_dir.glob('*.jpg')):
        return base_dir
    for jpg in sorted(base_dir.rglob('*.jpg')):
        return jpg.parent
    raise FileNotFoundError(f'No .jpg files found under {base_dir}')


def stage_data(config):
    """
    Runs the full staging pipeline: copies Phase 1 artifacts and extracts
    images into local runtime storage. On Colab this pulls from Drive; when
    running locally it validates that the expected files already exist under
    `config['paths']['project_dir']` and raises a clear error if not.

    Returns the resolved images directory path.
    """
    ensure_directories(config)
    paths = config['paths']
    in_colab = config['in_colab']
    phase1_artifacts = config['data']['phase1_artifacts']

    if in_colab:
        copy_phase1_artifacts(paths['drive_artifacts_dir'], paths['artifacts_dir'], phase1_artifacts)
        stage_images_from_zip(
            zip_path=paths['drive_val_zip'],
            target_dir=paths['local_images_dir'],
            min_expected_images=config['data']['min_expected_images'],
            staging_zip_path=Path('/content/val.zip'),
        )
        images_dir = find_images_dir(paths['local_images_dir'])
    else:
        project_dir = paths['project_dir']
        local_artifacts = paths['artifacts_dir']

        if all((local_artifacts / a).exists() for a in phase1_artifacts):
            paths['artifacts_dir'] = local_artifacts
        elif all((project_dir / a).exists() for a in phase1_artifacts):
            paths['artifacts_dir'] = project_dir
        else:
            raise FileNotFoundError(
                'Could not find Phase 1 artifacts locally. Expected them under '
                f'{local_artifacts} or {project_dir}. This project requires the '
                'VizWiz dataset artifacts to be downloaded separately — see README.'
            )
        images_dir = find_images_dir(project_dir)

    print(f"Running in Colab   : {in_colab}")
    print(f"Local artifacts dir: {paths['artifacts_dir']}")
    print(f"Images resolved to : {images_dir}")
    print(f"Checkpoints dir    : {paths['checkpoints_dir']}")
    print(f"Spatial cache dir  : {paths['spatial_features_dir']}")

    return images_dir


# --------------------------------------------------------------------------
# Vocabulary, splits, and reference caption loading
# --------------------------------------------------------------------------

def reference_to_string(ref):
    """Converts one raw reference caption (string, list, or Series) into scorer-ready text."""
    if isinstance(ref, str):
        return ref
    if isinstance(ref, (list, tuple, np.ndarray, pd.Series)):
        return ' '.join(str(token) for token in ref)
    return str(ref)


def normalise_ref(r):
    """
    Safely converts a reference caption into a plain string, regardless of
    whether it arrives as a string, list of tokens, or dict.
    """
    if isinstance(r, str):
        return r
    if isinstance(r, list):
        return ' '.join(str(t) for t in r)
    if isinstance(r, dict):
        for key in ('caption', 'text', 'sentence'):
            if key in r:
                return str(r[key])
        return ' '.join(str(v) for v in r.values())
    return str(r)


def _split_ids(payload, split_name):
    """Returns integer image identifiers for a named split from splits.json."""
    if split_name not in payload:
        raise KeyError(f"Missing '{split_name}' in splits.json")
    return [int(image_id) for image_id in payload[split_name]]


def load_vocabulary(artifacts_dir):
    """
    Loads word2idx.json and validates the expected special tokens.

    Returns:
        dict with keys: word2idx, idx2word, pad_idx, sos_idx, eos_idx, unk_idx
    """
    artifacts_dir = Path(artifacts_dir)
    with open(artifacts_dir / 'word2idx.json', 'r') as f:
        word2idx = json.load(f)

    expected_specials = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    if {k: word2idx.get(k) for k in expected_specials} != expected_specials:
        raise ValueError('word2idx.json special token indices do not match Phase 1')

    return {
        'word2idx': word2idx,
        'idx2word': {idx: word for word, idx in word2idx.items()},
        'pad_idx': word2idx['<pad>'],
        'sos_idx': word2idx['<sos>'],
        'eos_idx': word2idx['<eos>'],
        'unk_idx': word2idx['<unk>'],
    }


def load_captions_dataframe(artifacts_dir):
    """Loads and lightly validates captions_clean.parquet."""
    artifacts_dir = Path(artifacts_dir)
    df_captions = pd.read_parquet(artifacts_dir / 'captions_clean.parquet').copy()

    required_columns = {
        'caption_id', 'image_id', 'file_name', 'caption', 'tokens', 'n_tokens',
        'is_rejected', 'is_precanned', 'text_detected', 'split',
    }
    missing_columns = required_columns - set(df_captions.columns)
    if missing_columns:
        raise KeyError(f"captions_clean.parquet missing columns: {sorted(missing_columns)}")

    df_captions['image_id'] = df_captions['image_id'].astype(int)
    df_captions['file_name'] = df_captions['file_name'].apply(lambda x: Path(str(x)).name)
    df_captions['image'] = df_captions['file_name']
    return df_captions


def load_splits_and_references(artifacts_dir, images_dir, df_captions):
    """
    Cross-validates splits.json / references_by_image.json against the
    captions dataframe and the images actually present on disk, then builds
    the per-split file lists and scorer-ready reference dictionaries.

    Returns:
        dict with keys: splits, references, references_clean, df_captions
        (df_captions is filtered down to images that survived the disk check)
    """
    artifacts_dir = Path(artifacts_dir)
    images_dir = Path(images_dir)

    with open(artifacts_dir / 'splits.json', 'r') as f:
        splits_raw = json.load(f)
    with open(artifacts_dir / 'references_by_image.json', 'r') as f:
        references_raw = json.load(f)

    disk_files = {p.name for p in images_dir.glob('*.jpg')}
    id_to_file = (
        df_captions[['image_id', 'image']]
        .drop_duplicates('image_id')
        .set_index('image_id')['image']
        .to_dict()
    )

    splits_by_image_id = {s: _split_ids(splits_raw, s) for s in ['train', 'val', 'test']}
    splits = {}
    available_split_ids = set()

    for split_name, image_ids in splits_by_image_id.items():
        missing_from_parquet = [i for i in image_ids if i not in id_to_file]
        if missing_from_parquet:
            raise ValueError(
                f"{split_name} has {len(missing_from_parquet)} ids absent from captions_clean.parquet"
            )

        missing_on_disk = [id_to_file[i] for i in image_ids if id_to_file[i] not in disk_files]
        if missing_on_disk:
            print(f"WARNING: {split_name} missing {len(missing_on_disk)} image files on disk")

        split_files = [id_to_file[i] for i in image_ids if id_to_file[i] in disk_files]
        splits[split_name] = split_files
        available_split_ids.update(i for i in image_ids if id_to_file[i] in disk_files)

    df_captions = df_captions[df_captions['image_id'].isin(available_split_ids)].reset_index(drop=True)

    for split_name, image_ids in splits_by_image_id.items():
        expected_ids = {i for i in image_ids if id_to_file[i] in disk_files}
        parquet_ids = set(df_captions.loc[df_captions['split'] == split_name, 'image_id'])
        if expected_ids != parquet_ids:
            raise ValueError(f"Split mismatch between splits.json and captions_clean.parquet for {split_name}")

    # Scorer-ready format: {file_name: [caption1, caption2, ...]}
    references = {}
    for split_name, image_ids in splits_by_image_id.items():
        raw_split_refs = references_raw.get(split_name, {})
        references[split_name] = {}
        for image_id in image_ids:
            file_name = id_to_file[image_id]
            if file_name not in disk_files:
                continue
            if str(image_id) not in raw_split_refs:
                raise ValueError(f"Missing references for {split_name} image_id={image_id}")
            refs = raw_split_refs.get(str(image_id), [])
            references[split_name][file_name] = [reference_to_string(r) for r in refs]

    references_clean = {
        split_name: {
            img_id: [normalise_ref(r) for r in refs]
            for img_id, refs in split_refs.items()
        }
        for split_name, split_refs in references.items()
    }

    return {
        'splits': splits,
        'references': references,
        'references_clean': references_clean,
        'df_captions': df_captions,
    }


def load_dataset_artifacts(config, images_dir):
    """
    Convenience wrapper: loads vocabulary, captions dataframe, and
    splits/references in one call. This is the main entry point most
    training/notebook code should use.

    Returns a single dict merging the outputs of load_vocabulary(),
    load_captions_dataframe(), and load_splits_and_references().
    """
    artifacts_dir = config['paths']['artifacts_dir']

    vocab = load_vocabulary(artifacts_dir)
    df_captions = load_captions_dataframe(artifacts_dir)
    split_data = load_splits_and_references(artifacts_dir, images_dir, df_captions)

    print(f"Vocabulary entries : {len(vocab['word2idx']):,}")
    print(
        f"Special token ids  : pad={vocab['pad_idx']}, sos={vocab['sos_idx']}, "
        f"eos={vocab['eos_idx']}, unk={vocab['unk_idx']}"
    )
    print(f"Captions available : {len(split_data['df_captions']):,}")

    for split_name in ('train', 'val', 'test'):
        n_caps = int((split_data['df_captions']['split'] == split_name).sum())
        n_images = len(split_data['splits'][split_name])
        n_ref_groups = len(split_data['references'][split_name])
        print(f"{split_name:5s}: {n_images:>5} images, {n_caps:>6} captions, {n_ref_groups:>5} reference groups")

    return {**vocab, **split_data}


# --------------------------------------------------------------------------
# Tokenizer & vision processor
# --------------------------------------------------------------------------

def get_gpt2_tokenizer(model_name='gpt2'):
    """
    Loads the GPT-2 BPE tokenizer and applies the standard Hugging Face
    workaround for batching: GPT-2 has no native padding token, so the
    padding token is assigned to EOS.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Vocab size    : {tokenizer.vocab_size}")
    print(f"Pad token ID  : {tokenizer.pad_token_id}  (shared with EOS)")
    return tokenizer


def get_clip_processor(model_name='openai/clip-vit-base-patch32'):
    """
    Loads the pre-trained CLIP image processor. Handles resizing to 224x224
    and normalizes pixel values using CLIP's dataset statistics
    (mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758]).
    """
    return CLIPProcessor.from_pretrained(model_name)