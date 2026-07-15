"""
Dataset classes and feature-caching utilities for the VizWiz image
captioning pipeline. Shared by both Architecture 1 (ClipCap, global CLIP
embeddings) and Architecture 2 (SR-ClipCap, SigLIP spatial features).

This module is import-safe: importing it has no side effects. All datasets
are constructed explicitly by the caller (notebook or training script) with
already-loaded dataframes, splits, and tokenizers — see
`src/data/preprocessing.py` for how to obtain those.
"""

import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# --------------------------------------------------------------------------
# Feature tensor caching
# --------------------------------------------------------------------------

class FeatureTensorLRUCache:
    """
    Small explicit LRU cache for `torch.load`-ed feature tensors, so cached
    spatial/CLIP feature files aren't re-read from disk on every epoch.
    """

    def __init__(self, max_items=2048):
        self.max_items = int(max_items)
        self._cache = OrderedDict()

    def get(self, path, weights_only=True):
        path = Path(path)
        key = (str(path.resolve()), bool(weights_only))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        tensor = torch.load(path, weights_only=weights_only)
        self._cache[key] = tensor
        self._cache.move_to_end(key)

        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)

        return tensor

    def clear(self):
        self._cache.clear()

    def __len__(self):
        return len(self._cache)


# Module-level shared cache instance, used by VizWizCaptionDataset in
# 'spatial' feature mode. Safe to share across dataset instances/splits.
FEATURE_TENSOR_CACHE = FeatureTensorLRUCache()


def load_pt_feature_cached(path, weights_only=True, expected_shape=None):
    """Loads a .pt feature tensor through the shared LRU cache, with an optional shape check."""
    tensor = FEATURE_TENSOR_CACHE.get(path, weights_only=weights_only)
    if expected_shape is not None and tuple(tensor.shape) != tuple(expected_shape):
        raise ValueError(f"Expected feature shape {expected_shape} for {path}, got {tuple(tensor.shape)}")
    return tensor


# --------------------------------------------------------------------------
# Core caption dataset (shared base for both architectures)
# --------------------------------------------------------------------------

class VizWizCaptionDataset(Dataset):
    """
    Unified caption dataset supporting two feature modes:
      - 'clip'    : in-memory global CLIP embeddings (Architecture 1)
      - 'spatial' : cached per-image SigLIP spatial feature tensors on disk (Architecture 2)

    Tokenizes captions with an EOS-wrapped, fixed-length, padded scheme
    suitable for GPT-2 style autoregressive training.
    """

    def __init__(self, df, split_dict, split_name, tokenizer, *, feature_mode,
                 clip_features=None, feature_dir=None, max_length=22,
                 expected_spatial_shape=(196, 768)):
        """
        Args:
            df: Captions dataframe (must have 'image', 'tokens'/'caption' columns).
            split_dict: Mapping from split name -> list of image filenames.
            split_name: Which split ('train' / 'val' / 'test') this instance exposes.
            tokenizer: GPT-2 tokenizer (or compatible) for caption encoding.
            feature_mode: 'clip' or 'spatial'.
            clip_features: dict {image_name: tensor}, required if feature_mode='clip'.
            feature_dir: directory of cached `{image_name}.pt` files, required if feature_mode='spatial'.
            max_length: fixed sequence length captions are padded/truncated to.
            expected_spatial_shape: sanity-checked shape for cached spatial tensors.
        """
        if feature_mode not in {'clip', 'spatial'}:
            raise ValueError(f"Unsupported feature_mode={feature_mode!r}")

        img_names = {Path(str(x)).name for x in split_dict[split_name]}
        filtered_df = df[df['image'].isin(img_names)].reset_index(drop=True)

        self.data = filtered_df.to_dict('records')
        self.tokenizer = tokenizer
        self.feature_mode = feature_mode
        self.clip_features = clip_features
        self.feature_dir = Path(feature_dir) if feature_dir is not None else None
        self.max_length = max_length
        self.expected_spatial_shape = expected_spatial_shape

    def __len__(self):
        return len(self.data)

    def _caption_text(self, row):
        tokens = row.get('tokens')
        if isinstance(tokens, (list, tuple, np.ndarray, pd.Series)):
            return ' '.join(str(token) for token in tokens)
        return str(row.get('caption', ''))

    def _feature(self, img_name):
        if self.feature_mode == 'clip':
            return self.clip_features[img_name]

        feature_path = self.feature_dir / f"{img_name}.pt"
        return load_pt_feature_cached(
            feature_path,
            weights_only=True,
            expected_shape=self.expected_spatial_shape,
        )

    def __getitem__(self, idx):
        row = self.data[idx]
        img_name = Path(str(row['image'])).name
        caption = self._caption_text(row)

        caption_ids = self.tokenizer.encode(caption, add_special_tokens=False)
        caption_ids = caption_ids[: self.max_length - 2]

        ids = [self.tokenizer.eos_token_id] + caption_ids + [self.tokenizer.eos_token_id]
        attention = [1] * len(ids)

        pad_len = self.max_length - len(ids)
        ids = ids + [self.tokenizer.pad_token_id] * pad_len
        attention = attention + [0] * pad_len

        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.tensor(attention, dtype=torch.long)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        labels[0] = -100

        item = {
            'image_id': img_name,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

        if self.feature_mode == 'clip':
            item['clip_embed'] = self._feature(img_name)
        else:
            item['spatial_tokens'] = self._feature(img_name)

        return item


class VizWizCachedDataset(VizWizCaptionDataset):
    """Architecture 1 (ClipCap) dataset: fixed-length captions + in-memory CLIP embeddings."""

    def __init__(self, df, split_dict, split_name, clip_features, tokenizer, max_length=22):
        super().__init__(
            df, split_dict, split_name, tokenizer,
            feature_mode='clip', clip_features=clip_features, max_length=max_length,
        )


class SpatialCachedDataset(VizWizCaptionDataset):
    """Architecture 2 (SR-ClipCap) dataset: fixed-length captions + cached SigLIP spatial features."""

    def __init__(self, df, split_dict, split_name, tokenizer, feature_dir=None, max_length=22):
        super().__init__(
            df, split_dict, split_name, tokenizer,
            feature_mode='spatial', feature_dir=feature_dir,
            max_length=max_length, expected_spatial_shape=(196, 768),
        )


# --------------------------------------------------------------------------
# Raw-image datasets used for one-time feature extraction
# --------------------------------------------------------------------------

class VizWizRawDataset(Dataset):
    """
    Loads raw images from disk and returns CLIPProcessor pixel tensors.
    Used exclusively for the one-time CLIP feature extraction pass
    (Architecture 1). Each image appears exactly once regardless of caption count.
    """

    def __init__(self, split_dict, split_name, images_dir, processor):
        """
        Args:
            split_dict: Mapping from split names to image filenames.
            split_name: Split to expose through this dataset.
            images_dir: Directory containing VizWiz image files.
            processor: CLIPProcessor used to prepare pixel tensors.
        """
        self.image_names = split_dict[split_name]
        self.images_dir = Path(images_dir)
        self.processor = processor

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = Path(self.image_names[idx]).name

        # Load image and ensure 3-channel RGB (handles greyscale/RGBA edge cases)
        image = Image.open(self.images_dir / img_name).convert("RGB")

        # Process image and remove the batch dimension added by return_tensors="pt"
        pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

        return {"image_id": img_name, "pixel_values": pixel_values}


class VizWizSpatialFeatureDataset(Dataset):
    """
    Returns raw image paths and target cache paths for SigLIP spatial
    feature extraction (Architecture 2). Used only during the one-time
    extraction pass, not during training.
    """

    def __init__(self, image_files, images_dir, feature_dir):
        self.image_files = list(image_files)
        self.images_dir = Path(images_dir)
        self.feature_dir = Path(feature_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        image_path = self.images_dir / file_name
        feature_path = self.feature_dir / f"{file_name}.pt"

        if not image_path.exists():
            raise FileNotFoundError(f"Missing source image: {image_path}")

        return {
            "file_name": file_name,
            "image_path": str(image_path),
            "feature_path": str(feature_path),
        }


def spatial_feature_collate(batch):
    """
    Custom collate function for VizWizSpatialFeatureDataset — keeps paths as
    lists rather than trying to stack/tensorize them, since image loading
    happens later inside the extraction loop.
    """
    return {
        "file_name": [item["file_name"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "feature_path": [item["feature_path"] for item in batch],
    }


# --------------------------------------------------------------------------
# Data manifest validation (pre-flight checks before training)
# --------------------------------------------------------------------------

def file_quick_hash(path, chunk_bytes=None):
    """Returns a stable metadata-based SHA256 signature for a file, or None if it doesn't exist."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None

    stat = path.stat()
    signature = f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode('utf-8')
    return hashlib.sha256(signature).hexdigest()


def build_data_manifest(df, splits, images_dir, *, clip_feature_paths=None, spatial_feature_dir=None):
    """
    Builds a pre-run manifest covering image, split, CLIP cache, and spatial
    feature state — used to fail fast (via validate_data_manifest) before a
    long training run starts, rather than crashing partway through an epoch.
    """
    rows = []
    id_lookup = (
        df[['image_id', 'image', 'split']]
        .drop_duplicates(['image_id', 'image', 'split'])
        .set_index('image')
        .to_dict('index')
    )

    clip_feature_paths = clip_feature_paths or {}
    spatial_feature_dir = Path(spatial_feature_dir) if spatial_feature_dir is not None else None

    for split_name in ['train', 'val', 'test']:
        clip_cache_path = clip_feature_paths.get(split_name)
        clip_cache_path = Path(clip_cache_path) if clip_cache_path is not None else None
        clip_cache_exists = clip_cache_path.exists() if clip_cache_path is not None else False
        clip_cache_hash = file_quick_hash(clip_cache_path) if clip_cache_exists else None

        for file_name in splits[split_name]:
            img_name = Path(str(file_name)).name
            image_path = Path(images_dir) / img_name
            meta = id_lookup.get(img_name, {})
            spatial_path = spatial_feature_dir / f"{img_name}.pt" if spatial_feature_dir is not None else None
            spatial_exists = spatial_path.exists() if spatial_path is not None else False

            rows.append({
                'image_id': meta.get('image_id'),
                'file_name': img_name,
                'split': split_name,
                'image_path': str(image_path),
                'image_exists': image_path.exists(),
                'clip_feature_cache_path': str(clip_cache_path) if clip_cache_path is not None else None,
                'clip_feature_cache_exists': clip_cache_exists,
                'clip_feature_cache_hash': clip_cache_hash,
                'spatial_feature_path': str(spatial_path) if spatial_path is not None else None,
                'spatial_feature_exists': spatial_exists,
                'spatial_feature_hash': file_quick_hash(spatial_path) if spatial_exists else None,
            })

    return pd.DataFrame(rows)


def validate_data_manifest(manifest, *, require_images=True, require_clip_cache=False, require_spatial=False):
    """Raises FileNotFoundError with example rows if any required artifact is missing."""
    checks = []
    if require_images:
        checks.append(('image_exists', 'source images'))
    if require_clip_cache:
        checks.append(('clip_feature_cache_exists', 'CLIP feature cache'))
    if require_spatial:
        checks.append(('spatial_feature_exists', 'SigLIP spatial features'))

    for column, label in checks:
        missing = manifest.loc[~manifest[column].fillna(False)]
        if len(missing):
            examples = missing[['split', 'file_name']].head(5).to_dict('records')
            raise FileNotFoundError(f"Missing {label}: {len(missing)} rows. Examples: {examples}")

    print(
        'Manifest validated: '
        f"{len(manifest):,} image rows | "
        f"images={manifest['image_exists'].sum():,} | "
        f"clip_cache_rows={manifest['clip_feature_cache_exists'].sum():,} | "
        f"spatial_rows={manifest['spatial_feature_exists'].sum():,}"
    )
    return manifest