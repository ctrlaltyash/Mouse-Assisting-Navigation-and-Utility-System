#!/usr/bin/env python3
"""
Download MediaPipe hand landmarker models.

This script downloads the required hand landmarker model files for MediaPipe Tasks API.
Run this once after installation to cache the models locally.

Usage:
    python download_models.py [--lite] [--force]

Options:
    --lite   Download lite model (smaller, faster, less accurate)
    --force  Re-download even if models already exist
"""

import os
import sys
import argparse
import urllib.request
import json
from pathlib import Path

def get_cache_dir():
    """Get the model cache directory."""
    cache_dir = Path.home() / '.cache' / 'mediapipe'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def download_model(model_name, force=False):
    """Download a model file."""
    cache_dir = get_cache_dir()
    model_path = cache_dir / f"{model_name}.task"
    
    if model_path.exists() and not force:
        print(f"✓ Model already cached: {model_path}")
        return str(model_path)
    
    # Try multiple URL formats
    urls = [
        f"https://storage.googleapis.com/mediapipe-tasks/{model_name}/{model_name}.task",
        f"https://storage.googleapis.com/mediapipe-models/{model_name}/{model_name}.task",
        f"https://storage.googleapis.com/mediapipe-assets/{model_name}.task",
        f"https://storage.googleapis.com/mediapipe-tasks/{model_name}.task",
    ]
    
    # Try GitHub releases as fallback
    github_urls = [
        f"https://github.com/google/mediapipe/releases/download/v0.10.33/{model_name}.task",
        f"https://github.com/google-ai-edge/mediapipe/releases/download/v0.10.33/{model_name}.task",
    ]
    
    all_urls = urls + github_urls
    
    print(f"Downloading {model_name}...")
    for url in all_urls:
        try:
            print(f"  Trying: {url}")
            urllib.request.urlretrieve(url, model_path, reporthook=_download_progress)
            print(f"\n✓ Successfully downloaded: {model_path}")
            return str(model_path)
        except urllib.error.HTTPError as e:
            print(f"  ✗ HTTP {e.code}: Not found at this URL")
            continue
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    print(f"\n✗ Failed to download {model_name}")
    print("\nManual download instructions:")
    print("  1. Visit: https://ai.google.dev/mediapipe/solutions/vision/hand_landmarker")
    print("  2. Download the model files")
    print(f"  3. Place them in: {cache_dir}")
    
    return None

def _download_progress(block_num, block_size, total_size):
    """Display download progress."""
    downloaded = block_num * block_size
    percent = min(downloaded * 100 // total_size, 100) if total_size > 0 else 0
    sys.stdout.write(f"\r  Progress: {percent}%")
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="Download MediaPipe hand landmarker models")
    parser.add_argument('--lite', action='store_true', help='Download lite model only')
    parser.add_argument('--full', action='store_true', help='Download full model only')
    parser.add_argument('--both', action='store_true', help='Download both models (default)')
    parser.add_argument('--force', action='store_true', help='Re-download even if cached')
    
    args = parser.parse_args()
    
    # Determine which models to download
    models = []
    # Use single MediaPipe task model file available in current hosted repository
    # This is the new canonical model distribution from MediaPipe 0.10+
    if args.lite or args.full or args.both or (not args.lite and not args.full and not args.both):
        models = ['hand_landmarker']
    else:
        models = ['hand_landmarker']
    
    print("=" * 60)
    print("MediaPipe Hand Landmarker Model Downloader")
    print("=" * 60)
    
    results = {}
    for model_name in models:
        path = download_model(model_name, force=args.force)
        results[model_name] = path
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for model_name, path in results.items():
        if path:
            print(f"✓ {model_name}: {path}")
        else:
            print(f"✗ {model_name}: Failed to download")
    
    return 0 if all(results.values()) else 1

if __name__ == '__main__':
    sys.exit(main())
