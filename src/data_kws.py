"""Google Speech Commands v0.02 Dataset ingestion, extraction, and preprocessing module."""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import tarfile
import urllib.request
import glob
import random
import yaml
import numpy as np
import soundfile as sf
from src.features import compute_mfcc

def get_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def download_and_extract_speech_commands(data_dir="data"):
    target_dir = os.path.join(data_dir, "speech_commands")
    marker = os.path.join(target_dir, "validation_list.txt")
    if os.path.exists(marker):
        return target_dir
        
    os.makedirs(target_dir, exist_ok=True)
    tar_path = os.path.join(target_dir, "speech_commands_v0.02.tar.gz")
    url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    
    if not os.path.exists(tar_path):
        print(f"Downloading Google Speech Commands v0.02 from {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp, open(tar_path, 'wb') as out_f:
            total_size = int(resp.headers.get('content-length', 0))
            downloaded = 0
            block_size = 1024 * 1024 * 4 # 4MB blocks
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded * 100.0 / total_size
                    print(f"\rDownloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)", end="", flush=True)
            print()
            
    print(f"Extracting selective classes from {tar_path}...")
    # Extract only needed target words, background noise, lists, and a subset of unknown words
    needed_words = {"yes", "no", "stop", "go", "up", "down", "_background_noise_"}
    unknown_candidates = {"bed", "bird", "cat", "dog", "eight", "five", "four", "happy", "house", "left", "marvin", "nine", "off", "on", "one", "right", "seven", "sheila", "six", "tree", "two", "wow", "zero"}
    
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            name = member.name
            if name.endswith(".txt") or name.endswith(".md"):
                tar.extract(member, target_dir)
                continue
            parts = name.split("/")
            if len(parts) >= 2:
                word = parts[0] if parts[0] != "." else parts[1]
                if word in needed_words or word in unknown_candidates:
                    tar.extract(member, target_dir)
                    
    print("Selective extraction complete.")
    return target_dir

def load_audio_clip(wav_path, target_sr=16000):
    """Loads 1s audio clip, returns (16000,) float32."""
    data, sr = sf.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]
    if len(data) < target_sr:
        data = np.pad(data, (0, target_sr - len(data)), mode='constant')
    elif len(data) > target_sr:
        data = data[:target_sr]
    return data.astype(np.float32)

def generate_silence_clips(bg_dir, num_clips=1000, target_sr=16000):
    """Generates random 1s slices from background noise files."""
    bg_files = glob.glob(os.path.join(bg_dir, "*.wav"))
    if not bg_files:
        # Fallback to pure zeros / low white noise
        return [np.random.normal(0, 1e-4, target_sr).astype(np.float32) for _ in range(num_clips)]
        
    bg_audios = []
    for f in bg_files:
        try:
            d, _ = sf.read(f)
            if d.ndim > 1:
                d = d[:, 0]
            bg_audios.append(d)
        except Exception:
            pass
            
    clips = []
    for _ in range(num_clips):
        audio = random.choice(bg_audios)
        if len(audio) <= target_sr:
            clip = np.pad(audio, (0, target_sr - len(audio)), mode='constant')
        else:
            idx = random.randint(0, len(audio) - target_sr)
            clip = audio[idx:idx + target_sr]
        # Scale background noise to quiet volume
        scale = random.uniform(0.1, 0.4)
        clips.append((clip * scale).astype(np.float32))
    return clips

def load_kws_splits(data_dir="data", quick=False):
    config = get_config()
    cache_name = "kws_features_quick.npz" if quick else "kws_features.npz"
    cache_path = os.path.join(data_dir, cache_name)
    
    if os.path.exists(cache_path):
        print(f"Loading precomputed KWS features from {cache_path}...")
        cached = np.load(cache_path)
        return {
            "train": (cached["X_train"], cached["y_train"]),
            "val": (cached["X_val"], cached["y_val"]),
            "test": (cached["X_test"], cached["y_test"]),
            "classes": config["kws"]["all_classes"]
        }
        
    dataset_dir = download_and_extract_speech_commands(data_dir)
    val_list_file = os.path.join(dataset_dir, "validation_list.txt")
    test_list_file = os.path.join(dataset_dir, "testing_list.txt")
    
    with open(val_list_file, "r") as f:
        val_paths = set(line.strip().replace("/", os.sep) for line in f if line.strip())
    with open(test_list_file, "r") as f:
        test_paths = set(line.strip().replace("/", os.sep) for line in f if line.strip())
        
    target_words = config["kws"]["target_words"]
    all_classes = config["kws"]["all_classes"]
    word_to_idx = {w: i for i, w in enumerate(target_words)}
    word_to_idx["unknown"] = 6
    word_to_idx["silence"] = 7
    
    cap_train = config["kws"]["quick_sample_cap_train"] if quick else config["kws"]["sample_cap_train"]
    cap_val = int(cap_train * 0.15)
    cap_test = int(cap_train * 0.15)
    
    splits = {"train": [], "val": [], "test": []}
    
    # 1. Target words
    for word in target_words:
        word_dir = os.path.join(dataset_dir, word)
        wav_files = glob.glob(os.path.join(word_dir, "*.wav"))
        
        train_wavs, val_wavs, test_wavs = [], [], []
        for wav in wav_files:
            rel = os.path.relpath(wav, dataset_dir)
            if rel in val_paths:
                val_wavs.append(wav)
            elif rel in test_paths:
                test_wavs.append(wav)
            else:
                train_wavs.append(wav)
                
        random.seed(42)
        random.shuffle(train_wavs)
        random.shuffle(val_wavs)
        random.shuffle(test_wavs)
        
        for w in train_wavs[:cap_train]:
            splits["train"].append((w, word_to_idx[word]))
        for w in val_wavs[:cap_val]:
            splits["val"].append((w, word_to_idx[word]))
        for w in test_wavs[:cap_test]:
            splits["test"].append((w, word_to_idx[word]))
            
    # 2. Unknown class (sample from other words)
    unknown_candidates = ["bed", "bird", "cat", "dog", "eight", "five", "four", "happy", "house", "left", "marvin", "nine", "off", "on", "one", "right", "seven", "sheila", "six", "tree", "two", "wow", "zero"]
    unknown_train, unknown_val, unknown_test = [], [], []
    for uw in unknown_candidates:
        u_dir = os.path.join(dataset_dir, uw)
        if not os.path.isdir(u_dir):
            continue
        for wav in glob.glob(os.path.join(u_dir, "*.wav")):
            rel = os.path.relpath(wav, dataset_dir)
            if rel in val_paths:
                unknown_val.append(wav)
            elif rel in test_paths:
                unknown_test.append(wav)
            else:
                unknown_train.append(wav)
                
    random.seed(42)
    random.shuffle(unknown_train)
    random.shuffle(unknown_val)
    random.shuffle(unknown_test)
    
    for w in unknown_train[:cap_train]:
        splits["train"].append((w, word_to_idx["unknown"]))
    for w in unknown_val[:cap_val]:
        splits["val"].append((w, word_to_idx["unknown"]))
    for w in unknown_test[:cap_test]:
        splits["test"].append((w, word_to_idx["unknown"]))
        
    # 3. Silence class
    bg_dir = os.path.join(dataset_dir, "_background_noise_")
    total_silence_train = cap_train
    total_silence_val = cap_val
    total_silence_test = cap_test
    silence_clips_train = generate_silence_clips(bg_dir, num_clips=total_silence_train)
    silence_clips_val = generate_silence_clips(bg_dir, num_clips=total_silence_val)
    silence_clips_test = generate_silence_clips(bg_dir, num_clips=total_silence_test)
    
    # Process features
    print("Extracting MFCCs for all splits...")
    res = {}
    for split_name in ["train", "val", "test"]:
        items = splits[split_name]
        random.seed(42)
        random.shuffle(items)
        
        mfccs = []
        labels = []
        for wav_path, label in items:
            audio = load_audio_clip(wav_path)
            # Training augmentation: time shift
            if split_name == "train" and random.random() < 0.5:
                shift = random.randint(-1600, 1600)
                audio = np.roll(audio, shift)
                if shift > 0:
                    audio[:shift] = 0
                elif shift < 0:
                    audio[shift:] = 0
            feat = compute_mfcc(audio)
            mfccs.append(feat)
            labels.append(label)
            
        # Add silence clips
        silence_clips = {"train": silence_clips_train, "val": silence_clips_val, "test": silence_clips_test}[split_name]
        for clip in silence_clips:
            feat = compute_mfcc(clip)
            mfccs.append(feat)
            labels.append(word_to_idx["silence"])
            
        X = np.stack(mfccs, axis=0) # (N, 98, 40)
        y = np.array(labels, dtype=int)
        res[split_name] = (X, y)
        print(f"Split {split_name}: {X.shape}, labels distribution: {np.bincount(y, minlength=8)}")
        
    np.savez_compressed(
        cache_path,
        X_train=res["train"][0], y_train=res["train"][1],
        X_val=res["val"][0], y_val=res["val"][1],
        X_test=res["test"][0], y_test=res["test"][1]
    )
    print(f"Saved precomputed features to {cache_path}")
    
    return {
        "train": res["train"],
        "val": res["val"],
        "test": res["test"],
        "classes": all_classes
    }

if __name__ == "__main__":
    import sys
    quick_mode = "--quick" in sys.argv
    splits = load_kws_splits(quick=quick_mode)
    print("KWS Train shape:", splits["train"][0].shape)
    print("KWS Test shape:", splits["test"][0].shape)
