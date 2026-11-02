"""Feature extraction for IMU and Audio.
Provides deterministic 50 Hz IMU normalization and 16 kHz MFCC calculation.
"""

import numpy as np
from scipy.fft import dct

def normalize_imu_window(imu_data, phone_mode=False):
    """
    Normalizes IMU window data of shape (N, 6) or (batch, N, 6).
    Channels: [total_acc_x, total_acc_y, total_acc_z, body_gyro_x, body_gyro_y, body_gyro_z]
    
    UCI HAR units:
      acc: g (1 g ~ 9.81 m/s^2)
      gyro: rad/s
    
    If phone_mode=True, incoming acceleration is in m/s^2, so divide by 9.81.
    Incoming gyro is assumed already in rad/s.
    """
    data = np.array(imu_data, dtype=np.float32).copy()
    if phone_mode:
        data[..., :3] = data[..., :3] / 9.80665
    return data

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def get_mel_filterbank(sr=16000, n_fft=512, n_mels=40, f_min=20.0, f_max=4000.0):
    """Creates a triangular mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)."""
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max if f_max is not None else sr / 2)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
                
    return filterbank

# Cache filterbank
_MEL_FILTERBANK = get_mel_filterbank()

def compute_mfcc(audio_16k, sr=16000, n_mfcc=40, n_mels=40, frame_len_ms=25.0, frame_step_ms=10.0, n_fft=512):
    """
    Computes MFCC features for 1-second audio at 16 kHz.
    Returns array of shape (98, n_mfcc).
    """
    # Ensure length is at least 16000
    target_len = sr
    if len(audio_16k) < target_len:
        audio = np.pad(audio_16k, (0, target_len - len(audio_16k)), mode='constant')
    else:
        audio = audio_16k[:target_len]
        
    audio = audio.astype(np.float32)
    # Normalize if integer PCM
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / 32768.0
        
    # Pre-emphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    
    frame_len = int(round(sr * frame_len_ms / 1000.0))  # 400
    frame_step = int(round(sr * frame_step_ms / 1000.0)) # 160
    
    num_frames = 1 + int(np.floor((len(audio) - frame_len) / frame_step))
    pad_signal_len = (num_frames - 1) * frame_step + frame_len
    if len(audio) < pad_signal_len:
        audio = np.pad(audio, (0, pad_signal_len - len(audio)), mode='constant')
        
    indices = (
        np.tile(np.arange(0, frame_len), (num_frames, 1)) +
        np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
    )
    frames = audio[indices]
    
    # Hamming window
    window = np.hamming(frame_len)
    frames = frames * window
    
    # Magnitude Spectrogram
    mag_frames = np.abs(np.fft.rfft(frames, n_fft))
    pow_frames = (1.0 / n_fft) * (mag_frames ** 2)
    
    # Mel Filterbank
    mel_energies = np.dot(pow_frames, _MEL_FILTERBANK.T)
    mel_energies = np.where(mel_energies == 0, np.finfo(float).eps, mel_energies)
    log_mel = np.log(mel_energies)
    
    # DCT type 2, ortho norm
    mfcc = dct(log_mel, type=2, axis=-1, norm='ortho')[:, :n_mfcc]
    
    # Normalize per utterance (zero mean, unit variance)
    mean = np.mean(mfcc, axis=0, keepdims=True)
    std = np.std(mfcc, axis=0, keepdims=True) + 1e-6
    norm_mfcc = (mfcc - mean) / std
    
    return norm_mfcc.astype(np.float32)

if __name__ == "__main__":
    dummy_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
    mfcc_out = compute_mfcc(dummy_audio)
    print("MFCC Output Shape:", mfcc_out.shape, "dtype:", mfcc_out.dtype)
    assert mfcc_out.shape == (98, 40), f"Expected (98, 40), got {mfcc_out.shape}"
    print("Feature extraction test passed.")
