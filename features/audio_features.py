import librosa
import numpy as np

def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # MFCCs (13)
    mfcc = np.mean(
        librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T,
        axis=0
    )

    # Other speech biomarkers
    jitter = np.std(librosa.feature.zero_crossing_rate(y))
    shimmer = np.std(y)
    pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))
    rms = np.mean(librosa.feature.rms(y=y))

    return np.hstack([
        mfcc,       # 13
        jitter,     # 1
        shimmer,    # 1
        pitch,      # 1
        rms         # 1
    ])
