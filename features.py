"""Audio utilities and feature extraction for the EOT assignment.

Causality reminder: for a pause at `pause_start`, you may only touch
audio[0 : pause_start]. Note that `pause_end` is FUTURE information for a
hold pause — using it (e.g., pause duration) in features is a violation.
"""
import numpy as np
import soundfile as sf

FRAME_MS = 25
HOP_MS = 10


def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def speech_before(x, sr, pause_start, window_s=1.5):
    """The last `window_s` seconds of audio strictly before the pause."""
    end = int(pause_start * sr)
    start = max(0, end - int(window_s * sr))
    return x[start:end]


def frames(x, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    fl = int(sr * frame_ms / 1000)
    hp = int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.empty((0, fl), dtype=np.float32)
    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return x[idx]


def frame_energy_db(x, sr):
    """Short-time energy per frame, in dB."""
    fr = frames(x, sr)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def autocorr_f0(frame, sr, fmin=60.0, fmax=400.0, voicing_thresh=0.30):
    """Fundamental frequency of one frame via autocorrelation.

    Returns 0.0 for unvoiced/silent frames.
    """
    frame = frame - np.mean(frame)
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo = int(sr / fmax)
    hi = min(int(sr / fmin), len(ac) - 1)
    if hi <= lo:
        return 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < voicing_thresh:
        return 0.0
    return float(sr / lag)


def f0_contour(x, sr, frame_ms=40, hop_ms=HOP_MS):
    """Per-frame F0 (Hz), 0.0 where unvoiced. Longer frames help pitch."""
    fr = frames(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    return np.array([autocorr_f0(f, sr) for f in fr], dtype=np.float32)


# --- New DSP Features ---

def zero_crossing_rate(frame):
    """Zero Crossing Rate of a frame."""
    diff_sign = np.diff(np.sign(frame))
    return float(np.mean(np.abs(diff_sign) > 0))


def spectral_centroid_and_rolloff_and_ratio(frame, sr):
    """Computes spectral centroid, spectral roll-off (85%), and high/low ratio.
    
    Done in a single function to save redundant FFT computations.
    """
    N = len(frame)
    # Apply Hamming window
    w_frame = frame * np.hamming(N)
    spec = np.abs(np.fft.rfft(w_frame))
    freqs = np.fft.rfftfreq(N, 1/sr)
    
    sum_spec = np.sum(spec)
    if sum_spec < 1e-12:
        return 0.0, 0.0, 0.0
    
    # 1. Spectral Centroid
    centroid = float(np.sum(freqs * spec) / sum_spec)
    
    # 2. Spectral Roll-off (85%)
    cum_spec = np.cumsum(spec)
    threshold = 0.85 * sum_spec
    idx = np.where(cum_spec >= threshold)[0]
    rolloff = float(freqs[idx[0]]) if len(idx) > 0 else 0.0
    
    # 3. High/Low frequency ratio (split at 1500 Hz)
    low_mask = freqs <= 1500.0
    high_mask = freqs > 1500.0
    sum_low = np.sum(spec[low_mask])
    sum_high = np.sum(spec[high_mask])
    ratio = float(sum_high / (sum_low + 1e-12))
    
    return centroid, rolloff, ratio


def get_slope(y):
    """Fits a linear regression line to y and returns the slope."""
    N = len(y)
    if N < 2:
        return 0.0
    x = np.arange(N)
    cov = np.cov(x, y)
    if cov.shape == (2, 2):
        return float(cov[0, 1] / (cov[0, 0] + 1e-9))
    return 0.0


def voiced_stretches_durations(f0, hop_ms=HOP_MS):
    """Computes durations of all contiguous voiced segments in an F0 contour."""
    voiced = (f0 > 0).astype(int)
    durs = []
    current_len = 0
    for val in voiced:
        if val == 1:
            current_len += 1
        else:
            if current_len > 0:
                durs.append(current_len * (hop_ms / 1000.0))
                current_len = 0
    if current_len > 0:
        durs.append(current_len * (hop_ms / 1000.0))
    return durs


def extract_features_causal(x, sr, pause_start, pause_index):
    """Causally extracts speech features strictly before pause_start.
    
    We look at:
    - Context (0 to pause_start) for normalization.
    - Last 1.5 seconds for local prosodic features.
    - Last 3.0 seconds for speaker pitch context.
    """
    pause_index = float(pause_index)
    pause_start = float(pause_start)
    
    # CRITICAL: Enforce strict causality by slicing the input audio strictly at pause_start
    # This prevents any downstream features from ever seeing full file duration or future samples.
    end_sample = int(pause_start * sr)
    x_causal = x[:end_sample]
    
    # Context segment for speaker statistics (up to last 3 seconds of causal audio)
    x_context = speech_before(x_causal, sr, pause_start, window_s=3.0)
    
    # Target segment (last 1.5 seconds of causal audio)
    seg = speech_before(x_causal, sr, pause_start, window_s=1.5)
    
    # Fallbacks if segment is empty
    if len(seg) < sr // 10:
        return np.zeros(44, dtype=np.float32)
        
    # Energy features
    e_local = frame_energy_db(seg, sr)
    e_context = frame_energy_db(x_context, sr)
    
    global_median_e = np.median(e_context) if len(e_context) > 0 else -120.0
    global_max_e = np.max(e_context) if len(e_context) > 0 else -120.0
    global_std_e = np.std(e_context) if len(e_context) > 0 else 0.0
    
    # Local energy stats
    e_last_5_mean = e_local[-5:].mean() if len(e_local) >= 5 else e_local.mean()
    e_last_15_mean = e_local[-15:].mean() if len(e_local) >= 15 else e_local.mean()
    e_last_30_mean = e_local[-30:].mean() if len(e_local) >= 30 else e_local.mean()
    
    e_diff_5 = e_last_5_mean - global_median_e
    e_diff_15 = e_last_15_mean - global_median_e
    e_diff_30 = e_last_30_mean - global_median_e
    
    e_diff_5_max = e_last_5_mean - global_max_e
    e_diff_15_max = e_last_15_mean - global_max_e
    
    e_slope_10 = get_slope(e_local[-10:]) if len(e_local) >= 10 else 0.0
    e_slope_30 = get_slope(e_local[-30:]) if len(e_local) >= 30 else 0.0
    
    # Decay relative to local peak in the target segment
    e_decay_5 = global_max_e - e_last_5_mean
    e_decay_15 = global_max_e - e_last_15_mean
    e_decay_30 = global_max_e - e_last_30_mean
    
    # Pitch (F0) features
    f0_context = f0_contour(x_context, sr)
    f0_local = f0_contour(seg, sr)
    
    voiced_context = f0_context[f0_context > 0]
    voiced_local = f0_local[f0_local > 0]
    
    voiced_ratio = len(voiced_local) / len(f0_local) if len(f0_local) > 0 else 0.0
    
    if len(voiced_context) >= 3:
        global_median_f0 = np.median(voiced_context)
        global_std_f0 = np.std(voiced_context)
    else:
        global_median_f0 = 150.0
        global_std_f0 = 50.0
        
    if len(voiced_local) >= 3:
        f0_last_3_mean = voiced_local[-3:].mean()
        f0_last_10_mean = voiced_local[-10:].mean() if len(voiced_local) >= 10 else voiced_local.mean()
        
        # Z-score normalized pitch
        f0_diff_3 = (f0_last_3_mean - global_median_f0) / (global_std_f0 + 1e-6)
        f0_diff_10 = (f0_last_10_mean - global_median_f0) / (global_std_f0 + 1e-6)
        
        f0_slope_voiced = get_slope(voiced_local[-10:]) if len(voiced_local) >= 10 else get_slope(voiced_local)
    else:
        f0_last_3_mean = 0.0
        f0_last_10_mean = 0.0
        f0_diff_3 = 0.0
        f0_diff_10 = 0.0
        f0_slope_voiced = 0.0
        
    # Final-syllable lengthening features
    context_durs = voiced_stretches_durations(f0_context)
    # Find duration of the last voiced stretch in f0_local
    voiced_indices_local = np.where(f0_local > 0)[0]
    if len(voiced_indices_local) > 0:
        last_idx = voiced_indices_local[-1]
        start_idx = last_idx
        while start_idx - 1 in voiced_indices_local:
            start_idx -= 1
        last_voiced_stretch_dur = (last_idx - start_idx + 1) * (HOP_MS / 1000.0)
    else:
        last_voiced_stretch_dur = 0.0
        
    if len(context_durs) > 0:
        median_context_dur = np.median(context_durs)
        voiced_stretch_ratio = last_voiced_stretch_dur / (median_context_dur + 1e-6)
    else:
        voiced_stretch_ratio = 1.0
        
    # Zero Crossing Rate (ZCR) features
    fr_seg = frames(seg, sr)
    fr_context = frames(x_context, sr)
    
    zcr_local = np.array([zero_crossing_rate(f) for f in fr_seg], dtype=np.float32)
    zcr_context = np.array([zero_crossing_rate(f) for f in fr_context], dtype=np.float32)
    
    global_median_zcr = np.median(zcr_context) if len(zcr_context) > 0 else 0.0
    
    zcr_last_5_mean = zcr_local[-5:].mean() if len(zcr_local) >= 5 else zcr_local.mean()
    zcr_last_15_mean = zcr_local[-15:].mean() if len(zcr_local) >= 15 else zcr_local.mean()
    
    zcr_diff_5 = zcr_last_5_mean - global_median_zcr
    zcr_diff_15 = zcr_last_15_mean - global_median_zcr
    
    zcr_slope_10 = get_slope(zcr_local[-10:]) if len(zcr_local) >= 10 else 0.0
    
    # Spectral Centroid, Roll-off, and High/Low ratio
    sc_local = []
    sr_local = []
    hl_local = []
    
    for f in fr_seg:
        c, r, h = spectral_centroid_and_rolloff_and_ratio(f, sr)
        sc_local.append(c)
        sr_local.append(r)
        hl_local.append(h)
        
    sc_local = np.array(sc_local, dtype=np.float32)
    sr_local = np.array(sr_local, dtype=np.float32)
    hl_local = np.array(hl_local, dtype=np.float32)
    
    sc_context = []
    sr_context = []
    hl_context = []
    for f in fr_context:
        c, r, h = spectral_centroid_and_rolloff_and_ratio(f, sr)
        sc_context.append(c)
        sr_context.append(r)
        hl_context.append(h)
        
    global_median_sc = np.median(sc_context) if len(sc_context) > 0 else 1000.0
    global_median_sr = np.median(sr_context) if len(sr_context) > 0 else 2000.0
    global_median_hl = np.median(hl_context) if len(hl_context) > 0 else 0.1
    
    sc_last_5_mean = sc_local[-5:].mean() if len(sc_local) >= 5 else sc_local.mean()
    sr_last_5_mean = sr_local[-5:].mean() if len(sr_local) >= 5 else sr_local.mean()
    hl_last_5_mean = hl_local[-5:].mean() if len(hl_local) >= 5 else hl_local.mean()
    
    sc_last_15_mean = sc_local[-15:].mean() if len(sc_local) >= 15 else sc_local.mean()
    sr_last_15_mean = sr_local[-15:].mean() if len(sr_local) >= 15 else sr_local.mean()
    hl_last_15_mean = hl_local[-15:].mean() if len(hl_local) >= 15 else hl_local.mean()
    
    sc_diff_5 = sc_last_5_mean - global_median_sc
    sr_diff_5 = sr_last_5_mean - global_median_sr
    hl_diff_5 = hl_last_5_mean - global_median_hl
    
    sc_diff_15 = sc_last_15_mean - global_median_sc
    sr_diff_15 = sr_local[-15:].mean() - global_median_sr
    hl_diff_15 = hl_local[-15:].mean() - global_median_hl
    
    # --- New Advanced Causal Features ---
    # 1. Pitch Delta: slope of voiced pitch in the final 200ms (last 20 frames)
    voiced_last_20 = voiced_local[-20:] if len(voiced_local) > 0 else []
    f0_slope_last_20 = get_slope(voiced_last_20) if len(voiced_last_20) >= 3 else 0.0
    
    # 2. ZCR Spike: shift in ZCR in the final 100ms relative to context
    zcr_spike = zcr_local[-10:].mean() - zcr_local[-30:-10].mean() if len(zcr_local) >= 30 else 0.0
    
    # 3. Energy Drop: local energy drop in final 100ms relative to context
    e_drop_last_10 = e_local[-30:-10].mean() - e_local[-10:].mean() if len(e_local) >= 30 else 0.0
    
    # 4. Breath Feature: product of positive ZCR spike and energy drop
    breath_feature = max(0.0, zcr_spike) * max(0.0, e_drop_last_10)
    
    features_vec = np.array([
        pause_index,
        pause_start,
        
        # Energy features
        e_last_5_mean,
        e_last_15_mean,
        e_last_30_mean,
        e_diff_5,
        e_diff_15,
        e_diff_30,
        e_diff_5_max,
        e_diff_15_max,
        e_slope_10,
        e_slope_30,
        
        # Pitch features
        voiced_ratio,
        f0_last_3_mean,
        f0_last_10_mean,
        f0_diff_3,
        f0_diff_10,
        f0_slope_voiced,
        
        # ZCR features
        zcr_last_5_mean,
        zcr_last_15_mean,
        zcr_diff_5,
        zcr_diff_15,
        zcr_slope_10,
        
        # Spectral features
        sc_last_5_mean,
        sr_last_5_mean,
        hl_last_5_mean,
        sc_last_15_mean,
        sr_last_15_mean,
        hl_last_15_mean,
        sc_diff_5,
        sr_diff_5,
        hl_diff_5,
        sc_diff_15,
        sr_diff_15,
        hl_diff_15,
        
        # Additions
        last_voiced_stretch_dur,
        voiced_stretch_ratio,
        e_decay_5,
        e_decay_15,
        e_decay_30,
        
        # Advanced additions
        f0_slope_last_20,
        zcr_spike,
        e_drop_last_10,
        breath_feature
    ], dtype=np.float32)
    
    features_vec = np.nan_to_num(features_vec, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features_vec


def verify_causality_invariance(x, sr, pause_start, pause_index):
    """Regression test verifying that audio after pause_start cannot affect extraction.
    
    Corrupts the audio following the causal pause boundary and asserts bit-identical features.
    """
    f1 = extract_features_causal(x, sr, pause_start, pause_index)
    
    # Corrupt audio past pause_start with random noise
    end_sample = int(float(pause_start) * sr)
    x_corrupted = x.copy()
    if end_sample < len(x_corrupted):
        x_corrupted[end_sample:] = np.random.normal(0.0, 1.0, len(x_corrupted) - end_sample)
        
    f2 = extract_features_causal(x_corrupted, sr, pause_start, pause_index)
    
    # Assert bit-identity
    np.testing.assert_array_almost_equal(
        f1, f2, decimal=6, 
        err_msg="Causality violation regression guard triggered: features changed after audio corruption!"
    )
