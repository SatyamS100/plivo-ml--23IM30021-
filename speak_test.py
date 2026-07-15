"""Optional utility for live speaking tests.

Usage:
  python speak_test.py --wav <path_to_wav> --pause_start <seconds>

Example:
  python speak_test.py --wav voice_memo.wav --pause_start 3.2
"""
import os
import pickle
import argparse
import numpy as np

from features import load_wav, extract_features_causal

def main():
    ap = argparse.ArgumentParser(description="Live speech End-of-Turn prediction check.")
    ap.add_argument("--wav", required=True, help="Path to your 16 kHz mono WAV file")
    ap.add_argument("--pause_start", required=True, type=float, help="Time in seconds when you paused")
    args = ap.parse_args()

    # Verify model exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "best_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found. Ensure best_model.pkl is in {script_dir}")

    # Load model
    print(f"Loading Voting Ensemble model from {model_path}...")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    clf = model_data["model"]

    # Load audio
    print(f"Loading audio file: {args.wav}...")
    if not os.path.exists(args.wav):
        raise FileNotFoundError(f"Audio file not found: {args.wav}")
    x, sr = load_wav(args.wav)
    print(f"Loaded audio with sample rate {sr} Hz, total length {len(x)/sr:.2f} seconds.")

    if args.pause_start > len(x) / sr:
        raise ValueError(f"pause_start ({args.pause_start}s) is greater than total audio duration ({len(x)/sr:.2f}s).")

    # Extract causal features (40D vector)
    # Pause index is assumed to be 0 for a single test pause
    print(f"Extracting features strictly up to t = {args.pause_start}s...")
    feat = extract_features_causal(x, sr, args.pause_start, pause_index=0)

    # Predict
    p_eot = clf.predict_proba(feat.reshape(1, -1))[0, 1]

    print("\n==========================================")
    print(f"Audio File:  {args.wav}")
    print(f"Pause Start: {args.pause_start}s")
    print(f"Predicted p_eot: {p_eot:.4f}")
    if p_eot >= 0.45:
        print("Result:      [END OF TURN] (Agent will take the floor)")
    else:
        print("Result:      [HOLD] (Agent will keep waiting for speech)")
    print("==========================================")
    print("Note: The 0.45 operating point threshold represents the VotingEnsemble's optimal threshold.")

if __name__ == "__main__":
    main()
