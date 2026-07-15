import os
import csv
import pickle
import argparse
import numpy as np

from features import load_wav, extract_features_causal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Determine paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "best_model.pkl")
    
    if not os.path.exists(model_path):
        model_path = os.path.join(script_dir, "starter", "best_model.pkl")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found. Ensure best_model.pkl is in {script_dir}")
        
    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    clf = model_data["model"]
    scaler = model_data["scaler"]
    top_indices = model_data["selected_feature_indices"]
    print(f"Model loaded successfully. Description: {model_data['description']}")

    labels_path = os.path.join(args.data_dir, "labels.csv")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels CSV not found at {labels_path}")

    out_path = args.out if args.out is not None else os.path.join(args.data_dir, "predictions.csv")

    with open(labels_path, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} pauses from labels.csv")

    audio_cache = {}
    predictions = []

    for i, r in enumerate(rows):
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in audio_cache:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Audio file not found: {path}")
            audio_cache[path] = load_wav(path)
            
        x, sr = audio_cache[path]
        
        # Extract features (44D vector)
        feat = extract_features_causal(x, sr, float(r["pause_start"]), int(r["pause_index"]))
        
        # Scale and prune features causally using saved parameters
        feat_scaled = scaler.transform(feat.reshape(1, -1))
        if top_indices is not None:
            feat_final = feat_scaled[:, top_indices]
        else:
            feat_final = feat_scaled
            
        # Predict continuous raw probability of EOT (class 1)
        p_eot = clf.predict_proba(feat_final)[0, 1]
        
        predictions.append({
            "turn_id": r["turn_id"],
            "pause_index": int(r["pause_index"]),
            "p_eot": float(p_eot)
        })
        
        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            print(f"Processed {i + 1} / {len(rows)} pauses")

    # Write predictions (raw continuous probabilities, no thresholding)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for p in predictions:
            w.writerow([p["turn_id"], p["pause_index"], f"{p['p_eot']:.4f}"])

    print(f"Successfully wrote {len(predictions)} predictions -> {out_path}")

if __name__ == "__main__":
    main()
