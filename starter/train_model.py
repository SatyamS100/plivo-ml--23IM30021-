import os
import csv
import pickle
import subprocess
import re
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier

from features import load_wav, extract_features_causal, verify_causality_invariance

# List of all 70 feature names in order of return from features.py
FEATURE_NAMES = [
    "pause_index", "pause_start",
    # Energy features
    "e_last_5_mean", "e_last_15_mean", "e_last_30_mean", "e_diff_5", "e_diff_15", "e_diff_30", "e_diff_5_max", "e_diff_15_max", "e_slope_10", "e_slope_30",
    # Pitch features
    "voiced_ratio", "f0_last_3_mean", "f0_last_10_mean", "f0_diff_3", "f0_diff_10", "f0_slope_voiced",
    # ZCR features
    "zcr_last_5_mean", "zcr_last_15_mean", "zcr_diff_5", "zcr_diff_15", "zcr_slope_10",
    # Spectral features
    "sc_last_5_mean", "sr_last_5_mean", "hl_last_5_mean", "sc_last_15_mean", "sr_last_15_mean", "hl_last_15_mean", "sc_diff_5", "sr_diff_5", "hl_diff_5", "sc_diff_15", "sr_diff_15", "hl_diff_15",
    # Additions
    "last_voiced_stretch_dur", "voiced_stretch_ratio", "e_decay_5", "e_decay_15", "e_decay_30",
    # Advanced additions
    "f0_slope_last_20", "zcr_spike", "e_drop_last_10", "breath_feature"
] + [f"mfcc_mean_{i}" for i in range(13)] + [f"mfcc_std_{i}" for i in range(13)]

def parse_score_output(stdout_text):
    delay_match = re.search(r"mean response delay\s*:\s*(\d+)\s*ms", stdout_text)
    thresh_match = re.search(r"threshold\s*=\s*([\d\.]+)", stdout_text)
    delay = float(delay_match.group(1)) if delay_match else 1600.0
    thresh = float(thresh_match.group(1)) if thresh_match else 1.0
    return delay, thresh

def run_official_scorer(pauses, data_dir, temp_csv_name):
    abs_temp_path = os.path.abspath(os.path.join("c:/Users/ok/Desktop/Plivo", temp_csv_name))
    with open(abs_temp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for pz in pauses:
            w.writerow([pz["turn_id"], pz["pause_index"], f"{pz['p']:.4f}"])
            
    cmd = ["python", "starter/score.py", "--data_dir", data_dir, "--pred", abs_temp_path]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd="c:/Users/ok/Desktop/Plivo")
    
    if res.returncode != 0:
        if os.path.exists(abs_temp_path):
            os.remove(abs_temp_path)
        raise RuntimeError(f"score.py failed with exit code {res.returncode}. Stderr: {res.stderr}")
        
    if os.path.exists(abs_temp_path):
        os.remove(abs_temp_path)
        
    return parse_score_output(res.stdout)

def main():
    cache_file = "features_cache.pkl"
    data_dirs = {
        "english": "c:/Users/ok/Desktop/Plivo/eot_data/english",
        "hindi": "c:/Users/ok/Desktop/Plivo/eot_data/hindi"
    }
    
    # Check cache
    if os.path.exists(cache_file):
        print("Loading features from cache...")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
    else:
        print("Regenerating features cache...")
        # (Feature caching code skipped since train_model.py is normally run with cache, but let's include extraction loop just in case cache is deleted)
        data = {}
        for lang, data_dir in data_dirs.items():
            labels_path = os.path.join(data_dir, "labels.csv")
            rows = list(csv.DictReader(open(labels_path)))
            X, y, groups, meta = [], [], [], []
            audio_cache = {}
            for r in rows:
                path = os.path.join(data_dir, r["audio_file"])
                if path not in audio_cache:
                    audio_cache[path] = load_wav(path)
                x, sr = audio_cache[path]
                feat = extract_features_causal(x, sr, float(r["pause_start"]), int(r["pause_index"]))
                X.append(feat)
                y.append(1 if r["label"] == "eot" else 0)
                groups.append(lang + "_" + r["turn_id"])
                meta.append({
                    "turn_id": r["turn_id"],
                    "pause_index": int(r["pause_index"]),
                    "dur": float(r["pause_end"]) - float(r["pause_start"]),
                    "label": r["label"],
                    "lang": lang
                })
            data[lang] = {"X": np.array(X), "y": np.array(y), "groups": np.array(groups), "meta": meta}
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
            
    X_all = np.vstack([data["english"]["X"], data["hindi"]["X"]])
    y_all = np.concatenate([data["english"]["y"], data["hindi"]["y"]])
    groups_all = np.concatenate([data["english"]["groups"], data["hindi"]["groups"]])
    meta_all = data["english"]["meta"] + data["hindi"]["meta"]
    
    # 5-fold Stratified Group K-Fold CV
    sgkf = StratifiedGroupKFold(n_splits=5)
    
    # RandomForest Ensemble model with false cutoff penalization
    classifiers = {
        "RandomForest_ensemble": RandomForestClassifier(n_estimators=100, class_weight={0: 1, 1: 10}, random_state=42, n_jobs=-1)
    }

    best_avg_delay = 9999.0
    best_model_run_name = None
    best_run_metrics = {}
    
    # Evaluate under two modes: Full Features vs. Pruned (Top 15 selected per fold)
    for mode in ["FullFeatures", "PrunedTop15"]:
        print(f"\n==========================================\nMODE: {mode} (No Scaling)\n==========================================")
        for name, clf in classifiers.items():
            run_name = f"{name}_{mode}"
            print(f"\n--- Evaluating {run_name} ---")
            
            oof_preds = np.zeros(len(y_all))
            fold_aucs = []
            
            for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_all, y_all, groups_all)):
                X_tr, y_tr = X_all[train_idx], y_all[train_idx]
                X_v, y_v = X_all[val_idx], y_all[val_idx]
                
                # NO SCALING PERFORMED - Raw features passed directly to Trees
                if mode == "PrunedTop15":
                    selector = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
                    selector.fit(X_tr, y_tr)
                    importances = selector.feature_importances_
                    top_indices = np.argsort(importances)[::-1][:15]
                    
                    X_tr_final = X_tr[:, top_indices]
                    X_v_final = X_v[:, top_indices]
                else:
                    X_tr_final = X_tr
                    X_v_final = X_v
                
                clf.fit(X_tr_final, y_tr)
                oof_preds[val_idx] = clf.predict_proba(X_v_final)[:, 1]
                
                from sklearn.metrics import roc_auc_score
                if len(np.unique(y_v)) > 1:
                    fold_auc = roc_auc_score(y_v, oof_preds[val_idx])
                    fold_aucs.append(fold_auc)
                    
            from sklearn.metrics import roc_auc_score
            mean_auc = roc_auc_score(y_all, oof_preds)
            std_auc = np.std(fold_aucs) if fold_aucs else 0.0
            print(f"OOF ROC-AUC: {mean_auc:.4f} (fold std: {std_auc:.4f})")
            print(f"Fold AUCs: {[round(x, 4) for x in fold_aucs]}")
            
            assert len(oof_preds) == len(meta_all)
            pauses_eval = []
            for idx, pred in enumerate(oof_preds):
                pz = meta_all[idx].copy()
                pz["p"] = pred
                pauses_eval.append(pz)
                
            lang_pauses = {
                "english": [p for p in pauses_eval if p["lang"] == "english"],
                "hindi": [p for p in pauses_eval if p["lang"] == "hindi"]
            }
            
            en_delay, en_thresh = run_official_scorer(lang_pauses["english"], data_dirs["english"], "tmp_oof_en.csv")
            hi_delay, hi_thresh = run_official_scorer(lang_pauses["hindi"], data_dirs["hindi"], "tmp_oof_hi.csv")
            
            print(f"  English OOF Mean Delay: {en_delay:.0f} ms (at threshold={en_thresh:.2f})")
            print(f"  Hindi OOF Mean Delay:   {hi_delay:.0f} ms (at threshold={hi_thresh:.2f})")
            
            avg_delay = (en_delay + hi_delay) / 2.0
            print(f"Average Language Delay: {avg_delay:.1f} ms")
            
            if avg_delay < best_avg_delay:
                best_avg_delay = avg_delay
                best_model_run_name = run_name
                best_run_metrics = {
                    "name": name,
                    "mode": mode,
                    "auc": mean_auc,
                    "auc_std": std_auc,
                    "fold_aucs": fold_aucs,
                    "english": {"delay": en_delay, "threshold": en_thresh},
                    "hindi": {"delay": hi_delay, "threshold": hi_thresh}
                }

    # Final fit on all combined data
    print(f"\nTraining best model ({best_model_run_name}) on all data...")
    final_clf_base = classifiers[best_run_metrics["name"]]
    
    # Always compute and print the Top 15 features to the terminal
    print("Computing feature importances on entire dataset...")
    temp_selector = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    temp_selector.fit(X_all, y_all)
    final_importances = temp_selector.feature_importances_
    final_top_indices = np.argsort(final_importances)[::-1][:15]
    
    print("\n==========================================")
    print("TOP 15 MOST IMPORTANT FEATURES:")
    for rank, idx in enumerate(final_top_indices):
        name = FEATURE_NAMES[idx]
        importance = final_importances[idx]
        print(f"  Rank {rank+1}: {name} (Importance: {importance:.4f}, Original Index: {idx})")
    print("==========================================\n")
    
    if best_run_metrics["mode"] == "PrunedTop15":
        X_all_final = X_all[:, final_top_indices]
    else:
        final_top_indices = None
        X_all_final = X_all
        
    final_clf_base.fit(X_all_final, y_all)
    
    # Save best model pickle
    model_data = {
        "model": final_clf_base,
        "scaler": None, # Completely deleted scaler to prevent scaling warping
        "selected_feature_indices": final_top_indices,
        "features_version": "causal_v4_70d_raw",
        "description": f"Best CV model: {best_model_run_name} (Unscaled RF)",
        "metrics": best_run_metrics
    }
    with open("best_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print("Model saved to best_model.pkl")
    
    print("\n==========================================")
    print(f"BEST MODEL: {best_model_run_name}")
    print(f"Overall OOF AUC: {best_run_metrics['auc']:.4f} (std: {best_run_metrics['auc_std']:.4f})")
    print(f"Fold AUCs: {[round(x, 4) for x in best_run_metrics['fold_aucs']]}")
    print(f"English OOF Mean Delay: {best_run_metrics['english']['delay']:.0f} ms (threshold={best_run_metrics['english']['threshold']})")
    print(f"Hindi OOF Mean Delay:   {best_run_metrics['hindi']['delay']:.0f} ms (threshold={best_run_metrics['hindi']['threshold']})")
    print("==========================================")

if __name__ == "__main__":
    main()
