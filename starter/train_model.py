import os
import csv
import pickle
import subprocess
import re
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

from features import load_wav, extract_features_causal, verify_causality_invariance

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_score_output(stdout_text):
    """Parses the console output of score.py to extract mean response delay and threshold.
    
    Expected format:
      mean response delay : X ms
      operating point     : threshold=T, delay=D ms
    """
    delay_match = re.search(r"mean response delay\s*:\s*(\d+)\s*ms", stdout_text)
    thresh_match = re.search(r"threshold\s*=\s*([\d\.]+)", stdout_text)
    
    delay = float(delay_match.group(1)) if delay_match else 1600.0
    thresh = float(thresh_match.group(1)) if thresh_match else 1.0
    return delay, thresh

def run_official_scorer(pauses, data_dir, temp_csv_name):
    """Saves predictions to a temporary CSV and calls the official score.py script."""
    # Write predictions using a path relative to the repository root
    abs_temp_path = os.path.join(BASE_DIR, temp_csv_name)
    with open(abs_temp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for pz in pauses:
            w.writerow([pz["turn_id"], pz["pause_index"], f"{pz['p']:.4f}"])
            
    # Subprocess call to the official score.py
    cmd = ["python", "starter/score.py", "--data_dir", data_dir, "--pred", abs_temp_path]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
    
    # Ensure subprocess exited successfully
    if res.returncode != 0:
        if os.path.exists(abs_temp_path):
            os.remove(abs_temp_path)
        raise RuntimeError(f"score.py failed with exit code {res.returncode}. Stderr: {res.stderr}")
        
    # Clean up temp file
    if os.path.exists(abs_temp_path):
        os.remove(abs_temp_path)
        
    return parse_score_output(res.stdout)

def main():
    cache_file = "features_cache.pkl"
    
    # Remove old cache to force regeneration with advanced features
    if os.path.exists(cache_file):
        print("Removing old cache file to regenerate advanced features...")
        os.remove(cache_file)
        
    data_dirs = {
        "english": os.path.join(BASE_DIR, "eot_data", "english"),
        "hindi": os.path.join(BASE_DIR, "eot_data", "hindi")
    }
    
    print("Extracting features (this may take a minute)...")
    data = {}
    regression_test_sample = None
    
    for lang, data_dir in data_dirs.items():
        labels_path = os.path.join(data_dir, "labels.csv")
        rows = list(csv.DictReader(open(labels_path)))
        
        X, y, groups, meta = [], [], [], []
        audio_cache = {}
        
        for i, r in enumerate(rows):
            path = os.path.join(data_dir, r["audio_file"])
            if path not in audio_cache:
                audio_cache[path] = load_wav(path)
            x, sr = audio_cache[path]
            
            # Save the first valid sample for causality regression testing
            if regression_test_sample is None and float(r["pause_start"]) > 2.0:
                regression_test_sample = (x, sr, float(r["pause_start"]), int(r["pause_index"]))
            
            # Extract features (44D vector)
            feat = extract_features_causal(x, sr, float(r["pause_start"]), int(r["pause_index"]))
            X.append(feat)
            y.append(1 if r["label"] == "eot" else 0)
            
            # Turn ID needs to be unique across languages
            groups.append(lang + "_" + r["turn_id"])
            
            meta.append({
                "turn_id": r["turn_id"],
                "pause_index": int(r["pause_index"]),
                "dur": float(r["pause_end"]) - float(r["pause_start"]),
                "label": r["label"],
                "lang": lang
            })
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1} / {len(rows)} pauses in {lang}")
        
        data[lang] = {
            "X": np.array(X),
            "y": np.array(y),
            "groups": np.array(groups),
            "meta": meta
        }
    
    # Run the causality verification regression test on the new 44D features
    if regression_test_sample is not None:
        print("Running automated causality regression test...")
        verify_causality_invariance(*regression_test_sample)
        print("Causality regression test PASSED: post-pause audio does not affect feature extraction.")
    else:
        print("Warning: Could not run causality regression test.")
        
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
        print("Features cached.")

    # Combine data for training and evaluation
    X_all = np.vstack([data["english"]["X"], data["hindi"]["X"]])
    y_all = np.concatenate([data["english"]["y"], data["hindi"]["y"]])
    groups_all = np.concatenate([data["english"]["groups"], data["hindi"]["groups"]])
    meta_all = data["english"]["meta"] + data["hindi"]["meta"]
    
    n_turns = len(np.unique(groups_all))
    n_pauses = X_all.shape[0]
    print(f"Total dataset: {n_pauses} samples, {n_turns} turns")
    
    # 5-fold Stratified Group K-Fold CV to balance holds/eots per fold
    sgkf = StratifiedGroupKFold(n_splits=5)
    
    print("Fold compositions:")
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_all, y_all, groups_all)):
        val_turns = len(np.unique(groups_all[val_idx]))
        val_pauses = len(val_idx)
        print(f"  Fold {fold + 1}: {val_turns} val turns, {val_pauses} val pauses")
    
    # Model candidates
    classifiers = {
        "LogisticRegression_C0.1": LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", random_state=42),
        "RandomForest_d8": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42),
        "SVC_rbf": SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=60, max_depth=4, learning_rate=0.05, class_weight="balanced", random_state=42, verbose=-1),
        "VotingEnsemble": VotingClassifier(estimators=[
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)),
            ("svc", SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
            ("lr", LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", random_state=42))
        ], voting="soft")
    }

    best_avg_delay = 9999.0
    best_model_run_name = None
    best_run_metrics = {}
    
    # Evaluate each classifier under two modes: Full Features vs. Pruned (Top 15 selected per fold)
    for mode in ["FullFeatures", "PrunedTop15"]:
        print(f"\n==========================================\nMODE: {mode}\n==========================================")
        for name, clf in classifiers.items():
            run_name = f"{name}_{mode}"
            print(f"\n--- Evaluating {run_name} ---")
            
            oof_preds = np.zeros(len(y_all))
            fold_aucs = []
            
            # OOF cross-validation loop
            for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_all, y_all, groups_all)):
                X_tr, y_tr = X_all[train_idx], y_all[train_idx]
                X_v, y_v = X_all[val_idx], y_all[val_idx]
                
                # Preprocessing 1: Standard scaling
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_v_scaled = scaler.transform(X_v)
                
                # Preprocessing 2: Feature selection (inside fold to avoid leakage)
                if mode == "PrunedTop15":
                    selector = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
                    selector.fit(X_tr_scaled, y_tr)
                    importances = selector.feature_importances_
                    top_indices = np.argsort(importances)[::-1][:15]
                    
                    X_tr_final = X_tr_scaled[:, top_indices]
                    X_v_final = X_v_scaled[:, top_indices]
                else:
                    X_tr_final = X_tr_scaled
                    X_v_final = X_v_scaled
                
                # Train the fold model
                clf.fit(X_tr_final, y_tr)
                oof_preds[val_idx] = clf.predict_proba(X_v_final)[:, 1]
                
                # Calculate fold ROC-AUC
                from sklearn.metrics import roc_auc_score
                if len(np.unique(y_v)) > 1:
                    fold_auc = roc_auc_score(y_v, oof_preds[val_idx])
                    fold_aucs.append(fold_auc)
                    
            from sklearn.metrics import roc_auc_score
            mean_auc = roc_auc_score(y_all, oof_preds)
            std_auc = np.std(fold_aucs) if fold_aucs else 0.0
            print(f"OOF ROC-AUC: {mean_auc:.4f} (fold std: {std_auc:.4f})")
            print(f"Fold AUCs: {[round(x, 4) for x in fold_aucs]}")
            
            # CRITICAL DENOMINATOR BUG PROTECTION: Assert OOF matches input labels count exactly
            assert len(oof_preds) == len(meta_all), f"Length mismatch: {len(oof_preds)} predictions vs {len(meta_all)} labels."
            
            # Assemble pauses for evaluation
            pauses_eval = []
            for idx, pred in enumerate(oof_preds):
                pz = meta_all[idx].copy()
                pz["p"] = pred
                pauses_eval.append(pz)
                
            # Split and call the official score.py via subprocess separately per language
            lang_pauses = {
                "english": [p for p in pauses_eval if p["lang"] == "english"],
                "hindi": [p for p in pauses_eval if p["lang"] == "hindi"]
            }
            
            # Verify splits are complete
            assert len(lang_pauses["english"]) == len(data["english"]["meta"]), "English OOF slice size mismatch."
            assert len(lang_pauses["hindi"]) == len(data["hindi"]["meta"]), "Hindi OOF slice size mismatch."
            
            # Execute subprocess calls
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
    
    # Scale all features
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all)
    
    # Prune if the best mode was PrunedTop15
    if best_run_metrics["mode"] == "PrunedTop15":
        print("Selecting features on entire dataset...")
        final_selector = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        final_selector.fit(X_all_scaled, y_all)
        final_importances = final_selector.feature_importances_
        final_top_indices = np.argsort(final_importances)[::-1][:15]
        X_all_final = X_all_scaled[:, final_top_indices]
    else:
        final_top_indices = None
        X_all_final = X_all_scaled
        
    final_clf_base.fit(X_all_final, y_all)
    
    # Save best model pickle
    model_data = {
        "model": final_clf_base,
        "scaler": final_scaler,
        "selected_feature_indices": final_top_indices,
        "features_version": "causal_v4_44d",
        "description": f"Best CV model: {best_model_run_name}",
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
