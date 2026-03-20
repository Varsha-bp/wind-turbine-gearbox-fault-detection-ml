"""
train_model.py
--------------
Generates dataset (if needed) and trains the Random Forest model.
Run this BEFORE launching the Flask app.

Usage:
    python train_model.py
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

DATA_PATH = os.path.join('data', 'gearbox_vibration_data.csv')
CLASS_NAMES = {0:'Healthy', 1:'Inner_Race_Fault', 2:'Outer_Race_Fault', 3:'Ball_Fault', 4:'Gear_Wear'}

def main():
    print("=" * 55)
    print("  WindGuard AI — Model Training Pipeline")
    print("=" * 55)

    # ── Step 1: Generate dataset if missing ──────────────
    if not os.path.exists(DATA_PATH):
        print("\n[1] Dataset not found. Generating now...")
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, os.path.join('data', 'generate_dataset.py')],
            capture_output=False
        )
        if result.returncode != 0:
            print("ERROR: Dataset generation failed.")
            sys.exit(1)
    else:
        print(f"\n[1] Dataset found: {DATA_PATH}")
        import pandas as pd
        df = pd.read_csv(DATA_PATH)
        print(f"    Shape: {df.shape}  |  Classes: {df['condition'].value_counts().to_dict()}")

    # ── Step 2: Preprocess ────────────────────────────────
    print("\n[2] Preprocessing data...")
    from src.preprocessing import full_pipeline
    t0 = time.time()
    X_train, X_test, y_train, y_test, feature_names, scaler = full_pipeline(DATA_PATH)
    print(f"    Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"    Features: {len(feature_names)}")
    print(f"    Done in {time.time()-t0:.1f}s")

    # ── Step 3: Train Random Forest ───────────────────────
    print("\n[3] Training Random Forest (300 trees)...")
    from sklearn.ensemble import RandomForestClassifier
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")

    # ── Step 4: Evaluate ──────────────────────────────────
    print("\n[4] Evaluating...")
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = rf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=list(CLASS_NAMES.values()))
    print(f"    Accuracy: {acc*100:.2f}%")
    print(report)

    if acc < 0.90:
        print("⚠  WARNING: Accuracy below 90% target.")
    else:
        print(f"✓  Target accuracy ≥90% achieved! ({acc*100:.2f}%)")

    # ── Step 5: Save ──────────────────────────────────────
    print("\n[5] Saving model and metadata...")
    import joblib
    os.makedirs('models', exist_ok=True)

    joblib.dump(rf, os.path.join('models', 'best_model.pkl'))

    meta = {
        'model_name':    'Random Forest',
        'feature_names': feature_names,
        'class_names':   {str(k): v for k, v in CLASS_NAMES.items()},
        'n_features':    len(feature_names),
        'accuracy':      round(acc, 4),
    }
    with open(os.path.join('models', 'model_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    summary = {
        'best_model':    'Random Forest',
        'best_accuracy': round(acc, 4),
        'feature_count': len(feature_names),
        'train_samples': int(X_train.shape[0]),
        'test_samples':  int(X_test.shape[0]),
    }
    with open(os.path.join('models', 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"    ✓ models/best_model.pkl")
    print(f"    ✓ models/model_metadata.json")
    print(f"    ✓ models/scaler.pkl  (saved by preprocessing)")
    print(f"    ✓ models/training_summary.json")

    print("\n" + "=" * 55)
    print(f"  TRAINING COMPLETE — {acc*100:.2f}% accuracy")
    print("  Launch app: python app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
