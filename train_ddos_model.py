"""
Multi-episode DDoS early-warning PREDICTIVE MODEL (2-train / 1-test version).

PURPOSE
-------
This is the predictive half of the pipeline (mitigation is a separate,
later component). It trains a two-stage cascade forecaster:

  Stage 1 (SVM, fast gatekeeper): looks at the CURRENT system snapshot
  (cpu/memory + their velocities) and quickly rules out "normal" traffic
  with high confidence, so the heavier model only has to run on the
  ambiguous cases.

  Stage 2 (LSTM, sequence model): looks at the last TIME_STEPS seconds of
  network-level features (requests/failures/ratios) and forecasts the
  system's state FORECAST_HORIZON seconds into the future -> this is what
  gives you an early warning before an attack fully lands.

WHY 2-TRAIN / 1-TEST
---------------------
The model is fit on TWO episodes' worth of normal->warning->attack
patterns (more variety for the LSTM/SVM to learn the general shape of an
attack ramp-up from) and then forecasts on the ONE remaining, completely
unseen episode. Rotation runs over every episode taking a turn as the
held-out test episode, so you get 3 rotations total and can see how
stable the model is across which episode is held out.

--- Expected input (YOUR REAL DATA, no synthetic placeholder) ---
Put your three real CSVs in EPISODES_DIR (default: "episodes/"), named
so each file's number is discoverable, e.g.:
    episode1.csv, episode2.csv, episode3.csv
  (episode_1.csv / Episode1.csv / EPISODE_1.csv also match)
Each CSV must have the SAME columns as your dataset (Timestamp,
cpu_percent, memory_percent, memory_usage, Requests/s, Failures/s,
Total Request Count, Total Failure Count, label, ...).

In Google Colab: use the file browser on the left ("Files" panel) ->
upload episode1.csv, episode2.csv, episode3.csv into a folder called
"episodes" (create it if it doesn't exist), or just upload them directly
into /content/ and set EPISODES_DIR = "" below. There is NO synthetic
data generation in this version -- if your files aren't found, the
script will stop and tell you what it looked for instead of silently
faking data.
"""

import os
import re
import glob
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# CONFIG
# ============================================================
EPISODES_DIR = "episodes"          # folder containing your real episode CSV files
TIME_STEPS = 5                      # lookback window (seconds)
FORECAST_HORIZON = 10               # how many seconds ahead to forecast
CONFIDENCE_THRESHOLD = 0.85         # Stage 1 SVM gatekeeper threshold

# Your real CSVs use different column names than the original placeholder
# schema for the timestamp and label columns -- point at them here instead
# of hardcoding 'Timestamp'/'label' throughout the code.
TIMESTAMP_COL = 'Timestamp.1'       # clean parseable datetime string column
                                     # (NOT 'Timestamp', which looks like a raw
                                     # epoch/counter and would parse wrong)
LABEL_COL = 'Phase'                 # e.g. 'Normal' / 'Warning' / 'Attack'

STAGE1_FEATURES = ['cpu_percent', 'memory_percent', 'memory_usage', 'cpu_velocity', 'memory_velocity']
STAGE2_FEATURES = ['Requests/s', 'Failures/s', 'Total Request Count', 'Total Failure Count',
                    'traffic_velocity', 'network_to_cpu_ratio', 'failure_rate_intensity']

N_TRAIN_EPISODES = 2      # train on 2 episodes, test on the remaining 1 (unseen)
RUN_FULL_ROTATION = True  # if True, tries every combination of train/test episode split


def find_episode_files(episodes_dir):
    """
    Find real episode CSVs regardless of exact naming style:
    episode1.csv, episode_1.csv, Episode1.csv, EPISODE_1.csv, etc.
    Raises a clear error (no silent synthetic fallback) if none are found.
    """
    if not os.path.isdir(episodes_dir):
        # Also allow files sitting directly in the current working directory
        candidates = glob.glob("episode*[0-9].csv") + glob.glob("Episode*[0-9].csv") + glob.glob("EPISODE*[0-9].csv")
    else:
        candidates = glob.glob(os.path.join(episodes_dir, "*.csv"))

    pattern = re.compile(r"episode[_\s]*([0-9]+)", re.IGNORECASE)
    matched = []
    for path in candidates:
        fname = os.path.basename(path)
        m = pattern.search(fname)
        if m:
            matched.append((int(m.group(1)), path))

    if not matched:
        raise FileNotFoundError(
            f"No real episode CSV files found in '{episodes_dir}/' (or current directory).\n"
            f"Expected files named like episode1.csv, episode2.csv, episode3.csv "
            f"(underscore/case optional).\n"
            f"In Colab: upload your CSVs via the Files panel on the left, either into a folder "
            f"named '{episodes_dir}' or directly into /content/, then re-run this cell."
        )

    matched.sort(key=lambda x: x[0])
    return [p for _, p in matched]
# ============================================================
# END CONFIG
# ============================================================


def parse_memory_to_mb(val):
    if pd.isna(val):
        return 0.0
    try:
        first_part = str(val).split('/')[0].strip().lower()
        if 'gb' in first_part:
            return float(first_part.replace('gb', '')) * 1024
        elif 'mb' in first_part:
            return float(first_part.replace('mb', ''))
        elif 'kb' in first_part:
            return float(first_part.replace('kb', '')) / 1024
        elif 'b' in first_part:
            return float(first_part.replace('b', '')) / (1024 * 1024)
        return float(first_part)
    except Exception:
        return 0.0


def load_and_engineer_episode(path, episode_id):
    """Load one episode CSV and compute features WITHIN that episode only."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if TIMESTAMP_COL not in df.columns:
        raise KeyError(
            f"Timestamp column '{TIMESTAMP_COL}' not found in {path}. "
            f"Available columns: {df.columns.tolist()}"
        )
    df['_ts'] = pd.to_datetime(df[TIMESTAMP_COL], errors='coerce')
    df = df.dropna(subset=['_ts']).sort_values('_ts').reset_index(drop=True)

    if 'memory_usage' in df.columns:
        df['memory_usage'] = df['memory_usage'].apply(parse_memory_to_mb)

    for col in ['cpu_percent', 'memory_percent', 'memory_usage', 'Requests/s', 'Failures/s']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    dt = df['_ts'].diff().dt.total_seconds().replace(0, np.nan)
    dt = dt.fillna(dt.median() if dt.notna().any() else 1.0)

    df['cpu_velocity'] = (df['cpu_percent'].diff().fillna(0) / dt).fillna(0)
    df['memory_velocity'] = (df['memory_usage'].diff().fillna(0) / dt).fillna(0)
    df['traffic_velocity'] = (df['Requests/s'].diff().fillna(0) / dt).fillna(0)
    df['network_to_cpu_ratio'] = df['Requests/s'] / (df['cpu_percent'] + 1e-5)
    df['failure_rate_intensity'] = df['Failures/s'] / (df['Requests/s'] + 1e-5)

    df['episode_id'] = episode_id
    return df


def create_forecast_sequences(df, s1_cols, s2_cols, target_col, time_steps, horizon):
    """Build (current-snapshot, past-window) -> future-label training examples."""
    S1 = df[s1_cols].values
    S2 = df[s2_cols].values
    labels = df[target_col].values

    X_s1, X_s2, y = [], [], []
    for i in range(time_steps, len(df) - horizon):
        X_s1.append(S1[i])
        X_s2.append(S2[i - time_steps:i])
        y.append(labels[i + horizon])
    return np.array(X_s1), np.array(X_s2), np.array(y)


def build_dataset(episode_paths, le):
    """Load all episodes and return a dict episode_id -> (X_s1, X_s2, y)."""
    per_episode = {}
    for path in sorted(episode_paths):
        ep_id = os.path.splitext(os.path.basename(path))[0]
        df = load_and_engineer_episode(path, ep_id)
        df['target_encoded'] = le.transform(df[LABEL_COL].astype(str))
        X_s1, X_s2, y = create_forecast_sequences(
            df, STAGE1_FEATURES, STAGE2_FEATURES, 'target_encoded', TIME_STEPS, FORECAST_HORIZON
        )
        per_episode[ep_id] = (X_s1, X_s2, y)
    return per_episode


def print_confusion_matrix(cm, class_names, title="Confusion matrix"):
    """Pretty-print a confusion matrix with row = true label, col = predicted label."""
    print(f"  {title} (rows=true, cols=predicted):")
    col_w = max(len(c) for c in class_names) + 2
    header = " " * (col_w + 2) + "".join(f"{c:>{col_w}}" for c in class_names)
    print(header)
    for i, row_label in enumerate(class_names):
        row_str = f"  {row_label:>{col_w}} " + "".join(f"{cm[i, j]:>{col_w}}" for j in range(len(class_names)))
        print(row_str)


def train_and_evaluate(train_ids, test_ids, per_episode, class_names):
    """Train cascade on train_ids' episode(s), evaluate on each unseen test episode."""
    normal_matches = np.where(np.char.lower(class_names.astype(str)) == 'normal')[0]
    if len(normal_matches) == 0:
        raise ValueError(
            f"Could not find a 'normal' class among your labels: {list(class_names)}. "
            f"Check the values in your '{LABEL_COL}' column."
        )
    normal_idx = normal_matches[0]

    X_train_s1 = np.concatenate([per_episode[e][0] for e in train_ids], axis=0)
    X_train_s2 = np.concatenate([per_episode[e][1] for e in train_ids], axis=0)
    y_train = np.concatenate([per_episode[e][2] for e in train_ids], axis=0)

    scaler_s1 = RobustScaler().fit(X_train_s1)
    X_train_s1_scaled = scaler_s1.transform(X_train_s1)

    scaler_s2 = RobustScaler()
    n, steps, feats = X_train_s2.shape
    X_train_s2_scaled = scaler_s2.fit_transform(X_train_s2.reshape(-1, feats)).reshape(n, steps, feats)

    y_train_cat = to_categorical(y_train, num_classes=len(class_names))

    stage1_model = SVC(kernel='linear', probability=True, random_state=SEED)
    stage1_model.fit(X_train_s1_scaled, y_train)

    lstm_model = Sequential([
        LSTM(64, input_shape=(TIME_STEPS, feats), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(class_names), activation='softmax')
    ])
    lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train_s2_scaled, y_train_cat, epochs=12, batch_size=32, verbose=0)

    results = {}
    for ep_id in test_ids:
        X_test_s1, X_test_s2, y_test = per_episode[ep_id]
        if len(y_test) == 0:
            continue
        X_test_s1_scaled = scaler_s1.transform(X_test_s1)
        n_t = X_test_s2.shape[0]
        X_test_s2_scaled = scaler_s2.transform(X_test_s2.reshape(-1, feats)).reshape(n_t, steps, feats)

        s1_preds = stage1_model.predict(X_test_s1_scaled)
        s1_probs = stage1_model.predict_proba(X_test_s1_scaled)
        lstm_preds = np.argmax(lstm_model.predict(X_test_s2_scaled, verbose=0), axis=1)

        # Gatekeeper logic: if Stage 1 is highly confident it's "normal", trust it
        # (cheap + fast). Otherwise defer to the Stage 2 LSTM forecast.
        final_preds = np.where(
            (s1_preds == normal_idx) & (s1_probs[:, normal_idx] > CONFIDENCE_THRESHOLD),
            s1_preds, lstm_preds
        )
        acc = accuracy_score(y_test, final_preds)
        report = classification_report(y_test, final_preds, labels=np.arange(len(class_names)),
                                        target_names=class_names, zero_division=0, output_dict=True)
        cm = confusion_matrix(y_test, final_preds, labels=np.arange(len(class_names)))
        results[ep_id] = {'accuracy': acc, 'report': report, 'confusion_matrix': cm}

    return results


def main():
    episode_paths = find_episode_files(EPISODES_DIR)
    print(f"Found {len(episode_paths)} episodes: {[os.path.basename(p) for p in episode_paths]}")

    all_labels = pd.concat([pd.read_csv(p, usecols=[LABEL_COL]) for p in episode_paths])[LABEL_COL].astype(str)
    le = LabelEncoder().fit(all_labels)
    class_names = le.classes_
    print(f"Classes: {list(class_names)}")

    per_episode = build_dataset(episode_paths, le)
    episode_ids = sorted(per_episode.keys())

    if RUN_FULL_ROTATION and len(episode_ids) >= N_TRAIN_EPISODES + 1:
        print(f"\n=== ROTATING OVER ALL {N_TRAIN_EPISODES}-train / "
              f"{len(episode_ids) - N_TRAIN_EPISODES}-test COMBINATIONS ===")
        all_accuracies = []
        summed_cm = np.zeros((len(class_names), len(class_names)), dtype=int)
        for train_combo in itertools.combinations(episode_ids, N_TRAIN_EPISODES):
            test_combo = [e for e in episode_ids if e not in train_combo]
            print(f"\n--- Train: {train_combo} | Test (unseen): {test_combo} ---")
            results = train_and_evaluate(train_combo, test_combo, per_episode, class_names)
            for ep_id, r in results.items():
                print(f"  Forecast accuracy on unseen episode '{ep_id}': {r['accuracy']:.4f}")
                print_confusion_matrix(r['confusion_matrix'], class_names,
                                        title=f"Confusion matrix for '{ep_id}'")
                all_accuracies.append(r['accuracy'])
                summed_cm += r['confusion_matrix']

        print(f"\n{'='*50}")
        print(f"Mean forecast accuracy across ALL rotations (unseen episodes only): "
              f"{np.mean(all_accuracies):.4f} (+/- {np.std(all_accuracies):.4f})")
        print(f"{'='*50}")
        print_confusion_matrix(summed_cm, class_names,
                                title="Summed confusion matrix across ALL rotations")
    else:
        train_ids = episode_ids[:N_TRAIN_EPISODES]
        test_ids = episode_ids[N_TRAIN_EPISODES:]
        print(f"\nTrain episode(s): {train_ids}")
        print(f"Test episodes (unseen): {test_ids}")
        results = train_and_evaluate(train_ids, test_ids, per_episode, class_names)
        for ep_id, r in results.items():
            print(f"\n=== Forecast results on unseen episode '{ep_id}' ===")
            print(f"Accuracy: {r['accuracy']:.4f}")
            for cls in class_names:
                m = r['report'][cls]
                print(f"  {cls:10s} precision={m['precision']:.2f} recall={m['recall']:.2f} f1={m['f1-score']:.2f}")
            print_confusion_matrix(r['confusion_matrix'], class_names,
                                    title=f"Confusion matrix for '{ep_id}'")


if __name__ == '__main__':
    main()