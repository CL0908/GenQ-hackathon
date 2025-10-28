# ============================================================
# QRC-driven Credit Risk and FPE Forecasting
# ============================================================
#
# This script keeps the quantum reservoir computing (QRC) core
# but repurposes the workflow for portfolio credit risk.
# It forecasts portfolio probability of default (PD) and
# exposure-at-default (EAD) across a horizon, then derives
# Expected Exposure (EE), Future Potential Exposure (FPE),
# and Expected Loss (EL) diagnostics.
#
# If a portfolio dataset is not present locally, a synthetic
# timeseries is generated so the pipeline can run end-to-end.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from qutip import basis, expect, sesolve, tensor, qeye, sigmax, sigmaz
from itertools import combinations
from tensorflow.keras import layers, models, callbacks, optimizers  # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------
# 0. Configuration
# ------------------------------------------------------------
DATA_PATH = Path("credit_portfolio_timeseries.csv")
MODEL_CHECKPOINT_PATH = Path("best_qrc_credit_model.keras")
PFE_PERCENTILE = 95

# Sliding window parameters
INPUT_WINDOW = 21
FORECAST_HORIZON = 7

# Feature configuration
FEATURE_COLS = [
    "EAD",
    "Utilization",
    "Macro_Distress",
    "Sentiment_Score",
    "Portfolio_LGD",
]
TARGET_COLS = ["Portfolio_PD", "EAD"]
EMBEDDING_FEATURE_WEIGHTS = {
    "EAD": 0.55,
    "Utilization": 0.2,
    "Macro_Distress": 0.15,
    "Sentiment_Score": 0.1,
}


# ------------------------------------------------------------
# Synthetic data utility (used when dataset is missing)
# ------------------------------------------------------------
def generate_synthetic_credit_data(path: Path, periods: int = 760) -> pd.DataFrame:
    """Generate a stylised credit portfolio timeseries for demo purposes."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=periods, freq="D")
    seasonal = np.linspace(0, 6 * np.pi, periods)

    macro_cycle = 0.5 * np.sin(seasonal) + 0.2 * np.sin(0.33 * seasonal)
    macro_shock = 0.15 * rng.standard_normal(periods)
    macro_distress = np.tanh(macro_cycle + macro_shock)

    sentiment = -0.7 * macro_distress + 0.25 * rng.standard_normal(periods)
    sentiment = np.clip(sentiment, -3.0, 3.0)

    trend = 220 + np.cumsum(0.6 + 0.35 * rng.standard_normal(periods))
    cyclical = 22 * np.sin(0.4 * seasonal) + 12 * np.sin(0.9 * seasonal + np.pi / 6)
    idiosyncratic = 18 * rng.standard_normal(periods)
    ead = trend + cyclical + idiosyncratic + 35 * macro_distress
    ead = np.maximum(ead, 110)

    utilization = 0.6 + 0.12 * macro_distress + 0.05 * rng.standard_normal(periods)
    utilization = np.clip(utilization, 0.35, 0.92)

    pd_level = 0.018 + 0.015 * np.maximum(macro_distress, 0) + 0.006 * rng.standard_normal(periods)
    pd_level += 0.004 * (1 - np.tanh(sentiment))
    pd_level = np.clip(pd_level, 0.002, 0.12)

    lgd = 0.4 + 0.1 * np.maximum(macro_distress, 0) + 0.05 * rng.standard_normal(periods)
    lgd = np.clip(lgd, 0.25, 0.75)

    df = pd.DataFrame(
        {
            "Date": dates,
            "EAD": ead,
            "Utilization": utilization,
            "Macro_Distress": macro_distress,
            "Sentiment_Score": sentiment,
            "Portfolio_PD": pd_level,
            "Portfolio_LGD": lgd,
        }
    )
    df.to_csv(path, index=False)
    print(f"Synthetic credit dataset generated at {path.resolve()}")
    return df


def load_credit_portfolio(path: Path) -> pd.DataFrame:
    """Load portfolio dataset or generate a synthetic one if missing."""
    if not path.exists():
        print(f"{path} not found. Generating synthetic dataset for demonstration.")
        df = generate_synthetic_credit_data(path)
    else:
        df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    required_cols = set(FEATURE_COLS + TARGET_COLS)
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(sorted(missing_cols))}"
        )
    return df


# ------------------------------------------------------------
# 1. Data preparation helpers
# ------------------------------------------------------------
def create_multi_step_supervised(
    features: np.ndarray,
    targets: np.ndarray,
    input_window: int,
    forecast_horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows for multi-step forecasting."""
    X, Y = [], []
    n_time = features.shape[0]
    for t in range(n_time - input_window - forecast_horizon + 1):
        X.append(features[t : t + input_window])
        Y.append(targets[t + input_window : t + input_window + forecast_horizon])
    return np.array(X), np.array(Y)


# ------------------------------------------------------------
# 2. Quantum reservoir computing core
# ------------------------------------------------------------
def make_qrc_params(atom_number: int = 6):
    return {
        "atom_number": atom_number,
        "encoding_scale": 5.0,
        "rabi_frequency": 2 * np.pi,
        "total_time": 2.0,
        "time_steps": 8,
        "readouts": "ZZ",
    }


def build_hamiltonian_global(x: np.ndarray, params, time_idx: int):
    N = params["atom_number"]
    s = params["encoding_scale"]
    Ω = params["rabi_frequency"]

    window_pos = int(time_idx * len(x) / params["time_steps"])
    window_pos = min(window_pos, len(x) - 1)
    Δ_global = -x[window_pos] * s

    sx = [tensor([sigmax() if j == i else qeye(2) for j in range(N)]) for i in range(N)]
    sz = [tensor([sigmaz() if j == i else qeye(2) for j in range(N)]) for i in range(N)]

    H = sum(Ω * sx[i] / 2 for i in range(N))
    H += Δ_global * sum(sz[i] / 2 for i in range(N))
    return H


def evolve_and_embed_global(x_window: np.ndarray, params) -> np.ndarray:
    from qutip import Options

    N = params["atom_number"]
    T = params["total_time"]
    steps = params["time_steps"]

    psi0 = tensor([basis(2, 0) for _ in range(N)])
    times = np.linspace(0, T, steps)

    embedding = []
    current_state = psi0
    opts = Options(nsteps=100000, atol=1e-8, rtol=1e-6)

    for step_idx in range(steps):
        H_t = build_hamiltonian_global(x_window, params, step_idx)

        if step_idx < steps - 1:
            dt = times[step_idx + 1] - times[step_idx]
            result = sesolve(H_t, current_state, [0, dt], options=opts)
            current_state = result.states[-1]

        for i in range(N):
            z_op = tensor([sigmaz() if j == i else qeye(2) for j in range(N)])
            embedding.append(expect(z_op, current_state))

        if params["readouts"] == "ZZ":
            for i, j in combinations(range(N), 2):
                zi_op = tensor([sigmaz() if k == i else qeye(2) for k in range(N)])
                zj_op = tensor([sigmaz() if k == j else qeye(2) for k in range(N)])
                zz_op = zi_op * zj_op
                embedding.append(expect(zz_op, current_state))

    return np.array(embedding, dtype=np.float32)


def project_to_window(x_feature_vec: np.ndarray, window_size: int) -> np.ndarray:
    x = np.asarray(x_feature_vec).reshape(-1)
    if len(x) < window_size:
        return np.pad(x, (0, window_size - len(x)), mode="edge")
    if len(x) > window_size:
        indices = np.linspace(0, len(x) - 1, window_size)
        return np.interp(indices, np.arange(len(x)), x)
    return x


def build_signal_aggregator(
    feature_names: Iterable[str],
    scalers: Dict[str, MinMaxScaler],
    feature_weights: Dict[str, float],
):
    feature_names = list(feature_names)
    feature_to_idx = {name: feature_names.index(name) for name in feature_names}
    missing = set(feature_weights).difference(feature_to_idx)
    if missing:
        raise KeyError(f"Feature weights reference unknown columns: {', '.join(sorted(missing))}")

    weights = np.array([feature_weights[name] for name in feature_weights], dtype=np.float64)
    weights = weights / weights.sum()

    def aggregate(window: np.ndarray) -> np.ndarray:
        combined = np.zeros(window.shape[0], dtype=np.float64)
        for weight, feature_name in zip(weights, feature_weights):
            idx = feature_to_idx[feature_name]
            scaled = scalers[feature_name].transform(window[:, idx].reshape(-1, 1)).reshape(-1)
            combined += weight * scaled
        return combined

    return aggregate


def build_qrc_embeddings_from_windows(
    X_windows: np.ndarray,
    params,
    scalers: Dict[str, MinMaxScaler],
    feature_names: Iterable[str],
    feature_weights: Dict[str, float],
    max_samples: int | None = None,
) -> np.ndarray:
    """Build QRC embeddings from sliding windows using weighted features."""
    from tqdm import tqdm

    aggregate_window = build_signal_aggregator(feature_names, scalers, feature_weights)

    n_samples_total = len(X_windows)
    n_samples = n_samples_total if max_samples is None else min(n_samples_total, max_samples)

    sample0 = X_windows[0]
    x_proj0 = project_to_window(aggregate_window(sample0), sample0.shape[0])
    embedding_dim = evolve_and_embed_global(x_proj0, params).size
    embeddings = np.zeros((n_samples, embedding_dim), dtype=np.float32)

    print(f"Building QRC embeddings: samples={n_samples}, embedding_dim={embedding_dim}")
    for i in tqdm(range(n_samples), desc="QRC embeddings", unit="sample"):
        window = X_windows[i]
        aggregated_signal = aggregate_window(window)
        x_proj = project_to_window(aggregated_signal, window.shape[0])
        embeddings[i, :] = evolve_and_embed_global(x_proj, params)
    return embeddings


# ------------------------------------------------------------
# 3. Neural model
# ------------------------------------------------------------
def build_qrc_model(input_dim: int, output_dim: int) -> models.Model:
    """Dense MLP on top of QRC embeddings."""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.LayerNormalization()(inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(output_dim)(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="huber",
        metrics=["mae"],
    )
    return model


def flatten_targets_for_training(
    Y: np.ndarray,
    target_names: Iterable[str],
    scalers: Dict[str, MinMaxScaler],
) -> np.ndarray:
    """Scale and flatten multi-target horizon matrices for model fitting."""
    flattened_segments = []
    target_names = list(target_names)
    for idx, name in enumerate(target_names):
        scaler = scalers[name]
        scaled = scaler.transform(Y[:, :, idx])
        flattened_segments.append(scaled.reshape(len(Y), -1))
    return np.concatenate(flattened_segments, axis=1)


def decode_predictions(
    flat: np.ndarray,
    target_names: Iterable[str],
    scalers: Dict[str, MinMaxScaler],
    horizon: int,
) -> Dict[str, np.ndarray]:
    """Inverse-transform flat predictions back to per-target horizon matrices."""
    decoded: Dict[str, np.ndarray] = {}
    offset = 0
    for name in target_names:
        segment = flat[:, offset : offset + horizon]
        decoded[name] = scalers[name].inverse_transform(segment)
        offset += horizon
    return decoded


# ------------------------------------------------------------
# 4. Load data
# ------------------------------------------------------------
df = load_credit_portfolio(DATA_PATH)
print(f"Loaded portfolio dataset with {len(df)} rows spanning {df['Date'].min().date()} - {df['Date'].max().date()}")

feature_matrix = df[FEATURE_COLS].to_numpy(dtype=np.float64)
target_matrix = df[TARGET_COLS].to_numpy(dtype=np.float64)

split_loc = int(len(df) * 0.8)
split_loc = max(split_loc, INPUT_WINDOW + FORECAST_HORIZON + 5)
train_mask = np.zeros(len(df), dtype=bool)
train_mask[:split_loc] = True
test_mask = ~train_mask

train_indices = np.where(train_mask)[0]
test_indices = np.where(test_mask)[0]
if len(test_indices) == 0:
    raise ValueError("Test set is empty; extend the dataset or adjust split.")

train_start_idx = train_indices[0]
train_end_idx = train_indices[-1] + 1
test_start_idx = test_indices[0]
test_end_idx = test_indices[-1] + 1

train_features = feature_matrix[train_start_idx:train_end_idx]
train_targets = target_matrix[train_start_idx:train_end_idx]
test_features = feature_matrix[test_start_idx:test_end_idx]
test_targets = target_matrix[test_start_idx:test_end_idx]

if train_features.shape[0] < INPUT_WINDOW + FORECAST_HORIZON:
    raise ValueError("Training window is too short for the specified input/horizon.")
if test_features.shape[0] < INPUT_WINDOW + FORECAST_HORIZON:
    raise ValueError("Test window is too short for the specified input/horizon.")

X_train, Y_train = create_multi_step_supervised(train_features, train_targets, INPUT_WINDOW, FORECAST_HORIZON)
X_test, Y_test = create_multi_step_supervised(test_features, test_targets, INPUT_WINDOW, FORECAST_HORIZON)

print(f"Train windows: {X_train.shape}, Test windows: {X_test.shape}")


# ------------------------------------------------------------
# 5. Fit feature scalers and build QRC embeddings
# ------------------------------------------------------------
feature_scalers: Dict[str, MinMaxScaler] = {}
for feature_name in EMBEDDING_FEATURE_WEIGHTS:
    idx = FEATURE_COLS.index(feature_name)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_train[:, :, idx].reshape(-1, 1))
    feature_scalers[feature_name] = scaler

params = make_qrc_params(atom_number=6)
QRC_train = build_qrc_embeddings_from_windows(X_train, params, feature_scalers, FEATURE_COLS, EMBEDDING_FEATURE_WEIGHTS)
QRC_test = build_qrc_embeddings_from_windows(X_test, params, feature_scalers, FEATURE_COLS, EMBEDDING_FEATURE_WEIGHTS)


# ------------------------------------------------------------
# 6. Prepare targets
# ------------------------------------------------------------
target_scalers: Dict[str, MinMaxScaler] = {}
for idx, target_name in enumerate(TARGET_COLS):
    scaler = MinMaxScaler()
    scaler.fit(Y_train[:, :, idx])
    target_scalers[target_name] = scaler

Y_train_scaled = flatten_targets_for_training(Y_train, TARGET_COLS, target_scalers)


# ------------------------------------------------------------
# 7. Train model
# ------------------------------------------------------------
model = build_qrc_model(QRC_train.shape[1], Y_train_scaled.shape[1])
model.summary()

ckpt = callbacks.ModelCheckpoint(
    MODEL_CHECKPOINT_PATH,
    monitor="val_mae",
    save_best_only=True,
    verbose=1,
)
es = callbacks.EarlyStopping(
    monitor="val_mae",
    patience=35,
    restore_best_weights=True,
    verbose=1,
)

history = model.fit(
    QRC_train,
    Y_train_scaled,
    validation_split=0.2,
    epochs=350,
    batch_size=32,
    callbacks=[ckpt, es],
    verbose=1,
)


# ------------------------------------------------------------
# 8. Predict and inverse transform
# ------------------------------------------------------------
Y_pred_scaled = model.predict(QRC_test)
decoded_pred = decode_predictions(Y_pred_scaled, TARGET_COLS, target_scalers, FORECAST_HORIZON)

Y_pred_components = [decoded_pred[name][:, :, None] for name in TARGET_COLS]
Y_pred = np.concatenate(Y_pred_components, axis=2)

Y_true = Y_test  # already shaped (samples, horizon, targets)

pd_true = Y_true[:, :, TARGET_COLS.index("Portfolio_PD")]
pd_pred = Y_pred[:, :, TARGET_COLS.index("Portfolio_PD")]
ead_true = Y_true[:, :, TARGET_COLS.index("EAD")]
ead_pred = Y_pred[:, :, TARGET_COLS.index("EAD")]


# ------------------------------------------------------------
# 9. Evaluation metrics
# ------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    print(f"{label} -> MAE: {mae:.6f} | RMSE: {rmse:.6f} | R²: {r2:.4f}")


print("\n==== Portfolio-level Metrics ====")
compute_metrics(pd_true, pd_pred, "PD forecast")
compute_metrics(ead_true, ead_pred, "EAD forecast")


# ------------------------------------------------------------
# 10. Credit exposure diagnostics (EE / FPE / Expected Loss)
# ------------------------------------------------------------
current_ead = X_test[:, -1, FEATURE_COLS.index("EAD")]
current_lgd = X_test[:, -1, FEATURE_COLS.index("Portfolio_LGD")]

diagnostics = []
for day in range(FORECAST_HORIZON):
    residuals = ead_true[:, day] - ead_pred[:, day]
    centered_residuals = residuals - residuals.mean()

    scenario_matrix = ead_pred[:, day][:, None] + centered_residuals[None, :]
    scenario_matrix = np.maximum(scenario_matrix, 0.0)

    scenario_flat = scenario_matrix.flatten()
    ee = scenario_flat.mean()
    fpe = np.percentile(scenario_flat, PFE_PERCENTILE)

    lgd_matrix = current_lgd[:, None]
    expected_loss = np.mean(pd_pred[:, day][:, None] * lgd_matrix * scenario_matrix)
    diagnostics.append(
        {
            "day": day + 1,
            "mean_pd": pd_pred[:, day].mean(),
            "ee": ee,
            "fpe": fpe,
            "expected_loss": expected_loss,
        }
    )

diag_df = pd.DataFrame(diagnostics)
print("\n==== Exposure diagnostics (test set) ====")
print(f"{'day':<5} {'mean_pd':>10} {'ee':>12} {'fpe':>12} {'expected_loss':>16}")
for row in diag_df.itertuples(index=False):
    print(
        f"{row.day:<5} {row.mean_pd:>10.4f} {row.ee:>12.2f} {row.fpe:>12.2f} {row.expected_loss:>16.2f}"
    )


# ------------------------------------------------------------
# 11. Future forecast beyond dataset
# ------------------------------------------------------------
aggregate_signal_func = build_signal_aggregator(FEATURE_COLS, feature_scalers, EMBEDDING_FEATURE_WEIGHTS)

last_window = feature_matrix[-INPUT_WINDOW:]
last_signal = aggregate_signal_func(last_window)
last_embedding = evolve_and_embed_global(project_to_window(last_signal, INPUT_WINDOW), params)
future_pred_scaled = model.predict(last_embedding.reshape(1, -1))
decoded_future = decode_predictions(future_pred_scaled, TARGET_COLS, target_scalers, FORECAST_HORIZON)

last_date = df["Date"].iloc[-1]
future_rows = []
last_ead = last_window[-1, FEATURE_COLS.index("EAD")]
last_lgd = last_window[-1, FEATURE_COLS.index("Portfolio_LGD")]

pd_future = decoded_future["Portfolio_PD"][0]
ead_future = decoded_future["EAD"][0]

for day in range(FORECAST_HORIZON):
    forecast_date = last_date + pd.Timedelta(days=day + 1)
    exposure_delta = ead_future[day] - last_ead
    projected_exposure = last_ead + max(exposure_delta, 0.0)
    expected_loss = pd_future[day] * last_lgd * projected_exposure
    future_rows.append(
        {
            "forecast_date": forecast_date,
            "pred_pd": pd_future[day],
            "pred_ead": ead_future[day],
            "expected_loss": expected_loss,
        }
    )

future_df = pd.DataFrame(future_rows)
print("\n==== Forward projection (next horizon) ====")
print(f"{'date':<12} {'pred_pd':>10} {'pred_ead':>12} {'expected_loss':>16}")
for row in future_df.itertuples(index=False):
    print(
        f"{row.forecast_date.strftime('%Y-%m-%d'):<12} "
        f"{row.pred_pd:>10.4f} {row.pred_ead:>12.2f} {row.expected_loss:>16.2f}"
    )

print("\nTraining, evaluation, and forecasting complete ✅")
