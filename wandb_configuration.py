# --- IMPORTS ---
import os
import pprint
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, mixed_precision, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from wandb.integration.keras import WandbCallback
import wandb
import sys
from scipy.stats import truncnorm
from tensorflow.keras.layers import Reshape

# --- DATA PREPROCESSING (Run once; stays outside train_model) ---
file_path = "/dss/dsshome1/0A/ge56ron2/Masters_Thesis/results/results_gpu/pf_results_timeseries.csv"
all_merged = pd.read_csv(file_path)

all_merged.rename(columns={"time": "timestep"}, inplace=True)
# Filtering
all_merged = all_merged[all_merged["bus_name"] != "MV1.101 Bus 8"].copy()
all_merged = all_merged[all_merged["bus_name"] != "LV2.101 Bus 19"].copy()

# Aggregate substation signals (map LV2.101 Bus 19 p/q onto all rows by timestep)
grid_p = all_merged.loc[all_merged["bus_name"] == "Grid Equivalent", ["timestep", "p_mw"]].set_index("timestep")
all_merged["sum_p_mw"] = all_merged["timestep"].map(grid_p["p_mw"])
grid_q = all_merged.loc[all_merged["bus_name"] == "Grid Equivalent", ["timestep", "q_mvar"]].set_index("timestep")
all_merged["sum_q_mvar"] = all_merged["timestep"].map(grid_q["q_mvar"])

# Remove the substation row and sort by bus order
all_merged = all_merged[all_merged["bus_name"] != "Grid Equivalent"].copy()
all_merged["bus_number"] = all_merged["bus_name"].str.extract(r"Bus (\d+)").astype(int)
all_merged = all_merged.sort_values(by=["timestep", "bus_number"])

# Rename columns
all_merged = all_merged.rename(columns={
    "bus_name": "bus",
    "bus_number": "bus_name",
    "sum_p_mw": "net_power_demand_kw",
    "sum_q_mvar": "sum_q_kvar",
    "p_mw": "p_kw",
    "q_mvar": "q_kvar",
})

# Unit conversions
all_merged["sum_q_kvar"] = all_merged["sum_q_kvar"] * 1000
all_merged["net_power_demand_kw"] = all_merged["net_power_demand_kw"] * 1000
all_merged["q_kvar"] = all_merged["q_kvar"] * 1000
all_merged["p_kw"] = all_merged["p_kw"] * 1000

# Simple dynamics features (optional)
#all_merged["net_p_diff"] = all_merged["net_power_demand_kw"].diff().fillna(0)
#all_merged["net_q_diff"] = all_merged["sum_q_kvar"].diff().fillna(0)
#all_merged["net_p_std3"] = all_merged["net_power_demand_kw"].rolling(window=3).std().fillna(0)
#all_merged["net_q_std3"] = all_merged["sum_q_kvar"].rolling(window=3).std().fillna(0)
#all_merged["net_p_min5"] = all_merged["net_power_demand_kw"].rolling(window=3).min().fillna(0)

# Targets (vm, va) per timestep x bus
df = all_merged.copy()
vm_df = df.pivot(index="timestep", columns="bus_name", values="vm_pu")
va_df = df.pivot(index="timestep", columns="bus_name", values="va_degree")
targets_3d = np.stack([vm_df.values, va_df.values], axis=-1)  # shape: T x nbuses x 2

# Time features (optional)
df["timestep"] = pd.to_datetime(df["timestep"], format="%Y-%m-%d %H:%M:%S")
df["hour_seq"] = (df["timestep"] - df["timestep"].min()).dt.total_seconds() // 3600

# Extract basic time components
df["hour"] = df["timestep"].dt.hour  # 0–23
df["day_of_week"] = df["timestep"].dt.dayofweek + 1  # 1–7


def get_iso_week_number(dt):
    isocal = dt.isocalendar()
    week_num = isocal.week
    if isocal.year > dt.year:
        week_num = 53
    return week_num


# Add sin/cos encodings
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)


X_feat = df.drop_duplicates("timestep").sort_values("timestep")
input_cols= ["net_power_demand_kw",	"sum_q_kvar", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]

X_feat = X_feat[input_cols]

def add_bounded_relative_noise(df_in, cols, rel_error=0.01, min_amp=1e-4, seed=42, suffix="_noisy"):
    rng = np.random.default_rng(seed)
    Z = truncnorm(a=-2, b=2)
    noisy_df = df_in.copy()
    for col in cols:
        x = noisy_df[col].to_numpy(dtype=float)
        sigma = (rel_error * np.abs(x) / 2.0)
        mask = (np.abs(x) >= min_amp) & np.isfinite(x)
        noise = np.zeros_like(x, dtype=float)
        z_samples = Z.rvs(size=mask.sum(), random_state=rng)
        noise[mask] = sigma[mask] * z_samples
        noisy_df[f"{col}{suffix}"] = x + noise
    return noisy_df

noise_cols = ["net_power_demand_kw", "sum_q_kvar"]

T = len(X_feat)
train_size = int(T * 0.7)
val_size = int(T * 0.15)

X_train = X_feat.iloc[:train_size].copy()
X_val   = X_feat.iloc[train_size : train_size + val_size].copy()
X_test  = X_feat.iloc[train_size + val_size :].copy()

y_train_raw = targets_3d[:train_size]
y_val_raw   = targets_3d[train_size : train_size + val_size]
y_test_raw  = targets_3d[train_size + val_size :]

X_train_noisy = add_bounded_relative_noise(X_train, noise_cols, rel_error=0.01, seed=49)
X_val_noisy   = add_bounded_relative_noise(X_val,   noise_cols, rel_error=0.01, seed=42)
X_test_noisy  = add_bounded_relative_noise(X_test,  noise_cols, rel_error=0.01, seed=47)

X_train_raw = X_train_noisy
X_val_raw   = X_val_noisy
X_test_raw  = X_test_noisy

X_train_raw.drop(columns=["net_power_demand_kw"], inplace=True)
X_train_raw.drop(columns=["sum_q_kvar"], inplace=True)
X_train_raw.rename(columns={"net_power_demand_kw_noisy": "net_power_demand_kw"}, inplace=True)
X_train_raw.rename(columns={"sum_q_kvar_noisy": "sum_q_kvar"}, inplace=True)
   
X_val_raw.drop(columns=["net_power_demand_kw"], inplace=True)
X_val_raw.drop(columns=["sum_q_kvar"], inplace=True)
X_val_raw.rename(columns={"net_power_demand_kw_noisy": "net_power_demand_kw"}, inplace=True)
X_val_raw.rename(columns={"sum_q_kvar_noisy": "sum_q_kvar"}, inplace=True)

X_test_raw.drop(columns=["net_power_demand_kw"], inplace=True)
X_test_raw.drop(columns=["sum_q_kvar"], inplace=True)
X_test_raw.rename(columns={"net_power_demand_kw_noisy": "net_power_demand_kw"}, inplace=True)
X_test_raw.rename(columns={"sum_q_kvar_noisy": "sum_q_kvar"}, inplace=True)

print(X_train_raw)

# Scalers
scaler_p = StandardScaler()
scaler_q = StandardScaler()
scaler_va = StandardScaler()
scaler_vm_pu = StandardScaler()

scaler_p.fit(X_train_raw[["net_power_demand_kw"]])
scaler_q.fit(X_train_raw[["sum_q_kvar"]])

vm_train = y_train_raw[:, :, 0].reshape(-1, 1)
va_train = y_train_raw[:, :, 1].reshape(-1, 1)
scaler_vm_pu.fit(vm_train)
scaler_va.fit(va_train)

# Apply scaling (avoid chained assignment)
X_train_raw.loc[:, "net_power_demand_kw"] = scaler_p.transform(X_train_raw[["net_power_demand_kw"]])
X_val_raw.loc[:,   "net_power_demand_kw"] = scaler_p.transform(X_val_raw[["net_power_demand_kw"]])
X_test_raw.loc[:,  "net_power_demand_kw"] = scaler_p.transform(X_test_raw[["net_power_demand_kw"]])

X_train_raw.loc[:, "sum_q_kvar"] = scaler_q.transform(X_train_raw[["sum_q_kvar"]])
X_val_raw.loc[:,   "sum_q_kvar"] = scaler_q.transform(X_val_raw[["sum_q_kvar"]])
X_test_raw.loc[:,  "sum_q_kvar"] = scaler_q.transform(X_test_raw[["sum_q_kvar"]])

# Scale targets
y_train_raw[:, :, 0] = scaler_vm_pu.transform(vm_train).reshape(y_train_raw.shape[0], -1)
y_val_raw[:, :, 0]   = scaler_vm_pu.transform(y_val_raw[:, :, 0].reshape(-1, 1)).reshape(y_val_raw.shape[0], -1)
y_test_raw[:, :, 0]  = scaler_vm_pu.transform(y_test_raw[:, :, 0].reshape(-1, 1)).reshape(y_test_raw.shape[0], -1)

y_train_raw[:, :, 1] = scaler_va.transform(va_train).reshape(y_train_raw.shape[0], -1)
y_val_raw[:, :, 1]   = scaler_va.transform(y_val_raw[:, :, 1].reshape(-1, 1)).reshape(y_val_raw.shape[0], -1)
y_test_raw[:, :, 1]  = scaler_va.transform(y_test_raw[:, :, 1].reshape(-1, 1)).reshape(y_test_raw.shape[0], -1)

# Sequence creation

lookback = 96
def create_sequences_seq2seq(X_df, y_3d, lookback, B):
    # X_df: DataFrame of features indexed by time (shape T×F)
    # y_3d: numpy array (T, B, 2)
    X_seq, y_seq = [], []
    X = X_df.values
    T_total = len(X_df)
    for t in range(lookback, T_total):
        X_seq.append(X[t - lookback:t, :])              # (lookback, F)
        y_seq.append(y_3d[t - lookback:t, :, :])        # (lookback, B, 2)
    return np.array(X_seq), np.array(y_seq)

# rebuild sequences
B = 95
X_train_seq, y_train_seq = create_sequences_seq2seq(X_train_raw, y_train_raw, lookback, B)
X_val_seq,   y_val_seq   = create_sequences_seq2seq(X_val_raw,   y_val_raw,   lookback, B)
X_test_seq,  y_test_seq  = create_sequences_seq2seq(X_test_raw,  y_test_raw,  lookback, B)

print(X_train_seq.shape)  # (N, 96, F)
print(y_train_seq.shape)  # (N, 96, 95, 2)


# Sequence creation
#def create_sequences(X, y, lookback):
#    X_seq, y_seq = [], []
#    Xv = X.values
#    for i in range(lookback, len(X)):
#        X_seq.append(Xv[i - lookback : i, :])
#        y_seq.append(y[i])
#    return np.array(X_seq), np.array(y_seq)

#lookback = 96
#X_train_seq, y_train_seq = create_sequences(X_train_raw, y_train_raw, lookback)
#X_val_seq,   y_val_seq   = create_sequences(X_val_raw,   y_val_raw,   lookback)
#X_test_seq,  y_test_seq  = create_sequences(X_test_raw,  y_test_raw,  lookback)

# --- MODEL + LOSS ---
@tf.keras.utils.register_keras_serializable()
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))
def rmse_last_step(y_true, y_pred):
    # shapes: (N, T, B, 2)
    y_true_last = y_true[:, -1, :, :]
    y_pred_last = y_pred[:, -1, :, :]
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred_last - y_true_last)))
@tf.keras.utils.register_keras_serializable()
def rmse_seq(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))
    
import tensorflow as tf
from tensorflow import keras

@keras.saving.register_keras_serializable()
class ExtremaAwareLoss(keras.losses.Loss):
    """
    Base RMSE + penalty on errors at local maxima and minima (detected
    by 2nd derivative along time). Works with y of shape:
      - (N, T, B, 2)  or
      - (N, T, B*2)
    where last channel order is [vm, va].

    Args:
        vm_weight: weight for VM in base/extrema terms
        va_weight: weight for VA in base/extrema terms
        extrema_weight: multiplier for extrema penalty
        tau: curvature threshold (0 = any extremum)
        B: number of buses
        reduction: keras reduction (default SUM_OVER_BATCH_SIZE)
    """
    def __init__(self,
                 vm_weight=1.0,
                 va_weight=1.0,
                 extrema_weight=1.0,
                 tau=0.0,
                 B=95,
                 reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name="ExtremaAwareLoss"):
        super().__init__(reduction=reduction, name=name)
        self.vm_weight = float(vm_weight)
        self.va_weight = float(va_weight)
        self.extrema_weight = float(extrema_weight)
        self.tau = float(tau)
        self.B = int(B)
        # 2nd-derivative kernel along time (height)
        k = tf.constant([[1.0], [-2.0], [1.0]], dtype=tf.float32)  # (3,1)
        self.kernel = tf.reshape(k, [3, 1, 1, 1])                  # (kh,kw,inC,outC)

    def _ensure_NTBC2(self, y):
        """Return y shaped (N, T, B, 2) from (N,T,B,2) or (N,T,B*2)."""
        y = tf.cast(y, tf.float32)
        r = tf.rank(y)
        def reshape_from_flat():
            shp = tf.shape(y)                   # [N, T, B*2]
            N, T, FB2 = shp[0], shp[1], shp[2]
            # Expect FB2 == B*2; use provided B:
            return tf.reshape(y, [N, T, self.B, 2])
        return tf.cond(tf.equal(r, 3), reshape_from_flat, lambda: y)

    def _g2_time(self, x_NTB):
        """Second derivative along time via Conv2D. x: (N,T,B) -> (N,T,B)."""
        x4 = tf.expand_dims(x_NTB, axis=-1)           # (N,T,B,1)  NHWC
        g2 = tf.nn.conv2d(x4, self.kernel, strides=[1,1,1,1], padding="SAME")  # (N,T,B,1)
        return tf.squeeze(g2, axis=-1)                # (N,T,B)

    def _rmse(self, e):
        return tf.sqrt(tf.reduce_mean(tf.square(e)) + 1e-12)

    def _rmse_masked(self, e_sq, mask):
        # sqrt( sum(e^2 * mask) / (sum(mask)+eps) )
        num = tf.reduce_sum(e_sq * mask)
        den = tf.reduce_sum(mask) + 1e-12
        return tf.sqrt(num / den)

    def call(self, y_true, y_pred):
        y_true = self._ensure_NTBC2(y_true)           # (N,T,B,2)
        y_pred = self._ensure_NTBC2(y_pred)           # (N,T,B,2)

        vm_t, va_t = y_true[..., 0], y_true[..., 1]   # (N,T,B)
        vm_p, va_p = y_pred[..., 0], y_pred[..., 1]

        # ---- base RMSE (global)
        base = self.vm_weight * self._rmse(vm_p - vm_t) \
             + self.va_weight * self._rmse(va_p - va_t)

        # ---- extrema masks via second derivative over time
        g2_vm = self._g2_time(vm_t)                   # (N,T,B)
        g2_va = self._g2_time(va_t)

        thr = tf.constant(self.tau, dtype=tf.float32)
        min_mask_vm = tf.cast(g2_vm >  thr, tf.float32)
        max_mask_vm = tf.cast(g2_vm < -thr, tf.float32)
        min_mask_va = tf.cast(g2_va >  thr, tf.float32)
        max_mask_va = tf.cast(g2_va < -thr, tf.float32)

        vm_err2 = tf.square(vm_p - vm_t)
        va_err2 = tf.square(va_p - va_t)

        vm_min_pen = self._rmse_masked(vm_err2, min_mask_vm)
        vm_max_pen = self._rmse_masked(vm_err2, max_mask_vm)
        va_min_pen = self._rmse_masked(va_err2, min_mask_va)
        va_max_pen = self._rmse_masked(va_err2, max_mask_va)

        extrema_pen = self.vm_weight * (vm_min_pen + vm_max_pen) \
                    + self.va_weight * (va_min_pen + va_max_pen)

        return base + self.extrema_weight * extrema_pen

    def get_config(self):
        return {
            "vm_weight": self.vm_weight,
            "va_weight": self.va_weight,
            "extrema_weight": self.extrema_weight,
            "tau": self.tau,
            "B": self.B,
            "reduction": self.reduction,
            "name": self.name,
        }


def build_model(cfg, T, n_features, B=95):
    model = Sequential([
        LSTM(cfg.num_lstm_units, return_sequences=True, activation="tanh",
             recurrent_dropout=1e-6, input_shape=(T, n_features)),
        LSTM(cfg.num_lstm_units, return_sequences=True, activation="tanh",
             recurrent_dropout=1e-6),
        LSTM(cfg.num_lstm_units // 2, return_sequences=True, activation="tanh",
             recurrent_dropout=1e-6),
        LSTM(cfg.num_lstm_units // 2, return_sequences=True, activation="tanh",
             recurrent_dropout=1e-6),
        Dense(B * 2, activation="linear"),       # (N,T,190)
        Reshape((T, B, 2))                      # (N,T,95,2)
    ])
    return model


# -------------------------
# Train loop (wandb-agent friendly)
# -------------------------
def train_model(config=None):
    with wandb.init(config=config):
        cfg = wandb.config

        T = X_train_seq.shape[1]
        n_features = X_train_seq.shape[2]

        model = build_model(cfg, T, n_features, B=B)

        # choose optimizer
        if cfg.optimizer == "adam":
            opt = Adam(learning_rate=cfg.learning_rate)
        else:
            opt = RMSprop(learning_rate=cfg.learning_rate)

        # choose loss
        if cfg.loss_name == "extrema":
            loss_fn = ExtremaAwareLoss(
                vm_weight=cfg.vm_weight,
                va_weight=cfg.va_weight,
                extrema_weight=cfg.extrema_weight,
                tau=cfg.tau,
                B=B,
            )
        else:
            loss_fn = rmse_seq

        model.compile(
            optimizer=opt,
            loss=loss_fn,
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), rmse_seq, rmse_last_step],
        )

        model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            callbacks=[WandbCallback(save_model=False)],
            verbose=1,
        )
        
# --- SWEEP (you already have a sweep created; config below is just for reference/debug print) ---
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "num_lstm_units": {"distribution": "int_uniform", "min": 96, "max": 512},
        "optimizer": {"distribution": "categorical", "values": ["adam", "rmsprop"]},
        "learning_rate": {"distribution": "uniform", "min": 2e-4, "max": 5e-3},
        "batch_size": {"distribution": "int_uniform", "min": 64, "max": 224},
        "epochs": {"distribution": "int_uniform", "min": 15, "max": 35},
        "loss_name": {"distribution": "categorical", "values": ["extrema", "rmse"]},
        "vm_weight": {"distribution": "uniform", "min": 0.5, "max": 2.0},
        "va_weight": {"distribution": "uniform", "min": 0.5, "max": 2.0},
        "extrema_weight": {"distribution": "uniform", "min": 0.2, "max": 1.5},
        "tau": {"distribution": "uniform", "min": 0.0, "max": 0.02},
    },
}


pprint.pprint(sweep_config)

if __name__ == "__main__":
    SWEEP_ID = "xxx"
    wandb.agent(SWEEP_ID, function=train_model, count=40)
