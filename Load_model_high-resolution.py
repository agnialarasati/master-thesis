# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# !pip install matplotlib

# !pip install keras-tuner==1.4.7
# !pip install tqdm
# !pip install scikit-optimize
# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

from tensorflow.keras import layers,models

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import MultiHeadAttention, Dense, LayerNormalization, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import joblib


from skopt.space import Integer, Real, Categorical
from skopt import gp_minimize
from skopt.plots import plot_objective
from skopt.plots import plot_convergence
from tensorflow.keras.layers import Reshape
from skopt.utils import use_named_args
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
# +

from tensorflow.keras.callbacks import Callback

import tensorflow as tf, numpy as np, random
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# +
file_path = "/dss/dsshome1/0A/ge56ron2/Masters_Thesis/results/results_gpu/newsimbench_powerflow_results_decweak.csv"

all_merged = pd.read_csv(file_path)


buses = all_merged["bus_name"].unique()
print(buses)
#all_merged.rename(columns={"time": "timestep"}, inplace=True)

###FOR FIVE BUS ANALYSIS

#Scenario 1
#bus_list = ["LV2.101 Bus 11", "LV2.101 Bus 30", "LV2.101 Bus 42",
#            "LV2.101 Bus 46", "LV2.101 Bus 53"]
#Scenario 2
#bus_list = ["LV2.101 Bus 30", "LV2.101 Bus 41", "LV2.101 Bus 46", 
#           "LV2.101 Bus 50", "LV2.101 Bus 51"]

#filtered_df = all_merged[all_merged["bus_name"].isin(bus_list)].copy()
#filtered_df["bus_number"] = filtered_df["bus_name"].str.extract(r"Bus (\d+)").astype(int)
#filtered_df = filtered_df.sort_values(by=["timestep", "bus_number"], ascending=[True, True])

# +
all_merged = all_merged[all_merged["bus_name"] != "MV1.101 Bus 8"].copy()
all_merged = all_merged[all_merged["bus_name"] != "LV2.101 Bus 19"].copy()

grid_series = all_merged[all_merged["bus_name"] == "Grid Equivalent"][
    ["timestep", "p_mw"]
].set_index("timestep")
all_merged["sum_p_mw"] = all_merged["timestep"].map(grid_series["p_mw"])

grid_series = all_merged[all_merged["bus_name"] == "Grid Equivalent"][
    ["timestep", "q_mvar"]
].set_index("timestep")
all_merged["sum_q_mvar"] = all_merged["timestep"].map(grid_series["q_mvar"])

# -


all_merged = all_merged[all_merged["bus_name"] != "Grid Equivalent"].copy()
all_merged["bus_number"] = all_merged["bus_name"].str.extract(r"Bus (\d+)").astype(int)
all_merged = all_merged.sort_values(by=["timestep", "bus_number"], ascending=[True, True])

# +
all_merged.rename(columns={"bus_name": "bus"}, inplace=True)
all_merged.rename(columns={"bus_number": "bus_name"}, inplace=True)
all_merged.rename(columns={"sum_p_mw": "net_power_demand_kw"}, inplace=True)
all_merged.rename(columns={"sum_q_mvar": "sum_q_kvar"}, inplace=True)
all_merged["sum_q_kvar"] = all_merged["sum_q_kvar"] * 1000
all_merged["net_power_demand_kw"] = all_merged["net_power_demand_kw"] * 1000

all_merged.rename(columns={"p_mw": "p_kw"}, inplace=True)
all_merged.rename(columns={"q_mvar": "q_kvar"}, inplace=True)
all_merged["q_kvar"] = all_merged["q_kvar"] * 1000
all_merged["p_kw"] = all_merged["p_kw"] * 1000


###ALL BUS ANALYSIS
df = all_merged
vm_df = df.pivot(index="timestep", columns="bus_name", values="vm_pu")
va_df = df.pivot(index="timestep", columns="bus_name", values="va_degree")

###FOR FIVE BUS ANALYSIS
#all_merged = all_merged[all_merged["bus"].isin(bus_list)].copy()
#df = all_merged
#vm_df = filtered_df.pivot(index="timestep", columns="bus_name", values="vm_pu")
#va_df = filtered_df.pivot(index="timestep", columns="bus_name", values="va_degree")

targets_3d = np.stack([vm_df.values,va_df.values], axis=-1)
# shape: (T, 96, 2) vm_df.values, va_df.values, p_df.values, q_df.values


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
input_cols = ["net_power_demand_kw","sum_q_kvar", "hour_sin", "hour_cos", "dow_sin", "dow_cos"] 
X_feat = X_feat[input_cols]


from scipy.stats import truncnorm
def add_bounded_relative_noise(
    df: pd.DataFrame,
    cols,
    rel_error=0.01,     # 1% max amplitude
    min_amp=1e-4,
    seed=42,
    suffix="_noisy",
):
    """
    Adds symmetric truncated-Gaussian noise per value with bounds ±rel_error*|x|.
    Noise ~ sigma * Z, where Z ~ TruncNorm(a=-2, b=+2), sigma = rel_error*|x|/2.
    """
    rng = np.random.default_rng(seed)  # reproducible
    Z = truncnorm(a=-2, b=2)           # standard truncated normal (fixed)

    noisy_df = df.copy()
    for col in cols:
        x = noisy_df[col].to_numpy(dtype=float)
        sigma = (rel_error * np.abs(x) / 2.0)

        mask = (np.abs(x) >= min_amp) & np.isfinite(x)
        noise = np.zeros_like(x, dtype=float)

        # sample Z only for valid positions; use scipy with our numpy random bits
        z_samples = Z.rvs(size=mask.sum(), random_state=rng)
        noise[mask] = sigma[mask] * z_samples

        noisy_df[f"{col}{suffix}"] = x + noise

    meta = {
        "rel_error": rel_error,
        "min_amp": min_amp,
        "seed": seed,
        "cols": list(cols),
        "trunc_a_b": (-2, 2),
    }
    return noisy_df, meta

noise_cols = ["net_power_demand_kw", "sum_q_kvar"]


T = len(X_feat)
train_size = int(T * 0.7)
val_size   = int(T * 0.15)

X_train = X_feat.iloc[:train_size]
X_val  = X_feat.iloc[train_size:train_size+val_size]
X_test  = X_feat.iloc[train_size+val_size:]

y_train_raw = targets_3d[:train_size]
y_val_raw   = targets_3d[train_size:train_size+val_size]
y_test_raw  = targets_3d[train_size+val_size:]

X_train_noisy, meta_tr = add_bounded_relative_noise(X_train, noise_cols, rel_error=0.01, seed=49)
X_val_noisy,   meta_va = add_bounded_relative_noise(X_val,   noise_cols, rel_error=0.01, seed=42)
X_test_noisy,  meta_te = add_bounded_relative_noise(X_test,  noise_cols, rel_error=0.01, seed=47)

X_train_raw = X_train_noisy
X_val_raw  = X_val_noisy
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

# +
input_cols_scale = ["net_power_demand_kw",	"sum_q_kvar"]
input_cols_no_scale = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

output_cols_vm_pu = ["vm_pu"]
output_cols_va = ["va_degree"]

scaler_va = StandardScaler()
scaler_vm_pu = StandardScaler() #MinMaxScaler(feature_range=(0.8, 1.1))
scaler_p = StandardScaler()
scaler_q = StandardScaler()

scaler_p.fit(X_train_raw[["net_power_demand_kw"]])
scaler_q.fit(X_train_raw[["sum_q_kvar"]])



# +
## ---- Scale vm_pu (channel 0) ----
vm_train = y_train_raw[:, :, 0].reshape(-1, 1)
vm_val   = y_val_raw[:, :, 0].reshape(-1, 1)
vm_test  = y_test_raw[:, :, 0].reshape(-1, 1)


scaler_vm_pu.fit(vm_train)
vm_train_scaled = scaler_vm_pu.transform(vm_train).reshape(y_train_raw.shape[0], 95) ###CHANGE TO 5 FOR FIVE BUS ANALYSIS
vm_val_scaled   = scaler_vm_pu.transform(vm_val).reshape(y_val_raw.shape[0], 95)
vm_test_scaled  = scaler_vm_pu.transform(vm_test).reshape(y_test_raw.shape[0], 95)

y_train_raw[:, :, 0] = vm_train_scaled
y_val_raw[:, :, 0]   = vm_val_scaled
y_test_raw[:, :, 0]  = vm_test_scaled

# ---- Scale va_degree (channel 1) ----
va_train = y_train_raw[:, :, 1].reshape(-1, 1)
va_val   = y_val_raw[:, :, 1].reshape(-1, 1)
va_test  = y_test_raw[:, :, 1].reshape(-1, 1)

scaler_va.fit(va_train)
va_train_scaled = scaler_va.transform(va_train).reshape(y_train_raw.shape[0], 95)   ###CHANGE TO 5 FOR FIVE BUS ANALYSIS
va_val_scaled   = scaler_va.transform(va_val).reshape(y_val_raw.shape[0], 95)
va_test_scaled  = scaler_va.transform(va_test).reshape(y_test_raw.shape[0], 95)

y_train_raw[:, :, 1] = va_train_scaled
y_val_raw[:, :, 1]   = va_val_scaled
y_test_raw[:, :, 1]  = va_test_scaled

X_train_raw.loc[:, "net_power_demand_kw"] = scaler_p.transform(X_train_raw[["net_power_demand_kw"]])
X_val_raw.loc[:, "net_power_demand_kw"] = scaler_p.transform(X_val_raw[["net_power_demand_kw"]])
X_test_raw.loc[:, "net_power_demand_kw"] = scaler_p.transform(X_test_raw[["net_power_demand_kw"]])

X_train_raw.loc[:, "sum_q_kvar"] = scaler_q.transform(X_train_raw[["sum_q_kvar"]])
X_val_raw.loc[:, "sum_q_kvar"] = scaler_q.transform(X_val_raw[["sum_q_kvar"]])
X_test_raw.loc[:, "sum_q_kvar"] = scaler_q.transform(X_test_raw[["sum_q_kvar"]])

# +
def create_sequences(X, y, lookback):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

lookback = 60 #for 5 minutes. For 30 min, 360 lookback steps. 1 hour equals to 720 lookback steps
X_train_seq, y_train_seq = create_sequences(X_train_raw, y_train_raw, lookback)

# -

X_val_seq,   y_val_seq   = create_sequences(X_val_raw, y_val_raw, lookback)
X_test_seq,  y_test_seq  = create_sequences(X_test_raw, y_test_raw, lookback)


class TQDMProgressBar(Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.tqdm = tqdm(total=self.epochs, desc="Training Epochs", unit="epoch")

    def on_epoch_end(self, epoch, logs=None):
        self.tqdm.update(1)

    def on_train_end(self, logs=None):
        self.tqdm.close()


# === Define model ===
#from tensorflow.keras.layers import Bidirectional
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

save_dir = "/dss/dsshome1/0A/ge56ron2/Masters_Thesis"
results_dir = "/dss/dsshome1/0A/ge56ron2/Masters_Thesis/results/results_gpu"

file_path = f"{results_dir}/Model_TG_VMVA_5min_5s_feat_RMSE_seq2seq.keras" ###CHANGE MODEL HERE

@keras.saving.register_keras_serializable()
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))
@tf.keras.utils.register_keras_serializable()
def rmse_seq(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

model = load_model(
    file_path,
    custom_objects={
        #"ExtremaAwareLoss":ExtremaAwareLoss,
        "rmse": rmse,
        "rmse_seq": rmse_seq,
    }
)



###SAVE MODEL
#file_path = f"{results_dir}/lstm_model_vmva_rmse5s_vweak_5min_ssc5.keras"
#model.save(file_path)


y_pred_flat = model.predict(X_test_seq)  
y_pred_seq = y_pred_flat.reshape(-1, 95, 2)  # ###CHANGE TO y_pred_flat.reshape(-1, 5, 2) FOR FIVE BUS ANALYSIS


def inverse_transform_predictions(y_pred_seq, scalers):
    """
    Inversely transform the predicted outputs to original scale.

    Parameters:
        y_pred_seq: (samples, 96, 4) predicted outputs in scaled form
        scalers: tuple of fitted scalers (scaler_vm, scaler_va, scaler_pq)

    Returns:
        y_pred_inv: (samples, 96, 4) with original scale restored
    """
    scaler_vm_pu, scaler_va = scalers # scaler_vm_pu, scaler_va, scaler_pq

    samples, buses, _ = y_pred_seq.shape
    y_pred_inv = np.zeros_like(y_pred_seq)

    ## vm_pu (channel 0)
    vm_flat = y_pred_seq[:, :, 0].reshape(-1, 1)
    y_pred_inv[:, :, 0] = scaler_vm_pu.inverse_transform(vm_flat).reshape(samples, buses)

    ## va_degree (channel 1)
    va_flat = y_pred_seq[:, :, 1].reshape(-1, 1)
    y_pred_inv[:, :, 1] = scaler_va.inverse_transform(va_flat).reshape(samples, buses)

    return y_pred_inv




y_pred_inv = inverse_transform_predictions(y_pred_seq, scalers=(scaler_vm_pu, scaler_va)) 

y_true_inv = inverse_transform_predictions(y_test_seq, scalers=(scaler_vm_pu, scaler_va)) 



def plot_predictions_vs_true(y_pred_inv, y_true_inv, bus_index=0, variable_index=0, variable_name="vm_pu"):
    """
    Plots predicted vs. true values for a specific bus and variable.

    Parameters:
        y_pred_inv : ndarray of shape (samples, 96, 4), inverse-transformed model predictions
        y_true_seq     : ndarray of shape (samples, 96, 4), true values in original scale
        bus_index      : int, which bus to plot (0 to 95)
        variable_index : int, which variable to plot (0=vm_pu, 1=va_degree, 2=p_kW, 3=q_kVAr)
        variable_name  : str, label for the y-axis and title
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    plt.figure(figsize=(12, 4))
    plt.plot(y_true_inv[:, bus_index, variable_index], label="True", linewidth=1.5)
    plt.plot(y_pred_inv[:, bus_index, variable_index], label="Predicted", linestyle="--", linewidth=1.5)
    plt.title(f"{variable_name} Prediction for Bus {bus_index}")
    plt.xlabel("Timestep")
    plt.ylabel(variable_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f"{results_dir}/plot-{variable_name}_bus-{bus_index}_{timestamp}.png")
    plt.show()



# +


def compute_error_metrics(y_pred_inv, y_true_inv, variable_index=0):
    """
    Compute RMSE, MAE, MSE for a single variable across all buses and timesteps.
    
    Returns:
        dict with 'rmse', 'mae', 'mse'
    """
    y_true = y_true_inv[:, :, variable_index].flatten()
    y_pred = y_pred_inv[:, :, variable_index].flatten()

    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)

    V_base = 230 

    RMSE_V = rmse * V_base       
    MAE_V = mae * V_base         
    MSE_V = mse * V_base**2      
    return {"rmse": rmse, "mae": mae, "mse": mse, "rmse_v": RMSE_V, "mae_v": MAE_V, "mse_v": MSE_V}



# +

def compute_error_metrics_va(y_pred_inv, y_true_inv, variable_index=0):
    """
    Compute RMSE, MAE, MSE for a single variable across all buses and timesteps.
    
    Returns:
        dict with 'rmse', 'mae', 'mse'
    """
    y_true = y_true_inv[:, :, variable_index].flatten()
    y_pred = y_pred_inv[:, :, variable_index].flatten()

    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)

    return {"rmse": rmse, "mae": mae, "mse": mse}



def plot_all_buses_heatmap(y_pred_inv, y_true_inv, variable_index=0, variable_name="vm_pu"):
    """
    Plots a heatmap of prediction vs true error across all buses and timesteps.

    Parameters:
        y_pred_inv: (samples, 96, 4)
        y_true_seq: (samples, 96, 4)
        variable_index: which variable to plot (0–3)
        variable_name: label for plot
    """
    error = y_pred_inv[:, :, variable_index] - y_true_inv[:, :, variable_index] #y_true_seq
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    plt.figure(figsize=(12, 5))
    plt.imshow(error.T, aspect="auto", cmap="RdBu", interpolation="none")
    plt.colorbar(label="Prediction Error")
    plt.xlabel("Timestep")
    plt.ylabel("Bus Index")
    plt.title(f"Prediction Error Heatmap: {variable_name}")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/heatmap-{variable_name}_{timestamp}.png")
    plt.show()



# -

plot_predictions_vs_true(y_pred_inv, y_true_inv, bus_index=91, variable_index=0, variable_name="vm_pu")
plot_predictions_vs_true(y_pred_inv, y_true_inv, bus_index=15, variable_index=1, variable_name="va_degree")



# +
plot_all_buses_heatmap(y_pred_inv, y_true_inv, variable_index=0, variable_name="vm_pu")
plot_all_buses_heatmap(y_pred_inv, y_true_inv, variable_index=1, variable_name="va_degree")

metrics_vm = compute_error_metrics(y_pred_inv, y_true_inv, variable_index=0)
metrics_va = compute_error_metrics_va(y_pred_inv, y_true_inv, variable_index=1)

print("VM_pu  metrics:", metrics_vm)
print("VA_deg metrics:", metrics_va)




# +


def save_predictions_to_csv(y_pred_inv, y_true_inv, results_dir, filename="all_predictions.csv", bus_names=None):
    """
    Save predicted and true values for all buses and timesteps to a CSV file inside results_dir.

    Parameters:
    - y_pred_inv: (samples, buses, 4) predicted values [vm_pu, va_deg, p_kW, q_kVar]
    - y_true_inv: same shape, true values
    - results_dir: folder where the CSV will be saved
    - filename: name of the CSV file
    - bus_names: list of bus names (optional); otherwise uses indices
    """
    samples, buses, _ = y_pred_inv.shape
    records = []

    if bus_names is None:
        bus_names = [f"Bus {i}" for i in range(buses)]

    for t in range(samples):
        for b in range(buses):
            record = {
                "timestamp": t,
                "bus": bus_names[b],
                "vm_pu_true": y_true_inv[t, b, 0],
                "vm_pu_pred": y_pred_inv[t, b, 0],
                "va_deg_true": y_true_inv[t, b, 1],
                "va_deg_pred": y_pred_inv[t, b, 1],
                #"p_kW_true": y_true_inv[t, b, 0],
                #"p_kW_pred": y_pred_inv[t, b, 0],
                #"q_kVar_true": y_true_inv[t, b, 1],
                #"q_kVar_pred": y_pred_inv[t, b, 1],
            }
            records.append(record)

    df = pd.DataFrame(records)

    # Ensure results_dir exists
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    df.to_csv(path, index=False)
    print(f"✅ Saved predictions to {path}")



# -

save_predictions_to_csv(y_pred_inv, y_true_inv, results_dir, filename="pred_vmva_5min_rmse5s_ssc5.csv")

def plot_and_save_flattened_vm_pu(y_true_inv, y_pred_inv, results_dir):
    os.makedirs(results_dir, exist_ok=True)  # Make sure dir exists

    vm_true_flat = y_true_inv[:, :, 0].flatten()
    vm_pred_flat = y_pred_inv[:, :, 0].flatten()
    x = np.arange(len(vm_true_flat))

    plt.figure(figsize=(14, 5))
    plt.plot(x, vm_true_flat, label="True vm_pu", linewidth=0.5)
    plt.plot(
        x,
        vm_pred_flat,
        label="Predicted vm_pu",
        linewidth=0.5,
        alpha=0.7,
        linestyle="--",
    )
    plt.xlabel("Flattened Time Index (All Buses Mixed)")
    plt.ylabel("Voltage Magnitude [p.u.]")
    plt.title("Flattened Voltage Magnitude (vm_pu) Chronologically")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save before showing
    save_path = os.path.join(results_dir, "plot_allbus_vm_5s5min_ssc5.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.show
    plt.close()

def plot_and_save_flattened_va(y_true_inv, y_pred_inv, results_dir):
    os.makedirs(results_dir, exist_ok=True)  # Make sure dir exists

    va_true_flat = y_true_inv[:, :, 1].flatten()
    va_pred_flat = y_pred_inv[:, :, 1].flatten()
    x = np.arange(len(va_true_flat))

    plt.figure(figsize=(14, 5))
    plt.plot(x, va_true_flat, label="True va_degree", linewidth=0.5)
    plt.plot(
        x,
        va_pred_flat,
        label="Predicted va_degree",
        linewidth=0.5,
        alpha=0.7,
        linestyle="--",
    )
    plt.xlabel("Flattened Time Index (All Buses Mixed)")
    plt.ylabel("Voltage Angle [°]")
    plt.title("Flattened Voltage Angle (va_degree) Chronologically")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save before showing
    save_path = os.path.join(results_dir, "plot_allbus_va_5s5min_ssc5.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.show
    plt.close()


plot_and_save_flattened_vm_pu(y_true_inv, y_pred_inv, results_dir)
plot_and_save_flattened_va(y_true_inv, y_pred_inv, results_dir)




target_idx = 0  # 0: vm_pu, 1: va_degree, 2: p, 3: q
bus_names = [f"Bus {i}" for i in range(y_true_inv.shape[1])]  # Replace with real names if available

squared_errors = (y_true_inv[:, :, target_idx] - y_pred_inv[:, :, target_idx]) ** 2
rmse_per_bus = np.sqrt(np.mean(squared_errors, axis=0))

df_rmse = pd.DataFrame({
    "bus": bus_names,
    "rmse": rmse_per_bus
}).sort_values(by="rmse", ascending=False).reset_index(drop=True)

# Display top 10 worst-predicted and best-predicted buses
print("Best & Worst Bus: vm_pu (MAE)")
print(df_rmse.head(10))  # Highest RMSE (worst)
print(df_rmse.tail(10))  # Lowest RMSE (best)


target_idx = 1  # 0: vm_pu, 1: va_degree, 2: p, 3: q
bus_names = [f"Bus {i}" for i in range(y_true_inv.shape[1])]  # Replace with real names if available


squared_errors = (y_true_inv[:, :, target_idx] - y_pred_inv[:, :, target_idx]) ** 2
rmse_per_bus = np.sqrt(np.mean(squared_errors, axis=0))

df_rmse = pd.DataFrame({
    "bus": bus_names,
    "rmse": rmse_per_bus
}).sort_values(by="rmse", ascending=False).reset_index(drop=True)

# Display top 10 worst-predicted and best-predicted buses
print("Best & Worst Bus: va_deg (RMSE)")
print(df_rmse.head(10))  # Highest RMSE (worst)
print(df_rmse.tail(10))  # Lowest RMSE (best)

target_idx = 0  # 0: vm_pu, 1: va_degree, 2: p, 3: q
bus_names = [f"Bus {i}" for i in range(y_true_inv.shape[1])]  # Replace with real names if available

# Compute per-bus MAE
abs_errors = np.abs(y_true_inv[:, :, target_idx] - y_pred_inv[:, :, target_idx])
mae_per_bus = np.mean(abs_errors, axis=0)


df_mae = pd.DataFrame({
    "bus": bus_names,
    "mae": mae_per_bus
}).sort_values(by="mae", ascending=False).reset_index(drop=True)

# Display top 10 worst-predicted buses
print("Best & Worst Bus: vm_pu")
print(df_mae.head(10))
print(df_mae.tail(10))
#save_path = "/dss/dsshome1/0A/ge56ron2/Masters_Thesis/results/results_gpu/mae_bus_p.csv"
#df_mae.to_csv(save_path, index=False)

# +
target_idx = 1  # 0: vm_pu, 1: va_degree, 2: p, 3: q
bus_names = [f"Bus {i}" for i in range(y_true_inv.shape[1])]  # Replace with real names if available

# Compute per-bus MAE
abs_errors = np.abs(y_true_inv[:, :, target_idx] - y_pred_inv[:, :, target_idx])
mae_per_bus = np.mean(abs_errors, axis=0)
best_bus = df_mae.iloc[-1,-1]
worst_bus = df_mae.iloc[0,1]


df_mae_ang = pd.DataFrame({
    "bus": bus_names,
    "mae": mae_per_bus
}).sort_values(by="mae", ascending=False).reset_index(drop=True)

# Display top 10 worst-predicted buses
print("Best & Worst Bus: va_degree")
print(df_mae_ang.head(10))
print(df_mae_ang.tail(10))
#save_path = "/dss/dsshome1/0A/ge56ron2/Masters_Thesis/results/results_gpu/mae_bus_q.csv"
#df_mae_ang.to_csv(save_path, index=False)