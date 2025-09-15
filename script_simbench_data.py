# -*- coding: utf-8 -*-
"""script_simbench_data.ipynb

"""

!pip install pandapower
!pip install simbench
import simbench as sb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import pandapower as pp
import pandapower.topology as top
import pandapower.plotting as plot
!pip install tqdm
import tqdm as tqdm
import pandas as pd

# --- Part 1: Load Data Processing ---

load_data = pd.read_csv("Load.csv", delimiter=';')
load_profile = pd.read_csv("LoadProfile.csv", delimiter=';')

bus_profile_mapping = load_data[["node", "profile", "pLoad", "qLoad"]].copy()

# Melt the load_profile DataFrame to create a 'profile' and 'value' column
melted_df = load_profile.melt(id_vars=['time'], var_name='profile', value_name='value')

# Convert time column to datetime format immediately for load_profile
melted_df['time'] = pd.to_datetime(melted_df['time'], format="%d.%m.%Y %H:%M")

# Extract the bus name and load type (pload or qload) from the profile column
# Example: 'H0A_pload' -> 'bus_name'='H0A', 'load_type'='pload'
melted_df[['bus_name', 'load_type']] = melted_df['profile'].str.rsplit('_', n=1, expand=True)

# Aggregate duplicate entries by SUMMING, assuming components contribute additively
# This handles cases where multiple profile columns in LoadProfile.csv (e.g., H0A_pload_v1, H0A_pload_v2)
# might map to the same (bus_name, load_type) after rsplit, and their values should be summed.
melted_df = melted_df.groupby(['time', 'bus_name', 'load_type'])['value'].sum().reset_index()

# Pivot the table to have separate columns for pload and qload values
pivoted_df = melted_df.pivot(index=['time', 'bus_name'], columns='load_type', values='value').reset_index()

# Rename the columns to descriptive names (p_profile_value for active, q_profile_value for reactive)
final_df = pivoted_df.rename(columns={'pload': 'p_profile_value', 'qload': 'q_profile_value'})

# Ensure all necessary columns are present and ordered clearly
final_df = final_df[['time', 'bus_name', 'p_profile_value', 'q_profile_value']]

# Merge final_df (time series profiles) with bus_profile_mapping (base loads)
# This merge is now correct and will create duplicate (time, node) rows if bus_profile_mapping
# contains multiple profiles for the same node (which is intended, like H0A and H0B for LV2.101 Bus 23).
merged_df = final_df.merge(bus_profile_mapping, left_on="bus_name", right_on="profile", how="left")

# Calculate dynamic p_load and q_load using the profile values and base loads
merged_df["p_load"] = merged_df["p_profile_value"] * merged_df["pLoad"]
merged_df["q_load"] = merged_df["q_profile_value"] * merged_df["qLoad"]

# Select essential columns for the initial result before final aggregation
final_result_pre_agg = merged_df[["time", "node", "p_load", "q_load"]].copy()

# This crucial step sums up all load components (e.g., from H0A and H0B profiles) for the same node
# at the same timestep, resulting in a single row per (time, node) combination.
final_result = final_result_pre_agg.groupby(['time', 'node']).sum().reset_index()

# --- Part 2: RES (PV) Data Processing ---

# Load RES data
res_df = pd.read_csv("RES.csv", sep=';')
res_profile_df = pd.read_csv("RESProfile.csv", sep=';')

# Convert time columns to datetime for RES profiles
res_profile_df['time'] = pd.to_datetime(res_profile_df['time'], format="%d.%m.%Y %H:%M")

# Ensure final_result's time column is also datetime before merging with PV data
final_result['time'] = pd.to_datetime(final_result['time'])

# >>>>>>>>>>>>>>>>>> MORE PV (optional) <<<<<<<<<<<<<<<<<<
# Target capacity penetration = total PV nameplate / feeder peak load
#P_peak = final_result.groupby('time')['p_load'].sum().max()        # MW
#pv_cap_now = res_df['pRES'].sum()                                  # MW (sum of nameplates)
#target_pen = 0.60                                                  # e.g., 60% penetration

#scale = (target_pen * P_peak) / pv_cap_now
#print(f"Scaling PV capacities by ×{scale:.3f} to reach {target_pen*100:.0f}% of peak load.")
#res_df['pRES'] *= scale
# >>>>>>>>>>>>>>>>>  <<<<<<<<<<<<<<<<<<<
# Prepare RES data with node, profile, pRES for iteration
res_df_clean = res_df[['node', 'profile', 'pRES']].copy()

# Expand each RES unit into a time series based on its profile
pv_time_series = []

for _, row in res_df_clean.iterrows():
    node = row['node']
    profile = row['profile']
    capacity = row['pRES']

    if profile in res_profile_df.columns:
        temp_df = res_profile_df[['time', profile]].copy()
        temp_df['node'] = node
        # Calculate PV output by multiplying profile value with capacity
        temp_df['pv_output'] = temp_df[profile] * capacity
        pv_time_series.append(temp_df[['time', 'node', 'pv_output']])

# Combine all individual PV time series into a single DataFrame
pv_node_df = pd.concat(pv_time_series, ignore_index=True)

# Aggregate PV output by time and node (summing output from multiple PV units at the same node)
pv_node_aggregated = pv_node_df.groupby(['time', 'node'])['pv_output'].sum().reset_index()

# Merge the aggregated PV data with the final_result (loads)
final_result_with_pv = final_result.merge(pv_node_aggregated, on=['time', 'node'], how='left')

# Fill any NaN pv_output values (for nodes without PV) with 0
final_result_with_pv['pv_output'] = final_result_with_pv['pv_output'].fillna(0)

# Display a preview and save the final DataFrame (optional)
print(final_result_with_pv.tail())
final_result_with_pv.to_csv("final_result_with_pv_per_node.csv", index=False)

print("final_result_with_pv DataFrame created and saved successfully!")

sb_code1 = "1-LV-rural2--0-sw"  # rural MV grid of scenario 0 with full switchs
net = sb.get_simbench_net(sb_code1)


import pandapower as pp
import pandapower.networks as pn

sb_code1 = "1-LV-rural2--0-sw"  # rural MV grid of scenario 0 with full switchs
net = sb.get_simbench_net(sb_code1)
slack_bus = 62
# 2. Remove existing external grid(s)
net.ext_grid.drop(net.ext_grid.index, inplace=True)

# 3. Create a new upstream "grid bus"
grid_bus = pp.create_bus(net, vn_kv=net.bus.vn_kv.at[slack_bus], name="Grid Equivalent")

# 4. Add external grid (slack) at this upstream bus
pp.create_ext_grid(net, bus=grid_bus, vm_pu=1.02, va_degree=0.0)

# 5. Define Thevenin impedance (from S_sc and R/X)
S_sc = 5.0  # MVA
RX = 7.0
S_base = net.sn_mva

Z_mag = S_base / S_sc
X_pu = Z_mag / np.sqrt(RX**2 + 1)
R_pu = RX * X_pu

pp.create_impedance(net,
    from_bus=grid_bus, to_bus=slack_bus,
    rft_pu=R_pu, xft_pu=X_pu,
    rtf_pu=R_pu, xtf_pu=X_pu,
    sn_mva=S_sc,
    name="Source Impedance 5 MVA R/X=7"
)

# 6. Run load flow
pp.runpp(net)
#
# 8. View results
print("Slack bus voltage:", net.res_bus.vm_pu.at[slack_bus])
print("Grid‑side bus voltage:", net.res_bus.vm_pu.at[grid_bus])

#final_result = pd.read_csv("final_result (4).csv")
#final_result = final_result_with_pv


###Right power flow! dynamic

final_result = final_result_with_pv

# Convert time column to datetime format
final_result["time"] = pd.to_datetime(final_result["time"], format='%Y-%m-%d %H:%M:%S')
results = []

all_buses = net.bus.index.tolist()

for timestep in tqdm(final_result["time"].unique(), desc="Running Power Flow"):

    timestep_data = final_result[final_result["time"] == timestep]

    timestep_data = timestep_data.sort_values(by=["time", "node"])

    for index, row in timestep_data.iterrows():
        #bus = row['node']  # Get bus number from 'node' column
        bus_name_to_index = net.bus.reset_index().set_index("name")["index"].to_dict()
        bus_idx = bus_name_to_index.get(row["node"])

        if bus_idx is not None:
          net.load.loc[net.load["bus"] == bus_idx, "p_mw"] = row["p_load"]
          net.load.loc[net.load["bus"] == bus_idx, "q_mvar"] = row["q_load"]

          if "pv_output" in row and not pd.isna(row["pv_output"]):
            net.sgen.loc[net.sgen["bus"] == bus_idx, "p_mw"] = row["pv_output"]


    # Run Power Flow
    try:
        pp.runpp(net, enforce_q_lims=True)
        #pp.runpp(net, algorithm="bfsw", enforce_q_lims=False, init="flat",tolerance_mva=1e-2)
        #pp.runpp(net, algorithm="lindist", enforce_q_lims=True,  init="flat",tolerance_mva=1e-3)
    except pp.powerflow.LoadflowNotConverged:
        print(f"⚠️ Power flow did not converge at {timestep}")
        continue  # Skip storing results if power flow fails

    # Extract power flow results (after simulation)
    bus_results = net.res_bus[["vm_pu", "va_degree", "p_mw", "q_mvar"]].copy()
    bus_results["bus_name"] = net.bus["name"].values
    bus_results["timestep"] = timestep
    results.append(bus_results)


# Save final power flow results
all_bus_results = pd.concat(results, ignore_index=True)
all_bus_results.to_csv("newsimbench_powerflow_results.csv", index=False)

print("Power flow simulation complete! Results saved.")





