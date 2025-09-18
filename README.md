# master-thesis
These are the codes used for Agnia Larasati's Master's Thesis: Data-Driven State Estimation of Low-Voltage Distribution Network Based on Limited Smart Meter Measurements. The goal is to estimate voltage states from non-voltage measurements observed at the substation. This thesis uses SimBench LV rural2--0 data. The codes are executed with Slurm in the LRZ AI systems. myscript.sh is the example Slurm executor used to submit jobs.

1. Load Model files are used to load and evaluate saved models only.
- Load_model_TG.py
- Load_model_SG.py
- Load_model_high-resolution.py

2. Train Model files are used to train and evaluate a model with given parameters.
- Model_TG_24h_15min_feat_EAL2_seq2seq.py for the typical grid with adjustable LSTM parameters.
- Model_SG_24h_15min_feat_EAL2_seq2seq.py for the strong grid with adjustable LSTM parameters and number of buses (scenario is given).
- Model_TG_24h_15min_feat_EAL2_seq2seq_sc_5bus.py for the typical grid with adjustable LSTM parameters, only for five bus analysis. 
- Model_TG_5min_5s_feat_RMSE_seq2one.py for the typical grid with higher resolution and seq2one configuration. Adjustable for five bus analysis.

3. script_simbench_data.py for preprocessing SimBench data.
4. wandb_configuration.py for searching the best LSTM configuration in wandb.

