import numpy as np

# Load original .npz file
original_path = "/Users/justin/PycharmProjects/quattro-transformer-ilqr/examples/quadrotor/dec3_dmodel128_nhead4_ff512_drop0.1_epoch200_promptlen1_616.2k/tf_model_normalizer.npz"
data = np.load(original_path, allow_pickle=True)

# Manually specify the config as a plain dictionary (edit these based on your model!)
model_config = {
    "target_len": 49,
    "prompt_len": 1,
    "state_dim": 12,
    "control_dim": 52,
    "model_sel": "decoder",
    "d_model": 128,
    "nhead": 4,
    "num_encoder_layers": 0,        # not used in decoder-only
    "num_decoder_layers": 3,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "max_seq_len": 110,
    "num_epochs": 200,
    "quant_mode": "float16"
}

target_len = 49
prompt_len = 1
state_dim = 12
control_dim = 52
d_model = 128
nhead = 4
num_decoder_layers = 3
dim_feedforward = 512
dropout = 0.1
max_seq_len = 110
num_epochs = 200
quant_mode = "float16"

# Prepare new dict with all arrays and the plain dict
new_data = {
    "x_mean": data["x_mean"],
    "x_std": data["x_std"],
    "u_mean": data["u_mean"],
    "u_std": data["u_std"],
    "target_len": target_len,
    "prompt_len": prompt_len,
    "state_dim": state_dim,
    "control_dim": control_dim,
    "d_model": d_model,
    "nhead": nhead,
    "num_decoder_layers": num_decoder_layers,
    "dim_feedforward": dim_feedforward,
    "dropout": dropout,
    "max_seq_len": max_seq_len,
    "num_epochs": num_epochs,
    "quant_mode": quant_mode
}

# Save to new .npz file (compatible with updated load())
new_path = "tf_model_normalizer_updated.npz"
np.savez(new_path, **new_data)

print(f"Updated .npz file saved to: {new_path}")
