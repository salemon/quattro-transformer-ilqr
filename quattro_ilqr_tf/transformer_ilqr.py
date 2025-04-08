import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Import transformer components from the transformer_model module.
# from transformer_model import TransformerPredictor, DataNormalizer
from .transformer_model import TransformerPredictor, DataNormalizer

## @package transformer_ilqr
#  @brief This module defines the TransformerILQR class, which handles dataset creation,
#         training, saving, loading, and inference for the decoder-only transformer model.

## @class TransformerILQR
#  @brief Class for training and using a decoder-only transformer for iLQR problems.
class TransformerILQR:
    ## @brief Constructor for TransformerILQR.
    #  @param state_dim Dimension of the state trajectory.
    #  @param control_dim Dimension of the combined control data (e.g., 52).
    #  @param prompt_len Length of the prompt (last steps from the control sequence).
    #  @param d_model Embedding dimension.
    #  @param nhead Number of attention heads.
    #  @param num_decoder_layers Number of decoder layers.
    #  @param dim_feedforward Dimension of the feedforward network.
    #  @param dropout Dropout probability.
    #  @param max_seq_len Maximum sequence length.
    #  @param quant_mode Quantization mode ("none", "float16", or "int8").
    def __init__(self, state_dim, control_dim, prompt_len=10, d_model=64, nhead=8,
                 num_decoder_layers=3, dim_feedforward=128, dropout=0.1,
                 max_seq_len=100, quant_mode="none"):
        self.state_dim = state_dim
        self.control_dim = control_dim  # Should match the flattened dimension (e.g., 52)
        self.prompt_len = prompt_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.quant_mode = quant_mode
        self.target_len = None  # Will be determined based on sequence length

        # Set computation device.
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = None
        self.normalizer = DataNormalizer()

    ## @brief Creates a dataset from the provided dataframe.
    #  @param df A pandas DataFrame containing the keys:
    #         - 'x_seq': state trajectory, shape (T, state_dim)
    #         - 'k_seq': control sequence of shape (T, 4)
    #         - 'K_seq': sequence of matrices of shape (T, 4, 12)
    #  @return A tuple (x_data, kK_data) where:
    #          - x_data is a numpy array of shape (N, T, state_dim)
    #          - kK_data is a numpy array of shape (N, T, 52)
    def _create_dataset(self, df):
        x_list, kK_list = [], []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing dataset"):
            x_seq = np.array(row['x_seq'], dtype=np.float32)  # (T, state_dim)
            k_seq = np.array(row['k_seq'], dtype=np.float32)    # (T, 4)
            K_seq = np.array(row['K_seq'], dtype=np.float32)    # (T, 4, 12)
            
            # Expand k_seq to shape (T, 4, 1)
            k_seq = np.expand_dims(k_seq, axis=-1)
            
            # Stack k_seq and K_seq along the last dimension to obtain (T, 4, 13)
            kK_seq = np.concatenate([k_seq, K_seq], axis=-1)
            
            # Flatten the last two dimensions to obtain (T, 52)
            kK_seq = kK_seq.reshape(kK_seq.shape[0], -1)
            
            if kK_seq.shape[0] <= self.prompt_len:
                continue  # Skip sequences that are too short.
            x_list.append(x_seq)
            kK_list.append(kK_seq)
        x_data = np.array(x_list)    # Shape: (N, T, state_dim)
        kK_data = np.array(kK_list)  # Shape: (N, T, 52)
        return x_data, kK_data

    ## @brief Trains the transformer model using the provided training (and optional test) data.
    #  @param df A pandas DataFrame with training data.
    #  @param test_df (Optional) A pandas DataFrame with test data.
    #  @param num_epochs Number of training epochs.
    #  @param batch_size Batch size.
    #  @param learning_rate Learning rate for the optimizer.
    #  @param patience Number of epochs to wait for improvement before early stopping.
    #  @return Self instance with the trained model.
    def fit(self, df, test_df=None, num_epochs=50, batch_size=16, learning_rate=1e-3, patience=5):
        # Build training arrays.
        x_data, kK_data = self._create_dataset(df)
        N, T, _ = x_data.shape
        self.target_len = T - self.prompt_len

        # Fit normalizer and transform data.
        self.normalizer.fit(x_data, kK_data)
        x_data_norm = self.normalizer.transform_x(x_data)
        kK_data_norm = self.normalizer.transform_u(kK_data)

        x_input = x_data_norm  # (N, T, state_dim)
        u_prompt = kK_data_norm[:, -self.prompt_len:, :]  # (N, prompt_len, control_dim)
        u_target = kK_data_norm[:, :T - self.prompt_len, :]  # (N, target_len, control_dim)

        # Create dataset and DataLoader.
        x_tensor = torch.tensor(x_input)
        u_prompt_tensor = torch.tensor(u_prompt)
        u_target_tensor = torch.tensor(u_target)
        dataset = TensorDataset(x_tensor, u_prompt_tensor, u_target_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize the transformer model.
        self.model = TransformerPredictor(
            state_dim=self.state_dim,
            control_dim=self.control_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len,
            target_len=self.target_len,
            prompt_len=self.prompt_len
        )
        self.model.to(self.device)
        print(self.model)
        self.num_epochs = num_epochs

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        self.model.train()

        best_loss = float('inf')
        no_improvement = 0
        train_loss_history = []
        test_loss_history = []

        for epoch in range(num_epochs):
            total_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            for x_batch, u_prompt_batch, u_target_batch in pbar:
                x_batch = x_batch.to(self.device).float()
                u_prompt_batch = u_prompt_batch.to(self.device).float()
                u_target_batch = u_target_batch.to(self.device).float()

                optimizer.zero_grad()
                pred = self.model(x_batch, u_prompt_batch)
                loss = criterion(pred, u_target_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x_batch.size(0)
                pbar.set_postfix(loss=f"{loss.item():.6f}")
            total_loss /= len(dataset)
            train_loss_history.append(total_loss)

            # Optional evaluation on test data.
            if test_df is not None:
                x_test, kK_test = self._create_dataset(test_df)
                x_test_norm = self.normalizer.transform_x(x_test)
                kK_test_norm = self.normalizer.transform_u(kK_test)
                T_test = x_test_norm.shape[1]
                u_prompt_test = kK_test_norm[:, -self.prompt_len:, :]
                u_target_test = kK_test_norm[:, :T_test - self.prompt_len, :]

                x_test_tensor = torch.tensor(x_test_norm).to(self.device).float()
                u_prompt_test_tensor = torch.tensor(u_prompt_test).to(self.device).float()
                u_target_test_tensor = torch.tensor(u_target_test).to(self.device).float()

                self.model.eval()
                with torch.no_grad():
                    test_pred = self.model(x_test_tensor, u_prompt_test_tensor)
                    test_loss = criterion(test_pred, u_target_test_tensor).item()
                self.model.train()
                test_loss_history.append(test_loss)
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss:.6f}, Test Loss: {test_loss:.6f}")

                if test_loss < best_loss:
                    best_loss = test_loss
                    no_improvement = 0
                    best_model_state = self.model.state_dict()
                else:
                    no_improvement += 1

                if no_improvement >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    self.model.load_state_dict(best_model_state)
                    break
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss:.6f}")

        # Store loss history for external plotting if desired.
        self.train_loss_history = train_loss_history
        if test_df is not None:
            self.test_loss_history = test_loss_history

        return self

    ## @brief Saves the trained model and normalization parameters.
    #  @param base_name Base name for the saved model files.
    def save(self, base_name):
        total_params = sum(p.numel() for p in self.model.parameters())
        if total_params >= 1e6:
            param_str = f"{total_params/1e6:.1f}M"
        elif total_params >= 1e3:
            param_str = f"{total_params/1e3:.1f}k"
        else:
            param_str = str(total_params)

        hyperparams = f"decoder_dec{self.num_decoder_layers}_dmodel{self.d_model}_nhead{self.nhead}_ff{self.dim_feedforward}_drop{self.dropout}_epoch{self.num_epochs}_promptlen{self.prompt_len}"
        if self.quant_mode == "float16":
            self.model = self.model.half()
        elif self.quant_mode == "int8":
            self.model = torch.quantization.quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        dir_name = f"{timestamp}_{base_name}_{hyperparams}_{param_str}"
        os.makedirs(dir_name, exist_ok=True)
        model_file = os.path.join(dir_name, "tf_model.pt")
        norm_file = os.path.join(dir_name, "tf_model_normalizer.npz")

        torch.save(self.model.state_dict(), model_file)
        norm_params = {
            'x_mean': self.normalizer.x_mean,
            'x_std': self.normalizer.x_std,
            'u_mean': self.normalizer.u_mean,
            'u_std': self.normalizer.u_std,
            'target_len': self.target_len,
            'prompt_len': self.prompt_len,
            'state_dim': self.state_dim,
            'control_dim': self.control_dim,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_decoder_layers': self.num_decoder_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            'max_seq_len': self.max_seq_len,
            'num_epochs': self.num_epochs,
            'quant_mode': self.quant_mode
        }
        np.savez(norm_file, **norm_params)
        print(f"Model saved to {model_file}\nNormalizer and hyperparameters saved to {norm_file}.")

    ## @brief Loads a trained model and normalization parameters from a directory.
    #  @param model_path Path to the directory containing the model and normalizer files.
    #  @return Self instance with the loaded model.
    def load(self, model_path):
        model_file = os.path.join(model_path, "tf_model.pt")
        norm_file = os.path.join(model_path, "tf_model_normalizer.npz")
        data = np.load(norm_file, allow_pickle=True)

        # Restore normalizer parameters.
        self.normalizer.x_mean = data['x_mean']
        self.normalizer.x_std = data['x_std']
        self.normalizer.u_mean = data['u_mean']
        self.normalizer.u_std = data['u_std']

        self.target_len = int(data['target_len'])
        self.prompt_len = int(data['prompt_len'])
        self.state_dim = int(data['state_dim'])
        self.control_dim = int(data['control_dim'])
        self.d_model = int(data['d_model'])
        self.nhead = int(data['nhead'])
        self.num_decoder_layers = int(data['num_decoder_layers'])
        self.dim_feedforward = int(data['dim_feedforward'])
        self.dropout = float(data['dropout'])
        self.max_seq_len = int(data['max_seq_len'])
        self.num_epochs = int(data['num_epochs'])
        self.quant_mode = str(data['quant_mode'])

        self.model = TransformerPredictor(
            state_dim=self.state_dim,
            control_dim=self.control_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len,
            target_len=self.target_len,
            prompt_len=self.prompt_len
        )

        if self.quant_mode == "int8":
            self.model = torch.quantization.quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        if self.quant_mode == "float16":
            self.model = self.model.half()
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from directory: {model_path}")
        return self

    ## @brief Uses the trained model to predict missing controls for a new sample.
    #  @param x_seq A numpy array of shape (T, state_dim) representing the state trajectory.
    #  @param kK_seq A numpy array of shape (T, control_dim) representing the combined control data.
    #         The prompt is taken as the last prompt_len entries.
    #  @return A numpy array of shape (target_len, control_dim) with the predicted controls.
    def predict(self, x_seq, kK_seq):
        x_norm = self.normalizer.transform_x(x_seq)
        kK_norm = self.normalizer.transform_u(kK_seq)
        u_prompt_norm = kK_norm[-self.prompt_len:, :]
        x_tensor = torch.tensor(x_norm.astype(np.float32)).unsqueeze(0).to(self.device)
        u_prompt_tensor = torch.tensor(u_prompt_norm.astype(np.float32)).unsqueeze(0).to(self.device)
        if self.quant_mode == "float16":
            x_tensor = x_tensor.half()
            u_prompt_tensor = u_prompt_tensor.half()
        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(x_tensor, u_prompt_tensor)
        pred_norm = pred_norm.squeeze(0).cpu().numpy()
        pred = self.normalizer.inverse_transform_u(pred_norm)
        return pred
