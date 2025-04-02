import torch
import torch.nn as nn

## @package transformer_model
#  @brief This module defines the transformer model components used for the decoder-only architecture.
#         It includes the DataNormalizer, PositionalEncoding, and TransformerPredictor classes.

## @class DataNormalizer
#  @brief A class to normalize data using mean and standard deviation.
class DataNormalizer:
    ## @brief Constructor for DataNormalizer.
    #  @param eps A small epsilon value to avoid division by zero.
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.x_mean = None
        self.x_std = None
        self.u_mean = None
        self.u_std = None

    ## @brief Fits the normalizer to the data.
    #  @param x_data A numpy array of shape (N, T, state_dim) representing the state trajectories.
    #  @param u_data A numpy array of shape (N, T, control_dim) representing the control trajectories.
    def fit(self, x_data, u_data):
        self.x_mean = x_data.mean(axis=(0, 1))
        self.x_std = x_data.std(axis=(0, 1)) + self.eps
        self.u_mean = u_data.mean(axis=(0, 1))
        self.u_std = u_data.std(axis=(0, 1)) + self.eps

    ## @brief Transforms the state data using the fitted mean and std.
    #  @param x A numpy array representing the state data.
    #  @return A normalized numpy array.
    def transform_x(self, x):
        return (x - self.x_mean) / self.x_std

    ## @brief Transforms the control data using the fitted mean and std.
    #  @param u A numpy array representing the control data.
    #  @return A normalized numpy array.
    def transform_u(self, u):
        return (u - self.u_mean) / self.u_std

    ## @brief Inversely transforms the normalized control data back to the original scale.
    #  @param u A normalized numpy array.
    #  @return The denormalized control data.
    def inverse_transform_u(self, u):
        return u * self.u_std + self.u_mean


## @class PositionalEncoding
#  @brief Implements the positional encoding layer.
class PositionalEncoding(nn.Module):
    ## @brief Constructor for PositionalEncoding.
    #  @param d_model The embedding dimension.
    #  @param dropout Dropout probability.
    #  @param max_len The maximum sequence length.
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Precompute positional encodings.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    ## @brief Forward pass for positional encoding.
    #  @param x Input tensor of shape (batch, seq_len, d_model).
    #  @return Tensor with positional encoding applied.
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


## @class TransformerPredictor
#  @brief Implements a decoder-only transformer model for predicting control tokens.
class TransformerPredictor(nn.Module):
    ## @brief Constructor for TransformerPredictor.
    #  @param state_dim Dimension of the state input.
    #  @param control_dim Dimension of the control input (flattened, e.g., 52).
    #  @param d_model Embedding dimension.
    #  @param nhead Number of attention heads.
    #  @param num_decoder_layers Number of decoder layers.
    #  @param dim_feedforward Dimension of the feedforward network.
    #  @param dropout Dropout probability.
    #  @param max_seq_len Maximum sequence length.
    #  @param target_len Number of target tokens to predict.
    #  @param prompt_len Length of the prompt extracted from the control sequence.
    def __init__(self, state_dim, control_dim, d_model=64, nhead=8,
                 num_decoder_layers=3, dim_feedforward=128, dropout=0.1,
                 max_seq_len=100, target_len=20, prompt_len=10):
        super().__init__()
        self.d_model = d_model
        self.target_len = target_len
        self.prompt_len = prompt_len

        # Embedding layers for state and control inputs.
        self.state_embed = nn.Linear(state_dim, d_model)
        self.control_embed = nn.Linear(control_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        self.output_linear = nn.Linear(d_model, control_dim)

        # Build the decoder branch.
        decoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_decoder_layers)
        # Learnable target tokens.
        self.target_embedding = nn.Parameter(torch.zeros(target_len, d_model))
        nn.init.normal_(self.target_embedding, std=0.02)

    ## @brief Forward pass of the transformer model.
    #  @param x_seq Tensor of shape (batch, T, state_dim) representing the state trajectory.
    #  @param u_prompt Tensor of shape (batch, prompt_len, control_dim) representing the control prompt.
    #  @return Tensor of shape (batch, target_len, control_dim) with predicted control tokens.
    def forward(self, x_seq, u_prompt):
        batch_size = x_seq.size(0)
        # Embed state and control prompt.
        x_emb = self.state_embed(x_seq)
        u_emb = self.control_embed(u_prompt)
        # Concatenate state embeddings and prompt embeddings to form context.
        context = torch.cat([x_emb, u_emb], dim=1)
        # Append learnable target tokens.
        target_tokens = self.target_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        full_input = torch.cat([context, target_tokens], dim=1)
        full_input = self.pos_encoder(full_input)
        # Create causal mask for autoregressive behavior.
        L = full_input.size(1)
        causal_mask = torch.triu(torch.ones(L, L, device=full_input.device), diagonal=1).bool()
        decoded = self.transformer_decoder(full_input, mask=causal_mask)
        # Return predictions corresponding to the target tokens.
        return self.output_linear(decoded[:, -self.target_len:, :])
