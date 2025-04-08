import os
import pickle
import argparse
import pandas as pd

from quattro_ilqr_tf.transformer_ilqr import *


def load_ilqr_logs(file_path):
    """
    Load ILQR logs from a combined pickle file that contains a sequence of pickled dictionaries.
    Returns a list of dictionaries.
    """
    print(f"Loading ILQR logs from {file_path}...", flush=True)
    logs = []
    try:
        with open(file_path, 'rb') as f:
            while True:
                try:
                    entry = pickle.load(f)
                    logs.append(entry)
                except EOFError:
                    break
    except Exception as e:
        print(f"Error loading pickle file: {e}", flush=True)
        return None

    print(f"Loaded {len(logs)} log entries.", flush=True)
    return logs


def process_ilqr_logs(ilqr_logs):
    """
    Process raw ILQR logs (a list of dictionaries) and convert them into a pandas DataFrame.
    """
    processed_data = []
    for entry in ilqr_logs:
        # Additional processing can be added here if needed.
        new_entry = {key: value for key, value in entry.items()}
        processed_data.append(new_entry)
    df = pd.DataFrame(processed_data)
    return df


def split_data(df, train_fraction=0.8, random_state=42):
    """
    Shuffle the DataFrame and split it into training and testing sets.
    Returns a tuple (train_df, test_df).
    """
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_idx = int(train_fraction * len(df_shuffled))
    train_df = df_shuffled.iloc[:split_idx]
    test_df = df_shuffled.iloc[split_idx:]
    print("Training set size:", len(train_df), flush=True)
    print("Test set size:", len(test_df), flush=True)
    return train_df, test_df


def train_transformer(train_df, num_epochs=200, batch_size=256, learning_rate=2e-4,
                      prompt_len=1, d_model=128, nhead=8):
    """
    Instantiate and train the TransformerILQR model using the training DataFrame.
    Allows setting prompt length, model dimension, and number of heads via parameters.
    """
    model_wrapper = TransformerILQR(
        state_dim=4,
        control_dim=5,
        prompt_len=prompt_len,
        d_model=d_model,
        nhead=nhead,
        num_decoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=100,
        quant_mode="float16",
    )
    print(f"Transformer Loaded. Device is: {model_wrapper.device}", flush=True)
    
    # Train the model using the custom fit() method.
    model_wrapper.fit(train_df, test_df=None, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)
    return model_wrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer ILQR Model")
    parser.add_argument("--file_path", type=str, default="examples/cartpole/training/combined_ilqr_logs_range_-0.500_0.500_angle_-0.500_0.500.pkl",
                        help="Path to ILQR logs pickle file")
    parser.add_argument("--prompt_len", type=int, default=10, help="Prompt length")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--log_file", type=str, default="training.log", help="Log file name")
    parser.add_argument("--task_name", type=str, default="default_task", help="Task name")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Print task and log information.
    print(f"Task Name: {args.task_name}", flush=True)
    print(f"Log File: {args.log_file}", flush=True)
    
    # Load and process the ILQR logs.
    ilqr_logs = load_ilqr_logs(args.file_path)
    if ilqr_logs is None:
        print("Failed to load ILQR logs.", flush=True)
        return

    df = process_ilqr_logs(ilqr_logs)
    print("Sample data from logs:", flush=True)
    print(df.head(), flush=True)
    
    # Split data into training and testing sets.
    train_df, test_df = split_data(df, train_fraction=0.8, random_state=42)
    
    # Train the transformer model using the training set with provided parameters.
    model_wrapper = train_transformer(train_df,
                                      num_epochs=args.num_epochs,
                                      batch_size=args.batch_size,
                                      learning_rate=args.learning_rate,
                                      prompt_len=args.prompt_len,
                                      d_model=args.d_model,
                                      nhead=args.nhead)
    
    # Save the trained model.
    model_wrapper.save("cartpole")
    print("Model training complete and saved.", flush=True)


if __name__ == "__main__":
    main()
