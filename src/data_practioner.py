import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
# Path to the preprocessed file
# PREPROCESSED_DATA_PATH = 'data/Preprocessed/preprocessed_2017.csv'
PREPROCESSED_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', # This goes up one level from 'src'
    'data',
    'Preprocessed',
    'cicids2017_preprocessed.csv'
)
# Output directory for the client partitions
OUTPUT_DIR = 'data/federated_data'
# Number of clients (Raspberry Pis) in your testbed
NUM_CLIENTS = 3

# --- Partitioning Function ---
def create_non_iid_partitions(df, num_clients, target_col='label'):
    """
    Creates non-IID partitions of the dataset based on class distribution.
    This simulates a heterogeneous environment where clients have different data mixes.
    
    :param df: The preprocessed DataFrame.
    :param num_clients: Number of partitions to create.
    :param target_col: The name of the target label column.
    :return: A list of DataFrames, one for each client.
    """
    print("--- Creating Non-IID Data Partitions ---")
    
    # Get the unique labels (attack types)
    unique_labels = df[target_col].unique()
    
    client_partitions = [pd.DataFrame() for _ in range(num_clients)]
    
    # Split the data for each label across clients to create non-IID data
    for label in unique_labels:
        # Get all rows for this specific label
        label_data = df[df[target_col] == label]
        
        # Shuffle the data for this label
        label_data = label_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Distribute the data for this label unequally among clients
        # For example, client 0 gets more of this label, client 1 gets some, etc.
        split_points = [0]
        # This creates a non-uniform split. For 3 clients, it would be roughly 60/30/10 split
        # This is a conceptual way to create non-IID data.
        split_points.append(int(len(label_data) * 0.6))
        split_points.append(int(len(label_data) * 0.9))
        split_points.append(len(label_data))
        
        for i in range(num_clients):
            start = split_points[i]
            end = split_points[i+1]
            client_partitions[i] = pd.concat([client_partitions[i], label_data[start:end]], ignore_index=True)
            
    print(f"Created {num_clients} partitions.")
    for i, part in enumerate(client_partitions):
        print(f"Client {i}: {len(part)} samples")
        
    return client_partitions

# --- Main Script Execution ---
if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # 1. Load the preprocessed dataset
        preprocessed_df = pd.read_csv(PREPROCESSED_DATA_PATH)
        print(f"Loaded preprocessed data with shape: {preprocessed_df.shape}")
        
        # 2. Create non-IID partitions
        partitions = create_non_iid_partitions(preprocessed_df, NUM_CLIENTS, target_col='label')
        
        # 3. Save each partition to a separate CSV file
        for i, client_df in enumerate(partitions):
            # Shuffle each client's data one last time for good measure
            client_df = client_df.sample(frac=1, random_state=42).reset_index(drop=True)
            output_path = os.path.join(OUTPUT_DIR, f'client_{i}_data.csv')
            client_df.to_csv(output_path, index=False)
            print(f"Saved partition for Client {i} to '{output_path}'")
            
    except FileNotFoundError:
        print(f"Error: Preprocessed data file not found at {PREPROCESSED_DATA_PATH}.")
        print("Please ensure you have run your data_preprocessing notebook.")
    except Exception as e:
        print(f"An error occurred during partitioning: {e}")