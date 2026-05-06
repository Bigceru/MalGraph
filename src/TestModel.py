import json
import logging
import os
import zipfile
from datetime import datetime
from pathlib import Path

import torch
from torch_geometric.data import DataLoader, Batch
import sys

from models.HierarchicalGraphModel import HierarchicalGraphNeuralNetwork
from dataclasses import dataclass
from utils.Vocabulary import Vocab
from utils.RealBatch import create_real_batch_data

sys.path.append(str(Path("../samples")))  # Add the parent directory to the Python path
from samples.PreProcess import preprocess_pe, process_json_to_pyg


@dataclass
class ModelParams:
    gnn_type: str
    pool_type: str
    acfg_init_dims: int
    cfg_filters: str
    fcg_filters: str
    number_classes: int
    dropout_rate: float
    ablation_models: str


@dataclass
class TrainParams:
    processed_files_path: str
    # train_test_split_file: str
    max_epochs: int
    train_bs: int
    test_bs: int
    external_func_vocab_file: str
    max_vocab_size: int


def _parse_texttable_params(log_path):
    """
    Parse key/value pairs from the Texttable logs written by DistTrainModel.py.
    """
    params = {}
    current_key = None
    current_value_parts = []
    
    with open(log_path, "r", encoding="utf-8") as log_file:
        for raw_line in log_file:
            line = raw_line.rstrip("\n")
            if line.startswith("| Index | Parameters"):
                if current_key is not None:
                    params[current_key] = "".join(current_value_parts).strip()
                    current_key = None
                    current_value_parts = []
                continue

            if not line.startswith("|"):
                if current_key is not None:
                    params[current_key] = "".join(current_value_parts).strip()
                    current_key = None
                    current_value_parts = []
                continue

            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) < 3:
                continue

            _, key, value = cells[0], cells[1], cells[2]
            if key:
                if current_key is not None:
                    params[current_key] = "".join(current_value_parts).strip()
                current_key = key
                current_value_parts = [value] if value else []
            elif current_key is not None and value:
                current_value_parts.append(value)

    if current_key is not None:
        params[current_key] = "".join(current_value_parts).strip()

    return params


def _load_params_from_log(log_path):
    parsed_params = _parse_texttable_params(log_path)

    train_params = TrainParams(
        processed_files_path=parsed_params["processed_files_path"],
        max_epochs=int(parsed_params["max_epochs"]),
        train_bs=int(parsed_params["train_bs"]),
        test_bs=int(parsed_params["test_bs"]),
        external_func_vocab_file=parsed_params["external_func_vocab_file"],
        max_vocab_size=int(parsed_params["max_vocab_size"]),
    )

    model_params = ModelParams(
        gnn_type=parsed_params["gnn_type"],
        pool_type=parsed_params["pool_type"],
        acfg_init_dims=int(parsed_params["acfg_init_dims"]),
        cfg_filters=parsed_params["cfg_filters"],
        fcg_filters=parsed_params["fcg_filters"],
        number_classes=int(parsed_params["number_classes"]),
        dropout_rate=float(parsed_params["dropout_rate"]),
        ablation_models=parsed_params["ablation_models"],
    )

    return train_params, model_params


def find_training_log(model_path):
    """
    Find the training log written next to a checkpoint.
    DistTrainModel.py writes the log as a .txt file in the same output folder.
    """
    model_dir = Path(model_path).parent
    log_files = sorted(model_dir.glob("*.txt"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not log_files:
        raise FileNotFoundError(f"No training log (.txt) found next to {model_path}")
    return str(log_files[0])


def load_model_from_path(model_path, log_path=None, global_log=None, device=None):
    """
    Load a trained model from a checkpoint path using the saved training log.

    Args:
        model_path: Path to the saved model (.pt file)
        log_path: Optional path to the training .txt log written by DistTrainModel.py
        global_log: Optional logger passed into the model constructor
        device: Optional torch device or device string

    Returns:
        model: Loaded model in eval mode
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    if log_path is None:
        log_path = find_training_log(model_path)

    train_params, model_params = _load_params_from_log(log_path)

    # Initialize the vocabulary
    vocab_path = "../train_external_function_name_vocab.jsonl"
    global vocab
    vocab = Vocab(freq_file=vocab_path, max_vocab_size=train_params.max_vocab_size)

    # Initialize the global logger
    if global_log is None:
        global_log = logging.getLogger("TestModel")
        if not global_log.handlers:
            global_log.addHandler(logging.NullHandler())

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Loading model from: {model_path}")
    print(f"Using training log: {log_path}")

    model = HierarchicalGraphNeuralNetwork(model_params=model_params, external_vocab=vocab, global_log=global_log)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def _is_valid_checkpoint(model_path):
    """Check if a model file is a valid PyTorch checkpoint (valid zip archive)."""
    try:
        with zipfile.ZipFile(model_path, 'r') as zf:
            # Valid PyTorch checkpoint should have archive/data.pkl
            names = zf.namelist()
            return 'archive/data.pkl' in names or len(names) > 0
    except (zipfile.BadZipFile, OSError, FileNotFoundError):
        return False


def find_best_model(local_rank=0, search_dir="outputs"):
    """
    Find the best valid model file following DistTrainModel.py naming convention.
    Models are saved as: LocalRank_{rank}_best_model.pt
    Only returns valid checkpoint files (rejects corrupted ones).
    
    Args:
        local_rank: The rank/GPU ID used during training (default: 0)
        search_dir: Directory to search for models (default: "outputs")
    
    Returns:
        model_path: Path to the found valid model file
    """
    model_filename = f"LocalRank_{local_rank}_best_model.pt"
    search_path = Path(search_dir)
    
    # Search recursively for the model
    model_files = sorted(search_path.rglob(model_filename), 
                        key=lambda p: p.stat().st_mtime, 
                        reverse=True)
    
    if not model_files:
        raise FileNotFoundError(
            f"Model file '{model_filename}' not found in {search_dir}. "
            f"Expected from DistTrainModel.py training run."
        )
    
    # Find the first valid checkpoint (most recent first)
    for model_path in model_files:
        if _is_valid_checkpoint(str(model_path)):
            print(f"Found valid checkpoint: {model_path}")
            return str(model_path)
    
    # If no valid checkpoint found, raise error with details
    raise FileNotFoundError(
        f"Found {len(model_files)} model file(s) but all are corrupted or invalid. "
        f"Attempted files: {[str(p) for p in model_files]}"
    )


def load_files(input_path: str) -> list:
    path = Path(input_path)
    if path.is_file():
        return [path]
    
    files = []
    # Cycle through all files in the directory and subdirectories
    for path in path.glob("**/*"):
        if path.is_file():
            files.append(path)
    if files:
        return files
    
    # If no file has been found, raise an error
    raise ValueError(f"No files found in the provided path: {input_path}")


"""Extract statistics from the predictions, such as the count of benign vs. malware files for each file extension."""
def extract_statistics(predictions: dict) -> dict:
    # Collect all possible extensions (e.g., .exe, .dll) and their counts (benign vs. malware)
    extension_counts = {}

    for file_path, predicted_class in predictions.items():
        extension = file_path.suffix.lower()
        if extension in extension_counts:
            extension_counts[extension][predicted_class] += 1
        else:
            extension_counts[extension] = {"Malware": 0, "Benign": 0}   # Initialize the count for both classes
            extension_counts[extension][predicted_class] = 1

    return extension_counts


def print_statistics(statistics: dict) -> None:
    print("\nStatistics of Predictions by File Extension:")
    for extension, counts in statistics.items():
        total = counts["Benign"] + counts["Malware"]
        print(f"Extension: {extension} - Total: {total}")
        print(f"  Benign: {counts['Benign']}\t- {(counts['Benign'] / total * 100):.2f}%")
        print(f"  Malware: {counts['Malware']}\t- {(counts['Malware'] / total * 100):.2f}%\n")


# Example usage:
if __name__ == "__main__":
    try:
        # Search recursively through outputs/ for valid checkpoints
        model_path = find_best_model(local_rank=0, search_dir="outputs")
        model = load_model_from_path(model_path)
        print("Model loaded successfully via the saved checkpoint and training log.")

        # Test the model on the custom dataset
        pe_files = load_files("../../Datasets/MalwareBazaar/Malware")
        print(f"Loaded {len(pe_files)} PE files for testing.")

        predictions = {}
        for pe_file in pe_files:
            print(f"Processing file: {pe_file}")
            # Here you would add code to preprocess the PE file, create the appropriate input tensors,
            # and then pass them through the model to get predictions.
            # For example:
            json_item = preprocess_pe(str(pe_file))

            # Temp save the JSON data
            with open("temp_data.json", "w", encoding="utf-8") as f:
                json.dump(json_item, f, ensure_ascii=False)

            # Convert the JSON data to a PyG Data object and then to a tensor for model input
            pyg_file = process_json_to_pyg("temp_data.json", vocabulary=vocab)

            # Temp save the PyG Data object
            torch.save(pyg_file, "temp_data.pt")

            input_tensor = torch.load("temp_data.pt")  # Load the PyG Data object as a tensor

            batch = Batch.from_data_list([input_tensor])

            model.eval()    # Set model to evaluation mode

            # Convert to real batch format
            real_batch, positions, hashes, external_names, fcg_edges, labels = create_real_batch_data(batch)

            if real_batch is None:
                print(f"Skipping file {pe_file} due to preprocessing issues.")
                continue

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Make predictions with the model
            with torch.no_grad():
                prediction = model(
                    real_local_batch=real_batch.to(device),
                    real_bt_positions=positions,
                    bt_external_names=external_names,
                    bt_all_function_edges=fcg_edges,
                    local_device=device
                )

                predicted_class = torch.argmax(prediction, dim=1).item()
                predictions[str(pe_file)] = "Malware" if predicted_class == 0 else "Benign"
                print(f"Predicted class for {pe_file}: {predicted_class}")

        # Extract and print statistics
        stats = extract_statistics(predictions)
        print_statistics(stats)

        # Clean up temporary files
        if os.path.exists("temp_data.json"):
            os.remove("temp_data.json")
        if os.path.exists("temp_data.pt"):
            os.remove("temp_data.pt")

    except FileNotFoundError as exc:
        print(f"Error: {exc}")