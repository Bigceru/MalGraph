import os
import json
from tqdm import tqdm
from Vocabulary import Vocab

def iter_dataset_files(dataset_path: str):
    for root, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                yield file_path


def main(dataset_path: str, vocab_path: str):
    # Open the vocabulary file
    vocabulary = Vocab(freq_file=vocab_path, max_vocab_size=10000)

    # Iter through all the files in the dataset directory
    for file in tqdm(iter_dataset_files(dataset_path), desc="Processing files"):
        # Read the file and associate the indexes of "function_edges" with the names of "function_names"
        with open(file, "r") as f:
            data = json.load(f)
            function_names = data.get("function_names", [])
            function_edges = data.get("function_edges", [])     # Format: [[src_index, ...], [dst_index, ...]]

            if not function_names or not function_edges:
                continue  # Skip files that don't have the required fields

            # Create a mapping of function index to function name
            index_to_name = {index: name for index, name in enumerate(function_names)}

            # Create a new mapping between old indexes and new indexes based on the vocabulary
            new_index_to_name = {}
            for index, name in index_to_name.items():
                if name in vocabulary.token_2_index.keys():
                    new_index = vocabulary.token_2_index[name]
                    new_index_to_name[index] = new_index
                else:
                    # If the function name is not in the vocabulary, skip it
                    continue

            # Update the function indexes using the new old_index to new_index mapping
            new_function_edges = []
            for indexes in function_edges:
                new_indexes = []
                for index in indexes:
                    index = new_index_to_name.get(index, index)  # If the index is not in the mapping, keep it unchanged
                    new_indexes.append(index)

                new_function_edges.append(new_indexes)

            # Update the data with the new function edges
            data["function_edges"] = new_function_edges

        # Save the updated data back to the file
        with open(file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True)


if __name__ == "__main__":
    # Define variables
    dataset_path="/home/davide/Malware/MalGraph/samples/dataset"
    vocab_path="/home/davide/Malware/MalGraph/train_external_function_name_vocab.jsonl"
    
    main(dataset_path, vocab_path)