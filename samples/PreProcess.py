import json
import argparse
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import os
from r2_acfg_extractor import R2ACFGExtractor
from BuildExternalVocab import ExternalVocabBuilder
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from Vocabulary import Vocab


# Convert PE files to JSON format and build the vocabulary of external function names
def iter_dataset_files(dataset_path: str):
    for root, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                yield file_path


def _process_pe_file(job):
    input_file, dataset_path, output_dir = job
    output_file = input_file.replace(dataset_path, output_dir) + ".json"

    # Preserve directory structure for nested dataset paths.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(output_file):
        return

    extractor = R2ACFGExtractor(binary_path=input_file, output_path=output_file)
    extractor.run()


def pe_to_json(dataset_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    dataset_files = list(iter_dataset_files(dataset_path))
    jobs = [(input_file, dataset_path, output_dir) for input_file in dataset_files]

    # Use ProcessPoolExecutor to process files in parallel.
    with ProcessPoolExecutor(max_workers=12) as executor:
        list(
            tqdm(
                executor.map(_process_pe_file, jobs),
                total=len(dataset_files), 
                desc="Processing files (PE to JSON)"
                )
            )
    

    # for input_file in tqdm(dataset_files, desc="Processing files"):
    #     # Preserve directory structure: get relative path from dataset_path
    #     output_file = input_file.replace(dataset_path, output_dir) + ".json"  # Change extension to .json

    #     # If the output file already exists, skip processing
    #     if os.path.exists(output_file):
    #         continue
        
    #     extractor = R2ACFGExtractor(binary_path=input_file, output_path=output_file)
    #     extractor.run()


def parse_json_list_2_pyg_object(jsonl_file: str, label: int, vocab: Vocab, output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(jsonl_file, "r", encoding="utf-8") as file:
        for item in tqdm(file):
            item = json.loads(item)
            item_hash = item['hash']
            
            acfg_list = []
            for one_acfg in item['acfg_list']:  # list of dict of acfg
                block_features = one_acfg['block_features']
                block_edges = one_acfg['block_edges']
                one_acfg_data = Data(x=torch.tensor(block_features, dtype=torch.float), edge_index=torch.tensor(block_edges, dtype=torch.long))
                acfg_list.append(one_acfg_data)
            
            item_function_names = item['function_names']
            item_function_edges = item['function_edges']
            
            local_function_name_list = item_function_names[:len(acfg_list)]
            assert len(acfg_list) == len(local_function_name_list), "The length of ACFG_List should be equal to the length of Local_Function_List"
            external_function_name_list = item_function_names[len(acfg_list):]
            
            external_function_index_list = [vocab[f_name] for f_name in external_function_name_list]
            torch.save(
                Data(
                    hash=item_hash,
                    local_acfgs=acfg_list,
                    external_list=external_function_index_list,
                    function_edges=item_function_edges,
                    targets=label,
                ),
                output_file,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert MalGraph JSONL to PyG objects")
    parser.add_argument("--dataset-path", type=str, default="Datasets/pe-machine-learning-dataset/divided_dataset", help="Path to the dataset directory containing PE files")
    parser.add_argument("--output-dir-json", type=str, default="Datasets/pe-machine-learning-dataset/graph_data", help="Directory to save the converted JSON files")
    parser.add_argument("--output-dir-pyg", type=str, default="Datasets/pe-machine-learning-dataset/pyg_data", help="Directory to save the converted PyG objects")

    parser.add_argument("--vocab-file", type=str, default="./train_external_function_name_vocab.jsonl")
    parser.add_argument("--max-vocab-size", type=int, default=10000)
    parser.add_argument("--label", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir_json, exist_ok=True)
    os.makedirs(args.output_dir_pyg, exist_ok=True)

    # Convert PE files to JSON format
    pe_to_json(dataset_path=args.dataset_path, output_dir=args.output_dir_json)

    # Build vocabulary once from all generated JSON files
    external_vocab_builder = ExternalVocabBuilder(input_path=args.output_dir_json, output_file=args.vocab_file)
    external_vocab_builder.run()

    vocabulary = Vocab(freq_file=args.vocab_file, max_vocab_size=args.max_vocab_size)

    # Cycle for all the JSON files in the output directory and convert them to PyG objects
    for root, _, files in os.walk(args.output_dir_json):

        # Use ThreadPoolExecutor to process files in parallel
        with ThreadPoolExecutor(max_workers=12) as executor:
            json_files = [f for f in files if f.endswith(".json")]

            def _convert_one(json_file: str):
                json_file_path = os.path.join(root, json_file)
                output_file = os.path.join(
                    root.replace(args.output_dir_json, args.output_dir_pyg),
                    json_file.replace(".json", ".pt"),
                )

                if os.path.exists(output_file):
                    return

                parse_json_list_2_pyg_object(
                    jsonl_file=json_file_path,
                    label=args.label,
                    vocab=vocabulary,
                    output_file=output_file,
                )

            list(
                tqdm(
                    executor.map(_convert_one, json_files),
                    total=len(json_files),
                    desc="Converting JSON to PyG objects",
                    leave=False,
                )
            )

        # for json_file in tqdm(files, desc="Converting JSON to PyG objects", leave=False):
        #     if json_file.endswith(".json"):
        #         json_file_path = os.path.join(root, json_file)
                
        #         # Preserve directory structure in output
        #         output_file = json_file_path.replace(args.output_dir_json, args.output_dir_pyg).replace(".json", ".pt")

        #         # If the output file already exists, skip processing
        #         if os.path.exists(output_file):
        #             continue
                
        #         parse_json_list_2_pyg_object(jsonl_file=json_file_path, label=args.label, vocab=vocabulary, output_file=output_file)