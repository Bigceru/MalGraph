import csv
import os
import os.path as osp
import re
from datetime import datetime

import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .RealBatch import create_real_batch_data  # noqa


def _allowed_stems(manifest_csv, file_types, split=None):
    """Returns a set of allowed stems (filenames without extensions) based on the manifest CSV and specified file types and split.
    
    Args:
        manifest_csv (str): Path to the manifest CSV file containing file_type and split columns.
        file_types (list): List of file types to include (e.g., ['pe_exe', 'pe_dll']). If None, all file types are allowed.
        split (str, optional): The split to filter by (e.g., 'train', 'valid', 'test'). If None, no split filtering is applied.

    Returns:
        set: A set of allowed stems (lowercased) that match the specified file types and split. If file_types is None, returns None. Raises ValueError if file_types is specified but manifest_csv is not provided.
    """
    if not file_types:
        return None
    if not manifest_csv:
        raise ValueError("file_types filtering requires a manifest CSV (with file_type/split columns)")
    
    wanted = set(file_types)
    allowed = set()
    with open(manifest_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            # Skip rows that don't match the desired file types or split
            if row.get("file_type", "").strip() not in wanted:
                continue
            if split is not None and row.get("split", "").strip() != split:
                continue

            # Extract the stem (filename without extension) from the malgraph_path or filename
            ref = (row.get("malgraph_path") or "").strip()
            stem = osp.splitext(osp.basename(ref))[0] if ref else ""

            # If the stem is empty, try to extract a 64-character hex string from the filename
            if not stem:
                match = re.search(r"[0-9a-fA-F]{64}", row.get("filename", ""))
                stem = match.group(0) if match else ""

            # If we have a valid stem, add it to the allowed set (in lowercase)
            if stem:
                allowed.add(stem.lower())
    return allowed


class MalwareDetectionDataset(Dataset):
    def __init__(self, root, train_or_test, manifest_csv=None, file_types=None, transform=None, pre_transform=None):
        super(MalwareDetectionDataset, self).__init__(None, transform, pre_transform)
        self.flag = train_or_test.lower()
        self.malware_root = os.path.join(root, "{}".format(self.flag), "Malware")
        self.benign_root = os.path.join(root, "{}".format(self.flag), "Benign")

        # Collect valid .pt files for malware and benign samples
        # self.malware_files = self._collect_valid_pt_files(self.malware_root)
        # self.benign_files = self._collect_valid_pt_files(self.benign_root)
        self.malware_files = self._list_files_for_pt(self.malware_root)
        self.benign_files = self._list_files_for_pt(self.benign_root)

        # Filter the files based on the manifest CSV and specified file types
        allowed = _allowed_stems(manifest_csv, file_types, self.flag)
        if allowed is not None:
            self.malware_files = [f for f in self.malware_files if osp.splitext(f)[0].lower() in allowed]
            self.benign_files = [f for f in self.benign_files if osp.splitext(f)[0].lower() in allowed]
    
    @staticmethod
    def _list_files_for_pt(the_path):
        files = []
        for name in os.listdir(the_path):
            if os.path.splitext(name)[-1] == '.pt':
                files.append(name)
        return files

    @staticmethod
    def _collect_valid_pt_files(directory):
        """Collects valid .pt files from the given directory. Avoids loading files that cannot be loaded as PyTorch tensors."""
        valid_files = []
        for name in tqdm(sorted(os.listdir(directory)), desc=f"Checking .pt files in {directory}"):
            if os.path.splitext(name)[-1] != '.pt':
                continue

            full_path = osp.join(directory, name)
            try:
                item = torch.load(full_path, weights_only=False)
            except Exception:
                continue

            if item is None:
                continue

            valid_files.append(name)

        return valid_files
    
    def len(self):
        return len(self.malware_files) + len(self.benign_files)

    def __len__(self):
        return self.len()
    
    def get(self, idx):
        split = len(self.malware_files)
        # split = 100
        if idx < split:
            idx_data = torch.load(osp.join(self.malware_root, self.malware_files[idx]), weights_only=False)
            idx_data.targets = 1
        else:
            over_fit_idx = idx - split
            idx_data = torch.load(osp.join(self.benign_root, self.benign_files[over_fit_idx]), weights_only=False)
            idx_data.targets = 0
        return idx_data


def _simulating(_dataset, _batch_size: int):
    print("\nBatch size = {}".format(_batch_size))
    time_start = datetime.now()
    print("start time: " + time_start.strftime("%Y-%m-%d@%H:%M:%S"))
    
    # https://github.com/pytorch/fairseq/issues/1560
    # https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189
    # loaders_1 = DataLoader(dataset=benign_exe_dataset, batch_size=10, shuffle=True, num_workers=0)
    # increasing the shared memory: ulimit -SHn 51200
    loader = DataLoader(dataset=_dataset, batch_size=_batch_size, shuffle=True)  # default of prefetch_factor = 2 # num_workers=4
    
    for index, data in enumerate(loader):
        if index >= 3:
            break
        _real_batch, _position, _hash, _external_list, _function_edges, _true_classes = create_real_batch_data(one_batch=data)
        print(data)
        print("Hash: ", _hash)
        print("Position: ", _position)
        print("\n")
    
    time_end = datetime.now()
    print("end time: " + time_end.strftime("%Y-%m-%d@%H:%M:%S"))
    print("All time = {}\n\n".format(time_end - time_start))


if __name__ == '__main__':
    root_path: str = '/home/xiang/MalGraph/data/processed_dataset/DatasetJSON/'
    i_batch_size = 2
    
    train_dataset = MalwareDetectionDataset(root=root_path, train_or_test='train')
    print(train_dataset.malware_root, train_dataset.benign_root)
    print(len(train_dataset.malware_files), len(train_dataset.benign_files), len(train_dataset))
    _simulating(_dataset=train_dataset, _batch_size=i_batch_size)
    
    valid_dataset = MalwareDetectionDataset(root=root_path, train_or_test='valid')
    print(valid_dataset.malware_root, valid_dataset.benign_root)
    print(len(valid_dataset.malware_files), len(valid_dataset.benign_files), len(valid_dataset))
    _simulating(_dataset=valid_dataset, _batch_size=i_batch_size)
    
    test_dataset = MalwareDetectionDataset(root=root_path, train_or_test='test')
    print(test_dataset.malware_root, test_dataset.benign_root)
    print(len(test_dataset.malware_files), len(test_dataset.benign_files), len(test_dataset))
    _simulating(_dataset=test_dataset, _batch_size=i_batch_size)