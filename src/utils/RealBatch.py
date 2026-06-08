import torch
from torch_geometric.data import Batch
from pprint import pprint


def create_real_batch_data(one_batch: Batch):
    real = []
    position = [0]
    count = 0
    
    assert len(one_batch.external_list) == len(one_batch.function_edges) == len(one_batch.local_acfgs) == len(one_batch.hash), "size of each component must be equal to each other"
    
    for item in one_batch.local_acfgs:
        for acfg in item:
            real.append(acfg)
        count += len(item)
        position.append(count)
    
    if len(one_batch.local_acfgs) == 1 and len(one_batch.local_acfgs[0]) == 0:
        return (None for _ in range(6))
    else:
        real_batch = Batch.from_data_list(real)

        # Extract targets/labels from the batch - the result can be None if no labels are found
        targets = getattr(one_batch, "targets", None)
        if targets is None:
            targets = getattr(one_batch, "labels", None)
        if targets is None:
            targets = getattr(one_batch, "y", None)

        return real_batch, position, one_batch.hash, one_batch.external_list, one_batch.function_edges, targets