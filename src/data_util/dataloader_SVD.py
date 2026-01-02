import torch.utils.data as data_utils
import numpy as np

## dataloader for SVD

class SVDDataloader(data_utils.Dataset):
    # as we already converted to tensor, we can directly return the tensor
    def __init__(self, x: np.ndarray, svd_emb: np.ndarray, cons: np.ndarray, y: np.ndarray, c: np.ndarray) -> None:
        self.x = x
        self.svd_emb = svd_emb
        self.cons = cons
        self.y = y
        self.c = c
        super().__init__()
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.svd_emb[index], self.cons[index], self.y[index], self.c[index]