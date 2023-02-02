import pandas as pd
from torch.utils.data import Dataset


class PandasDataset(Dataset):
    def __init__(self, df: pd.DataFrame, columns: list, target_column: str=None):
        super().__init__()
        self.df = df
        self.columns = columns
        self.target_column = target_column

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        print(idx)
        batch = [self.df[col].values[idx] for col in self.columns]
        if self.target_column is None:
            return batch

        return batch, self.df.target_column.values[idx]