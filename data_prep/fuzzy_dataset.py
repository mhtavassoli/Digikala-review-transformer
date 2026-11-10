# data_prep/fuzzy_dataset.py
from torch.utils.data import Dataset
from utils.fuzzy_mapping import create_fuzzy_labels

class FuzzyDigikalaDataset(Dataset):
    """
    This dataset does NOT drop anomalies.
    It maps them to fuzzy probabilities.
    """
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Tokenization (The "Mesh" generation in FEM)
        text = str(row.get('body', ''))
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 2. Get Fuzzy Targets (The "Boundary Conditions")
        rec_probs, band_probs = create_fuzzy_labels(row)
        
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'rec_target': rec_probs,
            'band_target': band_probs
        }