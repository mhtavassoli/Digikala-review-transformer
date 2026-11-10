# models/curriculum.py
import torch
from torch.utils.data import Dataset

class CurriculumDataset(Dataset):
    """
    A dataset that returns a 'difficulty' score for each sample.
    Initial difficulty: 1.0 for easy (rate and rec align), 0.5 for hard.
    This can be updated dynamically during training.
    """
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Initial difficulty heuristic
        self.difficulties = self._compute_initial_difficulty()
    
    def _compute_initial_difficulty(self):
        # Align rate (1-5) with recommendation (recommended=1, not=0, no_idea=0.5)
        # If they agree, it's easy (low difficulty); if not, hard.
        difficulties = []
        for _, row in self.df.iterrows():
            rate = row.get('rate_for_band', np.nan)
            rec = row.get('recommendation_status', None)
            if pd.isna(rate) or rec is None:
                difficulties.append(1.0) # neutral
                continue
            # Map rate to expected rec
            if rate >= 4.0:
                expected = 'recommended'
            elif rate <= 2.5:
                expected = 'not_recommended'
            else:
                expected = 'no_idea' # or ambiguous
            if rec == expected:
                difficulties.append(0.5) # easy
            else:
                difficulties.append(1.5) # hard
        return torch.tensor(difficulties)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # ... tokenization ...
        return {
            'input_ids': ...,
            'attention_mask': ...,
            'rec_label': ...,
            'band_label': ...,
            'difficulty': self.difficulties[idx]
        }