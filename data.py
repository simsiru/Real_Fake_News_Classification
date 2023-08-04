import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class FakeTrueNewsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, 
                 max_sequence_length: int=256):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased", 
            do_lower_case=True
        )
        
        self.max_sequence_length = max_sequence_length

        self.df = df[['text', 'real_news']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        encoded = self.tokenizer.encode_plus(
            self.df['text'].iloc[idx],
            add_special_tokens=True,
            max_length=self.max_sequence_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            pad_to_max_length=True,
            return_tensors='pt',
        )

        encoded = {k:v.squeeze(0) for k,v in encoded.items()}
        encoded['labels'] = torch.tensor(self.df['real_news'].iloc[idx])

        return encoded