import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CommonLitDataset(Dataset):
    def __init__(self, max_length, excerpt, target=None, tokenizer=tokenizer):
        self.max_length = max_length
        self.excerpt = excerpt
        self.target = target
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.excerpt)

    def __getitem__(self, item):
        text = str(self.excerpt[item])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        results = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.target is not None:
            results['targets'] = torch.tensor(self.target[item], dtype=torch.float32)

        return results

