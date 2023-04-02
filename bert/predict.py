import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from lib.trainer import create_trainer, get_tokenizer
from lib.tokenization import tokenize_commonlitreadabilityprize_dataset

tokenizer = get_tokenizer('./fine_tuned_bert')

eval_df = pd.read_csv('../data/test.csv')
dataset = tokenize_commonlitreadabilityprize_dataset(Dataset.from_pandas(eval_df), tokenizer)

trainer = create_trainer('./fine_tuned_bert')

predictions, labels, metrics = trainer.predict(dataset)

print(pd.DataFrame({
    'id': eval_df['id'],
    'target': predictions.flatten(),
}))
