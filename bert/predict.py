import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from lib.utils import evaluate, get_device
from lib.dataset import CommonLitDataset

 # Set up
device = get_device()

batch_size = 16
max_length = 128

model = BertForSequenceClassification.from_pretrained('./fine_tuned_bert')

eval_df = pd.read_csv('../data/test.csv')

eval_dataset = CommonLitDataset(
    max_length,
    eval_df['excerpt'].values)

eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

predictions = evaluate(model, eval_dataloader, device)

print(pd.DataFrame({
    'id': eval_df['id'],
    'target': predictions
}))
