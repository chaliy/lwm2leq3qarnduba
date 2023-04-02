import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup

from lib.utils import evaluate, get_device
from lib.dataset import CommonLitDataset

def train(model, dataloader: DataLoader, optimizer, scheduler, device):
    model.train()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=targets.unsqueeze(-1))
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

def test(mode, test_df):
    test_dataset = CommonLitDataset(
        max_length,
        test_df['excerpt'].values
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    target = test_df['target']
    predictions = evaluate(model, test_dataloader, device)

    mse = mean_squared_error(target, predictions)
    r2 = r2_score(target, predictions)
    rmse = np.sqrt(mse)

    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R-squared: {r2:.2f}')
    print(f'Root Mean Squared Error: {rmse:.2f}')


 # Set up
device = get_device()
train_csv_path = '../data/train.csv'
epochs = 4
batch_size = 16
max_length = 128

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1).to(device)

# Read data
train_df, test_df = train_test_split(
    pd.read_csv(train_csv_path), 
    test_size=0.1, random_state=42)

# Create datasets and dataloaders
train_dataset = CommonLitDataset(
    max_length,
    train_df['excerpt'].values, 
    target=train_df['target'].values)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Prepare optimizer and learning rate scheduler
num_training_steps = len(train_dataloader) * epochs
num_warmup_steps = int(0.1 * num_training_steps)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

# Fine-tune the model
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    train(model, train_dataloader, optimizer, scheduler, device)
    test(model, test_df)

# Save the fine-tuned model
model.save_pretrained('fine_tuned_bert')