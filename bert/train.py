import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import Dataset, DatasetDict
from lib.trainer import create_trainer, get_tokenizer
from lib.tokenization import tokenize_commonlitreadabilityprize_dataset

# Read data
train_df, test_df = train_test_split(
    pd.read_csv('../data/train.csv'), 
    test_size=0.1, random_state=42)

datasets = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'eval': Dataset.from_pandas(test_df)
})

# Prep data
tokenizer = get_tokenizer('bert-large-uncased')
datasets = tokenize_commonlitreadabilityprize_dataset(datasets, tokenizer)

# Train

trainer = create_trainer(
    'bert-large-uncased',
    train_dataset=datasets["train"],
    eval_dataset=datasets["eval"]
)

print(trainer.train())

trainer.save_model()
trainer.save_state()

print(trainer.evaluate())