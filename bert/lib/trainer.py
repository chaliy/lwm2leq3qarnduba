from sklearn.metrics import mean_squared_error

from transformers import Trainer, TrainingArguments
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer
)

def _compute_metrics(pred):
    predictions, labels = pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {
        "rmse": rmse
    }

def get_tokenizer(model_name_or_path):
    return AutoTokenizer.from_pretrained(model_name_or_path)

def create_trainer(model_name_or_path, **kwargs):

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=1
    )
    tokenizer = get_tokenizer(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config
    )

    args = TrainingArguments(
        output_dir='./fine_tuned_bert',
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        use_mps_device=True,
    )

    return Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
        **kwargs
    )