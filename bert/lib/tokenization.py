def tokenize_commonlitreadabilityprize_dataset(dataset, tokenizer):

    def tokenize(examples):
        tokens = tokenizer(
            examples["excerpt"],
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        if "target" in examples:
            tokens["labels"] = examples["target"]
        
        return tokens

    return dataset.map(tokenize, batched=True, num_proc=4, remove_columns=["excerpt"])