import torch

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            predictions.extend(logits.cpu().numpy())
    return predictions


def get_device():
    #return torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    return torch.device('cuda' if torch.cuda.is_available() else'cpu')
