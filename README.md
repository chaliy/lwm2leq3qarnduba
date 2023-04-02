## Structure

- `lrm` - model implemented using a linear regression
- `bert` - fine-tuned pre-trained BERT model
- `funcs` - Azure functions deployment

## To play with

```sh
curl -X POST -d 'some text to predict' 'https://onr55m2crkbyuag7.azurewebsites.net/api/predict'
```

NOTE: Eventually I will drop this deployment