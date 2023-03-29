import os
import azure.functions as func
import pandas as pd
import joblib
import json
import logging

relative_path = lambda p: os.path.join(os.path.dirname(__file__), p)

vectorizer = joblib.load(relative_path('tfidf_vectorizer.pkl'))
model = joblib.load(relative_path('lrm.pkl'))

def main(req: func.HttpRequest) -> func.HttpResponse:

  req_text = req.get_body().decode('utf-8').strip(' \t\n\r\"')

  logging.info(f'Using text: {req_text}')

  X = [req_text]
  X_tfidf = vectorizer.transform(X)

  y_pred = model.predict(X_tfidf)

  logging.info(f'Prediction: {y_pred}')

  results = {
    'output': [{
      'excerpt': x,
      'prediction': y
    } for x, y in zip(X, y_pred)]
  }

  return func.HttpResponse(
    json.dumps(results),
    mimetype="application/json"
  )
