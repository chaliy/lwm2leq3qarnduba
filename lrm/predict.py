# %%
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
import joblib

# %%
print('Load the Test CSV file')
data = pd.read_csv('../data/test.csv')
X = data['excerpt']

# %%
print('Load saved vectorizer')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')
X_tfidf = loaded_vectorizer.transform(X)

# %%
print('Load saved model')
loaded_model = joblib.load('lrm.pkl')

# Use the loaded model for prediction
y_pred = loaded_model.predict(X_tfidf)

pd.DataFrame({
    'id': data['id'],
    'target': y_pred
})


# %%

