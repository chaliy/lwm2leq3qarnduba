# %%
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# %%
print('Load the CSV file')
data = pd.read_csv('../data/train.csv')
X = data['excerpt']
y = data['target']

plt.hist(y, bins=100)


# %%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%

print('Create a TfidfVectorizer to transform the text data')
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(vectorizer, '../func/tfidf_vectorizer.pkl')


# %%
print('Train a linear regression model')
reg = LinearRegression()
reg.fit(X_train_tfidf, y_train)

# %%
# Make predictions on the test set
y_pred = reg.predict(X_test_tfidf)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')

# %%
# Save the trained model
joblib.dump(reg, 'lrm.pkl')
joblib.dump(reg, '../func/lrm.pkl')

# %%
