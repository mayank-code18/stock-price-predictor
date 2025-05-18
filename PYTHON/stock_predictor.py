import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


ticker = "GAIL.NS"

#  Date range
data = yf.download(ticker, start="2015-01-01", end="2024-12-31")

#  Print first 5 rows
print(data.head())

#  Chart show karo
data['Close'].plot(title=f"{ticker} Closing Price", figsize=(10, 4))
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

#  Target: Closing Price
data = data[['Close']]

#  Shifted Close column to predict next day
data['Prediction'] = data[['Close']].shift(-30)

#  Features & Labels
X = np.array(data.drop(['Prediction'], axis=1))[:-30]
y = np.array(data['Prediction'])[:-30]

#  Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#  Model Training
lr = LinearRegression()
lr.fit(X_train, y_train)

#  Model Accuracy
accuracy = lr.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")
#  Last 30 Days
X_future = data.drop(['Prediction'], axis=1)[-30:]

#  Predict
future_prediction = lr.predict(X_future)

#  Dekho prediction values
print("\nNext 30 Days Prediction:")
print(future_prediction)
#  Visual Compare
predicted = pd.DataFrame(future_prediction, columns=['Predicted'])
predicted.index = data[-30:].index

plt.figure(figsize=(10, 5))
plt.plot(data['Close'][-60:], label='Actual')
plt.plot(predicted['Predicted'], label='Predicted', linestyle='dashed')
plt.title("Stock Price Prediction - Next 30 Days")
plt.legend()
plt.show()
