import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
# 📈 Stock prediction function
def predict_stock():
    symbol = symbol_entry.get()
    if symbol == "":
        messagebox.showerror("Error", "Stock symbol daalo bhai!")
        return

    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            messagebox.showerror("Error", "Koi data nahi mila. Symbol check karo.")
            return

        data['Prediction'] = data['Close'].shift(-30)
        data = data.dropna()

        X = np.array(data[['Close']])
        y = np.array(data['Prediction'])

        model = LinearRegression()
        model.fit(X, y)

        # Last close price se prediction
        future_input = np.array(data[['Close']].tail(1))
        predicted_price = model.predict(future_input)

        messagebox.showinfo("Prediction", f"📊 30 din baad ka predicted price: ₹{predicted_price[0]:.2f}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# 🪟 GUI Setup
window = tk.Tk()
window.title("📉 Stock Price Predictor - India")
window.geometry("400x200")

label = tk.Label(window, text="Stock Symbol daalo (e.g. TCS.NS):")
label.pack(pady=10)

symbol_entry = tk.Entry(window)
symbol_entry.pack(pady=5)

predict_button = tk.Button(window, text="🔮 Predict 30 Days", command=predict_stock)
predict_button.pack(pady=20)

window.mainloop()
