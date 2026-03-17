from flask import Flask, request
import os
from predict_rnn import predict_symbol

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    
    result = ""
    if request.method == "POST":
        try:
            symbol = request.form.get("symbol")
            prediction = predict_symbol(symbol)
            result = f"<h1>Prediction for {symbol}: {prediction}</h1>"
        except Exception as e:
            result = f"<h1>Error: {e}</h1>"
    return f"""
    <h1>Stock Price Prediction</h1>
    <h2>Enter the symbol you want to predict (e.g., BTC-USD, AAPL, AUDCAD=X)</h2>
    <form method="post">
        <input type="text" name="symbol" placeholder="Stock Symbol (e.g. BTC-USD)" required>
        <button type="submit">Predict</button>
    </form>
    {result}
    """


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
