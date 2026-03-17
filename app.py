from flask import Flask, request

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        try:
            n1 = float(request.form.get("n1"))
            n2 = float(request.form.get("n2"))
            result = f"<h1>Result: {n1 * n2}</h1>"
        except ValueError:
            result = "<h1>Invalid Input</h1>"
    return f"""
    <form method="post">
        <input type="number" name="n1" placeholder="Number 1" required>
        <input type="number" name="n2" placeholder="Number 2" required>
        <button type="submit">Multiply</button>
    </form>
    {result}
    """


if __name__ == "__main__":
    app.run(debug=True)
