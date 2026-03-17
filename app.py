from flask import Flask;


app = Flask(__name__);


@app.route("/")
def Home():
     return "Hello Vercel"


if __name__ == "__main__":
     app.run()


