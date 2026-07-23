from flask import Flask

app = Flask(__name__)
@app.route('/greet/<name>')
def index(name):
    return f"Flask server is running ya {name}"

if __name__ =='__main__':
    app.run(host='0.0.0.0',port=5555,debug = True)


