
from urllib import request
import flask
import training


app = flask.Flask(__name__)
i=-1
@app.route('/ask',methods=['POST'])
def index():
    global i
    data=flask.request.get_json()
    result=training.ask(data["input"])
    return flask.jsonify({"text":result})

if __name__=="__main__":
    app.run(debug=True)
