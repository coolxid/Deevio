from flask import Flask, jsonify, request, send_file
from predict import *

app = Flask(__name__)
#api = Api(app)

@app.route('/predict',methods=['GET'])
def run():
    url = request.args.get('image')
    result = do_pred(url)
    print(result)
    output = [{
        'result': result
    }]
    return jsonify({'Model_Prediction': output})

@app.route('/image/<path:image>')
def image(image):
    return send_file('image/'+image, mimetype='image/jpg')


if __name__ == '__main__':
    app.run(host='0.0.0.0')