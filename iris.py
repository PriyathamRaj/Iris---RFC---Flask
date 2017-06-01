import numpy as np

from flask import Flask, abort, jsonify, request
import six.moves.cPickle as pickle

my_random_forest = pickle.load(open("iris_rfc.pkl", "rb"))

app = Flask(__name__)

@app.route('/api', methods = ['POST'])
def make_predict():
	data = request.get_json(force = True)
	# convert json to a numpy array
	predict_request = [data['sl'], data['sw'], data['pl'], data['pw']]
	predict_request = np.array(predict_request)
	
	y_pred = my_random_forest.predict(predict_request)
	# return prediction
	output = [y_pred[0]]
	return jsonify(results=output)
	
if __name__ == '__main__':
	app.run(port = 9000, debug = True)