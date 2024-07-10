from flask import (
    Blueprint, render_template, request, jsonify
)

from predictor.iris.models import load_knn_model
# from predictor.db import knn_incorrect
from numpy import asarray

# from predictor.db import get_db

iris_bp = Blueprint('iris', __name__)
knn_model = load_knn_model()  # k nearest neighbors model from hw5

@iris_bp.route('/')
def iris():
    return render_template('iris/iris.html')

@iris_bp.route('/knn', methods=["GET"])
def knn_classifier():
    return render_template('iris/knn.html')

@iris_bp.route('/knn/predict', methods=["POST"])
def knn_predict():
    # index species
    SPECIES = ['setosa','versicolor','virginica']   # int to str

    prediction = None

    data = request.json

    Features = asarray([data['sepallen'],
                data['sepalwid'],
                data['petallen'],
                data['petalwid']],
                dtype=float)

    prediction = knn_model.predict([Features])
    prediction = int(round(prediction[0]))  # unpack the extra brackets
    prediction = SPECIES[prediction]  # change to string
    
    return jsonify({"species": prediction})


@iris_bp.route('/incorrect', methods=["POST"])
def knn_incorrect():
    # index species
    SPECIES_INDEX = {'setosa':0,'versicolor':1,'virginica':2}  # str to int
    
    correction = request.json
    species = float(SPECIES_INDEX[correction['correction']])

    print(species)
    # DATABASE STUFF

    return '', 204 # No Content response