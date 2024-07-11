from flask import (
    Blueprint, render_template, request, jsonify, session
)

from predictor.iris.models import load_knn_model
from predictor.database.db import get_db
from numpy import asarray

iris_bp = Blueprint('iris', __name__)

@iris_bp.route('/')
def iris():
    return render_template('iris/iris.html')

@iris_bp.route('/knn', methods=["GET"])
def knn_classifier():
    return render_template('iris/knn.html')

@iris_bp.route('/knn/predict', methods=["POST"])
def knn_predict():
    knn_model = load_knn_model()  # k nearest neighbors model from hw5

    # index species
    SPECIES = ['setosa','versicolor','virginica']   # int to str

    data = request.json
    session['iris_features'] = data   # store data in session in case of correction

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
    # Get species index
    SPECIES_INDEX = {'setosa':0,'versicolor':1,'virginica':2}  # str to int
    correction = request.json
    species = SPECIES_INDEX[correction['correction']]
    
    # Get inputted features from session
    iris_features = session['iris_features']

    # Add correction to database
    db = get_db()

    db.execute(
        "INSERT INTO iris (sepallen, sepalwid, petallen, petalwid, species)"
        "VALUES (?, ?, ?, ?, ?)",
        (iris_features['sepallen'], iris_features['sepalwid'], 
         iris_features['petallen'], iris_features['petalwid'], species)
    )
    db.commit()

    return '', 204 # No Content response