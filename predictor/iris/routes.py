from flask import (
    Blueprint, render_template, request, jsonify
)

from predictor.iris.models import load_knn_model
from predictor.db import knn_incorrect
import numpy as np

# from predictor.db import get_db

iris_bp = Blueprint('iris', __name__)

@iris_bp.route('/')
def iris():
    return render_template('iris/iris.html')

@iris_bp.route('/knn', methods=("GET", "POST"))
def knn_classifier():
    # k nearest neighbors model from hw5
    knn_model = load_knn_model()

    # index species
    SPECIES = ['setosa','versicolor','virginica']   # int to str
    SPECIES_INDEX = {'setosa':0,'versicolor':1,'virginica':2}  # str to int

    prediction = None

    if request.method == 'POST':
        if len(request.form) == 1:
            knn_incorrect()

        else:
            Features = np.asarray([request.form['sepallen'],
                        request.form['sepalwid'],
                        request.form['petallen'],
                        request.form['petalwid']],
                        dtype=float)

            prediction = knn_model.predict([Features])
            prediction = int(round(prediction[0]))  # unpack the extra brackets
            prediction = SPECIES[prediction]  # change to string

    return render_template('iris/knn.html', prediction=prediction)
