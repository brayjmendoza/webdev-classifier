import os
from time import sleep

from flask import (
    Blueprint, render_template, request, jsonify, session)

from classifier import socketio  # to give loading messages when clicking buttons

from classifier.commands import clean_files
from classifier.iris import model_loader
from classifier.database.db import get_db
from numpy import asarray, array, reshape, concatenate, arange, zeros, size
from sklearn.datasets import load_iris

cancer_bp = Blueprint('cancer', __name__)

# Keep track of target/feature indices
TARGET = ['malignant','benign']   # int to str
TARGET_INDEX = {'malignant':0,'benign':1}  # str to int
FEATURES = ['mean radius', 'mean texture', 'mean perimeter',
            'mean area', 'mean smoothness', 'mean compactness',
            'mean concavity', 'mean concave points', 
            'mean symmetry', 'mean fractal dimension', 
            'radius error', 'texture error', 'perimeter error',
            'area error', 'smoothness error', 'compactness error',
            'concavity error', 'concave points error', 'symmetry error',
            'fractal dimension error', 'worst radius', 'worst texture',
            'worst perimeter', 'worst area', 'worst smoothness',
            'worst compactness', 'worst concavity', 'worst concave points',
            'worst symmetry', 'worst fractal dimension']
FEATURES_INDEX = {feature: index for index, feature in enumerate(FEATURES)}

@cancer_bp.route('/')
def cancer():
    return render_template('cancer/home.html')


###################
## KNN FUNCTIONS
###################
@cancer_bp.route('/knn', methods=["GET"])
def knn_classifier():
    """View the page for the k-nearest neighbors cancer classifier"""

    # Clear knn models/plots created from past use
    clean_files(classify='iris', model='knn')

    db = get_db()
    corrections = db.execute(
        "SELECT * "
        "FROM cancer_features "
        "WHERE model LIKE 'knn'"
    ).fetchall()

    # Keep track if the model can be retrained or not
    if os.path.exists('classifier/models/iris/knn_new.pkl'):
        session['retrained'] = True
    else:
        session['retrained'] = False

    return render_template('cancer/knn.html',
                           corrections=corrections, 
                           index=TARGET,
                           instancePlotExists=False)

@cancer_bp.route('/knn/predict', methods=["POST"])
def knn_predict():
    """
    Make a breast cancer prediction using the KNN model.
    
    Also creates heatmap visualizations for the given features
    submitted on the web form if specified on the web page
    """
    # Load model
    socketio.emit('classify-status', {'message': 'Loading model...'})
    print('Loading model...')
    sleep(0.1)
    knn_model = model_loader.load_knn_model() 
    socketio.emit('classify-status', {'message': 'Loaded!'})
    print('Loaded!')

    # Get data
    socketio.emit('classify-status', {'message': 'Classifying...'})
    data = request.json

    # Determine if we need to plot
    last_key = list(data.keys())[-1]
    visualize = data.pop(last_key)

    # Store data in session to display on form
    session['cancer_features'] = data

    # Make prediction
    Features = asarray([data['radius'],
                data['texture'],
                data['perimeter'],
                data['area'],
                data['smoothness'],
                data['compactness'],
                data['concavity'],
                data['concave_points'],
                data['symmetry'],
                data['fractal_dimension']],
                dtype=float)

    prediction = knn_model.predict([Features])
    prediction = int(round(prediction[0]))  # unpack the extra brackets
    prediction = TARGET[prediction]  # change to string
    
    if visualize:
        # heatmap_visualization(knn_model, Features)
        return jsonify({"species": prediction, 
                        "images": render_template('iris/irisInstancePlots.html',
                                                    instancePlotExists=True)})
    
    return jsonify({"species": prediction})
