import os

from flask import Blueprint, render_template, request, jsonify, session

from predictor.iris import models
from predictor.database.db import get_db
from numpy import asarray, array, reshape, concatenate
from sklearn.datasets import load_iris

iris_bp = Blueprint('iris', __name__)

# index species
SPECIES = ['setosa','versicolor','virginica']   # int to str
SPECIES_INDEX = {'setosa':0,'versicolor':1,'virginica':2}  # str to int

@iris_bp.route('/')
def iris():
    return render_template('iris/iris.html')

@iris_bp.route('/knn', methods=["GET"])
def knn_classifier():
    db = get_db()
    corrections = db.execute(
        "SELECT sepallen, sepalwid, petallen, petalwid, species "
        "FROM iris "
        "WHERE model LIKE 'knn'"
    ).fetchall()

    # Keep track if the model can be retrained or not
    if corrections:
        session['retrained'] = False
    else:
        session['retrained'] = True

    return render_template('iris/knn.html', 
                           corrections=corrections, 
                           index=SPECIES)

@iris_bp.route('/knn/predict', methods=["POST"])
def knn_predict():
    print('loading model...')
    knn_model = models.load_knn_model()  # k nearest neighbors model from hw5
    print('loaded!')

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
    correction = request.json
    species = SPECIES_INDEX[correction['correction']]
    
    # Get inputted features from session
    iris_features = session['iris_features']

    # Add correction to database
    db = get_db()

    db.execute(
        "INSERT INTO iris (sepallen, sepalwid, petallen, petalwid, species, model)"
        "VALUES (?, ?, ?, ?, ?, 'knn')",
        (iris_features['sepallen'], iris_features['sepalwid'], 
         iris_features['petallen'], iris_features['petalwid'], species)
    )
    db.commit()

    # Get all new corrections to update corrections.html
    new_corrections = db.execute(
        "SELECT sepallen, sepalwid, petallen, petalwid, species "
        "FROM iris "
        "WHERE model LIKE 'knn'"
    ).fetchall()

    # Model can now be retrained
    session['retrained'] = False

    # Convert to list of dicts for JSON serialization
    all_corrections = [dict(row) for row in new_corrections]

    return render_template('corrections.html', 
                           corrections=all_corrections, 
                           index=SPECIES)

@iris_bp.route('/knn/retrain', methods=['POST'])
def knn_retrain():
    db = get_db()
    session['retrained'] = True

    iris_data = load_iris()

    # Get pre-existing feature and target data from dataset
    feature_data = iris_data['data']
    target_data = iris_data['target']

    # Get new features and target data from database
    new_features = db.execute(
        "SELECT sepallen, sepalwid, petallen, petalwid "
        "FROM iris "
        "WHERE model LIKE 'knn'"
    ).fetchall()
    new_targets = db.execute(
        "SELECT species FROM iris "
        "WHERE model LIKE 'knn'"
    ).fetchall()

    # Format as numpy array
    new_features = array(new_features, dtype='float64')
    new_targets = array(new_targets, dtype='float64')
    new_targets = reshape(new_targets, -1)  # convert to 1D array

    # Combine all features and target data
    all_features = concatenate((feature_data, new_features))
    all_targets = concatenate((target_data, new_targets))

    print("Obtained all data...")

    ####################
    ##  Retrain model! 
    #   NOTE: Look into incremental learning
    ####################
    from numpy.random import permutation
    from sklearn.neighbors import KNeighborsClassifier
    from joblib import dump

    # Scramble data to remove (potential) dependence on ordering
    indices = permutation(len(all_targets))
    all_features = all_features[indices]
    all_targets = all_targets[indices]
    
    print("Scrambled data!")

    ##### Since the dataset is small, we will use all data
    # Define training and testing sets
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(all_features, 
    #                                                    all_targets, 
    #                                                     test_size=models.TEST_PERCENT)
    
    
    # Train new model
    print("Training...")
    new_knn_model = KNeighborsClassifier(n_neighbors=models.BEST_K)
    new_knn_model.fit(all_features, all_targets)
    print("Trained!")
    
    # Save new model
    print("Saving...")
    dump(new_knn_model, 'predictor/iris/models/iris_knn_new.pkl')
    print("Saved!")

    return ''