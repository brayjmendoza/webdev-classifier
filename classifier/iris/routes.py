import os
from time import sleep

from flask import (
    Blueprint, render_template, request, jsonify, session, current_app)
# from flask_socketio import emit

from classifier.commands import clean_files
from classifier import socketio
from classifier.iris import model_loader
from classifier.database.db import get_db
from numpy import asarray, array, reshape, concatenate, arange, zeros
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

    clean_files(classify='iris', model='knn')

    db = get_db()
    corrections = db.execute(
        "SELECT sepallen, sepalwid, petallen, petalwid, species "
        "FROM iris "
        "WHERE model LIKE 'knn'"
    ).fetchall()

    # Keep track if the model can be retrained or not
    if os.path.exists('classifier/models/iris/knn_new.pkl'):
        session['retrained'] = True
    else:
        session['retrained'] = False

    return render_template('iris/knn.html', 
                           corrections=corrections, 
                           index=SPECIES,
                           model='knn')

@iris_bp.route('/knn/predict', methods=["POST"])
def knn_predict():
    # Load model
    socketio.emit('classify-status', {'message': 'Loading model...'})
    print('Loading model...')
    sleep(0.1)
    knn_model = model_loader.load_knn_model()  # k nearest neighbors model from hw5
    socketio.emit('classify-status', {'message': 'Loaded!'})
    print('Loaded!')

    # Make prediction
    socketio.emit('classify-status', {'message': 'Classifying...'})
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

def knn_retrain(features, targets):
    ## NOTE: Look into incremental learning
    from numpy.random import permutation
    from sklearn.neighbors import KNeighborsClassifier
    from joblib import dump

    # Scramble data to remove (potential) dependence on ordering
    indices = permutation(len(targets))
    features = features[indices]
    targets = targets[indices]

    ##### Since the dataset is small, we will use all data
    # Define training and testing sets
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(all_features, 
    #                                                    all_targets, 
    #                                                     test_size=model_loader.TEST_PERCENT)
    
    
    # Train new model
    socketio.emit('retraining-status', {'message': 'Retraining model...'})
    print("Retraining model...")
    new_knn_model = KNeighborsClassifier(n_neighbors=model_loader.BEST_K)
    new_knn_model.fit(features, targets)
    socketio.emit('retraining-status', {'message': 'Trained!'})
    print("Trained!")
    
    # Save new model
    socketio.emit('retraining-status', {'message': 'Saving retrained model...'})
    print("Saving retrained model...")
    dump(new_knn_model, 'classifier/models/iris/knn_new.pkl')
    socketio.emit('retraining-status', {'message': 'Saved!'})
    print("Saved!")

    return new_knn_model


@iris_bp.route('/knn/retrain_and_visualize', methods=['POST'])
def knn_retrain_and_visualize():
    session['retrained'] = True

    socketio.emit('retraining-status', {'message': 'Obtaining data...'})
    print('Obtaining data...')

    db = get_db()
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

    socketio.emit('retraining-status', {'message': 'Obtained all data!'})
    print("Obtained all data!")

    new_model = knn_retrain(all_features, all_targets)

    # Create plots for visualization
    model_visualization(db, new_model)

    return render_template('iris/irisRetrainPlots.html', model='knn')

@iris_bp.route('/dtree', methods=['GET'])
def dtree_classifier():
    clean_files(classify='iris', model='dtree')

    db = get_db()
    corrections = db.execute(
        "SELECT sepallen, sepalwid, petallen, petalwid, species "
        "FROM iris "
        "WHERE model LIKE 'dtree'"
    ).fetchall()

    # Keep track if the model can be retrained or not
    if os.path.exists('classifier/models/iris/dtree_new.pkl'):
        session['retrained'] = True
    else:
        session['retrained'] = False

    return render_template('iris/dtree.html', 
                           corrections=corrections, 
                           index=SPECIES,
                           model='dtree')

def model_visualization(db, model):
    """Create images to visualize retrained models"""
    
    socketio.emit('retraining-status', {'message': 'Calculating averages...'})
    print("Calculating averages...")
    
    # Sums of feature values for the original iris dataset (rounded) 
    SEPALLEN_DATA_TOTAL = 876.5
    SEPALWID_DATA_TOTAL = 458.6
    PETALLEN_DATA_TOTAL = 563.7
    PETALWID_DATA_TOTAL = 458.6
    IRIS_DATA_TOTAL = 150  # number of irises in the iris dataset

    # Average feature values for corrections
    SEPALLEN_CORRS = db.execute(
        "SELECT SUM(sepallen) FROM iris "
        "WHERE model LIKE ?", ('knn', )
    ).fetchone()

    SEPALWID_CORRS = db.execute(
        "SELECT SUM(sepalwid) FROM iris "
        "WHERE model LIKE ?", ('knn', )
    ).fetchone()
    PETALLEN_CORRS = db.execute(
        "SELECT SUM(petallen) FROM iris "
        "WHERE model LIKE ?", ('knn', )
    ).fetchone()
    PETALWID_CORRS = db.execute(
        "SELECT SUM(petalwid) FROM iris "
        "WHERE model LIKE ?", ('knn', )
    ).fetchone()
    IRIS_CORR_TOTAL = db.execute(
        "SELECT COUNT(id) FROM iris "
        "WHERE model LIKE ?", ('knn', )
    ).fetchone()

    # Convert types
    SEPALLEN_CORRS = float(SEPALLEN_CORRS[0])
    SEPALWID_CORRS = float(SEPALWID_CORRS[0])
    PETALLEN_CORRS = float(PETALLEN_CORRS[0])
    PETALWID_CORRS = float(PETALWID_CORRS[0])
    IRIS_CORR_TOTAL = int(IRIS_CORR_TOTAL[0])

    ## We can only plot 2 dimensions at a time!
    from seaborn import set, heatmap
    from matplotlib import use
    from matplotlib.pyplot import clf

    # Select matplotlib backend to allow for plot creation
    use('agg')

    ##################
    ## Sepal Plane
    ##################
    
    # Get averages including corrections
    SEPALLEN_AVG = (SEPALLEN_DATA_TOTAL + SEPALLEN_CORRS) / (IRIS_DATA_TOTAL + IRIS_CORR_TOTAL)
    SEPALWID_AVG = (SEPALWID_DATA_TOTAL + SEPALWID_CORRS) / (IRIS_DATA_TOTAL + IRIS_CORR_TOTAL)

    VERTICAL = arange(0,8,.1) # array of vertical input values
    HORIZONT = arange(0,8,.1) # array of horizontal input values
    PLANE = zeros( (len(HORIZONT),len(VERTICAL)) ) # the output array

    socketio.emit('retraining-status', {'message': 'Working on sepal plane...'})
    print("Working on sepal plane...")

    col = 0
    row = 0
    for petalwid in VERTICAL: # for every petal width
        for petallen in HORIZONT: # for every petal length
            Features = [ SEPALLEN_AVG, SEPALWID_AVG, petallen, petalwid ]
            output = model.predict([Features])
            PLANE[row,col] = int(round(output[0]))
            row += 1
        row = 0
        col += 1

    set(rc = {'figure.figsize':(12,8)})  # figure size!
    sepal = heatmap(PLANE, cbar_kws={'ticks': [0, 1, 2]})
    sepal.invert_yaxis() # to match our usual direction
    sepal.set(xlabel="Petal Width (mm)", ylabel="Petal Length (mm)")
    sepal.set_title(
        f"Model Predictions with the Average Sepal Length ({SEPALLEN_AVG:.2f} cm) and Sepal Width ({SEPALWID_AVG:.2f} cm)")
    sepal.set_xticks(sepal.get_xticks()[::4])
    sepal.set_yticks(sepal.get_yticks()[::4])

    # Modify color bar
    cbar = sepal.collections[0].colorbar # Get color bar from heatmap
    cbar.set_label('Species', labelpad=-95)  # Set the label for the color bar
    cbar.set_ticklabels(['setosa', 'versicolor', 'virginica'])  # Set the tick labels

    # Save Plot
    sepal.get_figure().savefig('classifier/static/img/iris_knn_sepal_new.png')
    
    clf() # clear figure to create next plot


    ##################
    ## Petal Plane
    ##################

    # Get averages including corrections
    PETALLEN_AVG = (PETALLEN_DATA_TOTAL + PETALLEN_CORRS) / (IRIS_DATA_TOTAL + IRIS_CORR_TOTAL)
    PETALWID_AVG = (PETALWID_DATA_TOTAL + PETALWID_CORRS) / (IRIS_DATA_TOTAL + IRIS_CORR_TOTAL)

    socketio.emit('retraining-status', {'message': 'Working on petal plane...'})
    print("Working on petal plane...")
    
    col = 0
    row = 0
    for sepalwid in VERTICAL: # for every petal length
        for sepallen in HORIZONT: # for every sepal length
            Features = [ sepallen, sepalwid, PETALLEN_AVG, PETALWID_AVG ]
            output = model.predict([Features])
            PLANE[row,col] = int(round(output[0]))
            row += 1
        row = 0
        col += 1

    set(rc = {'figure.figsize':(12,8)})  # figure size!
    petal = heatmap(PLANE, cbar_kws={'ticks': [0, 1, 2]})
    petal.invert_yaxis() # to match our usual direction
    petal.set(xlabel="Sepal Width (mm)", ylabel="Sepal Length (mm)")
    petal.set_title(
        f"Model Predictions with the Average Petal Length ({PETALLEN_AVG:.2f} cm) and Petal Width ({PETALWID_AVG:.2f} cm)")
    petal.set_xticks(petal.get_xticks()[::4])
    petal.set_yticks(petal.get_yticks()[::4])

    # Modify color bar
    cbar = petal.collections[0].colorbar # Get color bar from heatmap
    cbar.set_label('Species', labelpad=-95)  # Set the label for the color bar
    cbar.set_ticklabels(['setosa', 'versicolor', 'virginica'])  # Set the tick labels

    # Save Plot
    petal.get_figure().savefig('classifier/static/img/iris_knn_petal_new.png')
    clf()

    ### NOTE: Look into using BytesIO and base64 for sending heatmaps to server
    socketio.emit('retraining-status', {'message': 'Created plots!'})
    print('Created plots!')