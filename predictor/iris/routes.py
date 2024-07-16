import os

from flask import Blueprint, render_template, request, jsonify, session, url_for

from predictor.iris import model_loader
from predictor.database.db import get_db
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
    db = get_db()
    corrections = db.execute(
        "SELECT sepallen, sepalwid, petallen, petalwid, species "
        "FROM iris "
        "WHERE model LIKE 'knn'"
    ).fetchall()

    # Keep track if the model can be retrained or not
    if os.path.exists('predictor/models/iris/knn_new.pkl'):
        session['retrained'] = True
    else:
        session['retrained'] = False

    return render_template('iris/knn.html', 
                           corrections=corrections, 
                           index=SPECIES,
                           model='knn')

@iris_bp.route('/knn/predict', methods=["POST"])
def knn_predict():
    print('loading model...')
    knn_model = model_loader.load_knn_model()  # k nearest neighbors model from hw5
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
    print("Training...")
    new_knn_model = KNeighborsClassifier(n_neighbors=model_loader.BEST_K)
    new_knn_model.fit(features, targets)
    print("Trained!")
    
    # Save new model
    print("Saving...")
    dump(new_knn_model, 'predictor/models/iris/knn_new.pkl')
    print("Saved!")

    return new_knn_model


@iris_bp.route('/knn/retrain_and_visualize', methods=['POST'])
def knn_retrain_and_visualize():
    session['retrained'] = True

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

    print("Obtained all data...")

    new_model = knn_retrain(all_features, all_targets)

    # Create plots for visualization
    
    model_visualization(db, new_model)

    return render_template('iris/irisRetrainPlots.html', model='knn')

def model_visualization(db, model):
    """Create images to visualize retrained models"""
    
    # Average feature values for the original iris dataset (rounded) 
    SEPALLEN_DATA_AVG = 5.86
    SEPALWID_DATA_AVG = 3.05
    PETALLEN_DATA_AVG = 7.90
    PETALWID_DATA_AVG = 7.90
    IRIS_DATA_TOTAL = 150  # number of irises in the iris dataset

    # Average feature values for corrections
    SEPALLEN_CORR_AVG = db.execute(
        "SELECT AVG(sepallen) FROM iris "
        "WHERE model LIKE ?", ('knn', )
    ).fetchone()

    SEPALWID_CORR_AVG = db.execute(
        "SELECT AVG(sepalwid) FROM iris "
        "WHERE model LIKE ?", ('knn', )
    ).fetchone()
    PETALLEN_CORR_AVG = db.execute(
        "SELECT AVG(petallen) FROM iris "
        "WHERE model LIKE ?", ('knn', )
    ).fetchone()
    PETALWID_CORR_AVG = db.execute(
        "SELECT AVG(petalwid) FROM iris "
        "WHERE model LIKE ?", ('knn', )
    ).fetchone()
    IRIS_CORR_TOTAL = db.execute(
        "SELECT COUNT(id) FROM iris "
        "WHERE model LIKE ?", ('knn', )
    ).fetchone()

    # Convert types
    SEPALLEN_CORR_AVG = float(SEPALLEN_CORR_AVG[0])
    SEPALWID_CORR_AVG = float(SEPALWID_CORR_AVG[0])
    PETALLEN_CORR_AVG = float(PETALLEN_CORR_AVG[0])
    PETALWID_CORR_AVG = float(PETALWID_CORR_AVG[0])
    IRIS_CORR_TOTAL = int(IRIS_CORR_TOTAL[0])

    ## We can only plot 2 dimensions at a time!
    from seaborn import set, heatmap

    ##################
    ## Sepal Plane
    ##################
    
    # Get averages including corrections
    SEPALLEN_AVG = (SEPALLEN_DATA_AVG + SEPALLEN_CORR_AVG) / (IRIS_DATA_TOTAL + IRIS_CORR_TOTAL)
    SEPALWID_AVG = (SEPALWID_DATA_AVG + SEPALWID_CORR_AVG) / (IRIS_DATA_TOTAL + IRIS_CORR_TOTAL)

    VERTICAL = arange(0,8,.1) # array of vertical input values
    HORIZONT = arange(0,8,.1) # array of horizontal input values
    PLANE = zeros( (len(HORIZONT),len(VERTICAL)) ) # the output array

    col = 0
    row = 0
    for petalwid in VERTICAL: # for every sepal length
        for petallen in HORIZONT: # for every sepal width
            Features = [ SEPALLEN_AVG, SEPALWID_AVG, petallen, petalwid ]
            output = model.predict([Features])
            PLANE[row,col] = int(round(output[0]))
            row += 1
        row = 0
        col += 1

    # Create heatmap
    set(rc = {'figure.figsize':(12,8)})  # figure size!
    ax = heatmap(PLANE, cbar_kws={'ticks': [0, 1, 2]})
    ax.invert_yaxis() # to match our usual direction
    ax.set(xlabel="Petal Width (mm)", ylabel="Petal Length (mm)")
    ax.set_title(
        f"Model Predictions with the Average Sepal Length ({SEPALLEN_AVG:.2f} cm) and Sepal Width ({SEPALWID_AVG:.2f} cm)")
    ax.set_xticks(ax.get_xticks()[::4])
    ax.set_yticks(ax.get_yticks()[::4])

    # Modify color bar
    cbar = ax.collections[0].colorbar # Get color bar from heatmap
    cbar.set_label('Species', labelpad=-95)  # Set the label for the color bar
    cbar.set_ticklabels(['setosa', 'versicolor', 'virginica'])  # Set the tick labels

    # Save Plot
    ax.get_figure().savefig(url_for('static', filename='iris_knn_petal_new'))


    ##################
    ## Petal Plane
    ##################
    
    # Get averages including corrections
    PETALLEN_AVG = (PETALLEN_DATA_AVG + PETALLEN_CORR_AVG) / (IRIS_DATA_TOTAL + IRIS_CORR_TOTAL)
    PETALWID_AVG = (PETALWID_DATA_AVG + PETALWID_CORR_AVG) / (IRIS_DATA_TOTAL + IRIS_CORR_TOTAL)

    col = 0
    row = 0
    for sepalwid in VERTICAL: # for every sepal length
        for sepallen in HORIZONT: # for every sepal width
            Features = [ sepalwid, sepallen, PETALLEN_AVG, PETALWID_AVG ]
            output = model.predict([Features])
            PLANE[row,col] = int(round(output[0]))
            row += 1
        row = 0
        col += 1

    # Create heatmap
    set(rc = {'figure.figsize':(12,8)})  # figure size!
    ax = heatmap(PLANE, cbar_kws={'ticks': [0, 1, 2]})
    ax.invert_yaxis() # to match our usual direction
    ax.set(xlabel="Petal Width (mm)", ylabel="Petal Length (mm)")
    ax.set_title(
        f"Model Predictions with the Average Sepal Length ({PETALLEN_AVG:.2f} cm) and Sepal Width ({PETALWID_AVG:.2f} cm)")
    ax.set_xticks(ax.get_xticks()[::4])
    ax.set_yticks(ax.get_yticks()[::4])

    # Modify color bar
    cbar = ax.collections[0].colorbar # Get color bar from heatmap
    cbar.set_label('Species', labelpad=-95)  # Set the label for the color bar
    cbar.set_ticklabels(['setosa', 'versicolor', 'virginica'])  # Set the tick labels

    # Save Plot
    ax.get_figure().savefig(url_for('static', filename='img/iris_knn_petal_new'))