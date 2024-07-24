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

iris_bp = Blueprint('iris', __name__)

# Keep track of species/features indices
SPECIES = ['setosa','versicolor','virginica']   # int to str
SPECIES_INDEX = {'setosa':0,'versicolor':1,'virginica':2}  # str to int
FEATURES = ['sepallen', 'sepalwid', 'petallen', 'petalwid']
FEATURES_INDEX = {'sepallen':0,'sepalwid':1,'petallen':2,'petalwid':3}

@iris_bp.route('/')
def iris():
    return render_template('iris/home.html')


###################
## KNN FUNCTIONS
###################
@iris_bp.route('/knn', methods=["GET"])
def knn_classifier():
    """View the page for the k-nearest neighbors iris classifier"""

    # Clear knn models/plots created from past use
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
                           instancePlotExists=False)

@iris_bp.route('/knn/predict', methods=["POST"])
def knn_predict():
    """
    Make an iris species prediction using the KNN model.
    
    Also creates heatmap visualizations for the given features
    submitted on the web form if specified on the web page
    """
    # Load model
    socketio.emit('classify-status', {'message': 'Loading model...'})
    print('Loading model...')
    sleep(0.1)
    knn_model = model_loader.load_knn_model()  # k nearest neighbors model from hw5
    socketio.emit('classify-status', {'message': 'Loaded!'})
    print('Loaded!')

    # Get data
    socketio.emit('classify-status', {'message': 'Classifying...'})
    data = request.json

    # Determine if we need to plot
    last_key = list(data.keys())[-1]
    visualize =  data.pop(last_key)

    # Store data in session to display on form
    session['iris_features'] = data

    # Make prediction
    Features = asarray([data['sepallen'],
                data['sepalwid'],
                data['petallen'],
                data['petalwid']],
                dtype=float)

    prediction = knn_model.predict([Features])
    prediction = int(round(prediction[0]))  # unpack the extra brackets
    prediction = SPECIES[prediction]  # change to string
    
    if visualize:
        heatmap_visualization(knn_model, Features)
        return jsonify({"species": prediction, 
                        "images": render_template('iris/irisInstancePlots.html',
                                                    instancePlotExists=True)})
    
    return jsonify({"species": prediction})

@iris_bp.route('/knn/retrain_and_visualize', methods=['POST'])
def knn_retrain_and_visualize():
    """
    Retrains a KNN model and creates heatmaps to visualize the new model

    Takes all original iris data and all corrections for the KNN model
    to retrain the KNN model
    """
    session['retrained'] = True

    # Get data
    features, targets = get_all_data('knn')

    # Retrain model
    new_model = knn_retrain(features, targets)

    # Create plots for visualization
    image_paths = heatmap_visualization(new_model)

    return render_template('iris/irisRetrainPlots.html', images = image_paths)

def knn_retrain(features, targets):
    """
    Trains a new KNN model

    Parameters: features - feature data to train on
                targets - target data corresponding to the feature data

    Returns:    The new model
    """

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

#####################
## DTREE FUNCTIONS
#####################
@iris_bp.route('/dtree', methods=['GET'])
def dtree_classifier():
    """View the page for the decision tree iris classifier"""

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
                           index=SPECIES)

@iris_bp.route('/dtree/predict', methods=['POST'])
def dtree_predict():
    """
    Make an iris species prediction using the decision tree.
    
    Also creates tree plot visualizations for the given features
    submitted on the web form if specified on the web page
    """
    # Load model
    socketio.emit('classify-status', {'message': 'Loading model...'})
    print('Loading model...')
    sleep(0.1)
    dtree = model_loader.load_dtree_model()  # k nearest neighbors model from hw5
    socketio.emit('classify-status', {'message': 'Loaded!'})
    print('Loaded!')

    # Get data
    socketio.emit('classify-status', {'message': 'Classifying...'})
    data = request.json

    # Determine if we need to plot
    last_key = list(data.keys())[-1]
    visualize =  data.pop(last_key)

    # Store data in session to display on form
    session['iris_features'] = data

    # Make prediction
    Features = asarray([data['sepallen'],
                data['sepalwid'],
                data['petallen'],
                data['petalwid']],
                dtype=float)

    prediction = dtree.predict([Features])
    prediction = int(round(prediction[0]))  # unpack the extra brackets
    prediction = SPECIES[prediction]  # change to string
    
    if visualize:
        heatmap_visualization(dtree, Features)
        return jsonify({"species": prediction, 
                        "images": render_template('iris/irisInstancePlots.html', 
                                                    instancePlotExists=True)})
    
    return jsonify({"species": prediction})

@iris_bp.route('/dtree/retrain_and_visualize', methods=['POST'])
def dtree_retrain_and_visualize():
    """
    Retrains a dtree model and creates a tree plot to visualize it

    Takes all original iris data and all corrections for the dtree model
    to retrain it
    """
    session['retrained'] = True

    # Get data
    features, targets = get_all_data('dtree')

    # Retrain model
    new_model = dtree_retrain(features, targets)

    # Create plots for visualization
    image_path = dtree_visualization(new_model)

    return render_template('iris/irisRetrainPlots.html', images=image_path)

def dtree_retrain(features, targets):
    from numpy.random import permutation
    from sklearn.tree import DecisionTreeClassifier
    from joblib import dump

    # Scramble data to remove (potential) dependence on ordering
    indices = permutation(len(targets))
    features = features[indices]
    targets = targets[indices]

    ##### Since the dataset is small, we will use all data
    # Define training and testing sets
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(all_features, 
    #                                                     all_targets, 
    #                                                     test_size=model_loader.TEST_PERCENT)
    
    
    # Train new model
    socketio.emit('retraining-status', {'message': 'Retraining model...'})
    print("Retraining model...")
    new_dtree_model = DecisionTreeClassifier(max_depth=model_loader.BEST_DEPTH)
    new_dtree_model.fit(features, targets)
    socketio.emit('retraining-status', {'message': 'Trained!'})
    print("Trained!")
    
    # Save new model
    socketio.emit('retraining-status', {'message': 'Saving retrained model...'})
    print("Saving retrained model...")
    dump(new_dtree_model, 'classifier/models/iris/dtree_new.pkl')
    socketio.emit('retraining-status', {'message': 'Saved!'})
    print("Saved!")

    return new_dtree_model

def dtree_visualization(model):
    from sklearn.tree import plot_tree
    from matplotlib.pyplot import figure, clf
    from matplotlib import use

    use('agg')

    socketio.emit('retraining-status', {'message': 'Creating plot...'})
    print('Creating plot...')

    fig = figure(figsize=(10,10))
    plot_tree(model, feature_names=FEATURES,
                class_names=SPECIES,
                filled=True)
    
    # Save figure
    filename = "img/iris_dtree.png"
    fig.savefig(f'classifier/static/{filename}', bbox_inches='tight')
    clf()

    socketio.emit('retraining-status', {'message': 'Created plot!'})
    print('Created plot!')

    return [filename]

#####################
## GENERAL FUNCTIONS
#####################
def get_averages(model):
    """Calculate average feature values for all data for a model"""

    socketio.emit('retraining-status', {'message': 'Calculating averages...'})
    print("Calculating averages...")
    
    # Sums of feature values for the original iris dataset (rounded) 
    SEPALLEN_DATA_TOTAL = 876.5
    SEPALWID_DATA_TOTAL = 458.6
    PETALLEN_DATA_TOTAL = 563.7
    PETALWID_DATA_TOTAL = 458.6
    IRIS_DATA_TOTAL = 150  # number of irises in the iris dataset

    db = get_db()
    
    # Average feature values for corrections
    SEPALLEN_CORRS = db.execute(
        "SELECT SUM(sepallen) FROM iris "
        "WHERE model LIKE ?", (model, )
    ).fetchone()

    SEPALWID_CORRS = db.execute(
        "SELECT SUM(sepalwid) FROM iris "
        "WHERE model LIKE ?", (model, )
    ).fetchone()
    PETALLEN_CORRS = db.execute(
        "SELECT SUM(petallen) FROM iris "
        "WHERE model LIKE ?", (model, )
    ).fetchone()
    PETALWID_CORRS = db.execute(
        "SELECT SUM(petalwid) FROM iris "
        "WHERE model LIKE ?", (model, )
    ).fetchone()
    IRIS_CORR_TOTAL = db.execute(
        "SELECT COUNT(id) FROM iris "
        "WHERE model LIKE ?", (model, )
    ).fetchone()

    # Convert types
    SEPALLEN_CORRS = float(SEPALLEN_CORRS[0])
    SEPALWID_CORRS = float(SEPALWID_CORRS[0])
    PETALLEN_CORRS = float(PETALLEN_CORRS[0])
    PETALWID_CORRS = float(PETALWID_CORRS[0])
    IRIS_CORR_TOTAL = int(IRIS_CORR_TOTAL[0])

    # Calculate averages
    SEPALLEN_AVG = (SEPALLEN_DATA_TOTAL + SEPALLEN_CORRS) / (IRIS_DATA_TOTAL + IRIS_CORR_TOTAL)
    SEPALWID_AVG = (SEPALWID_DATA_TOTAL + SEPALWID_CORRS) / (IRIS_DATA_TOTAL + IRIS_CORR_TOTAL)
    PETALLEN_AVG = (PETALLEN_DATA_TOTAL + PETALLEN_CORRS) / (IRIS_DATA_TOTAL + IRIS_CORR_TOTAL)
    PETALWID_AVG = (PETALWID_DATA_TOTAL + PETALWID_CORRS) / (IRIS_DATA_TOTAL + IRIS_CORR_TOTAL)

    return SEPALLEN_AVG, SEPALWID_AVG, PETALLEN_AVG, PETALWID_AVG

def get_all_data(model):
    """
    Gets all available iris data.

    Includes the base iris dataset and any corrections for model
    found in the database
    """
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
        "WHERE model LIKE ?", (model, )
    ).fetchall()
    new_targets = db.execute(
        "SELECT species FROM iris "
        "WHERE model LIKE ?", (model, )
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

    return all_features, all_targets

@iris_bp.route('/clear_corrections', methods = ['POST'])
def clear_corrections():
    """
    Clears all corrections for a specified model

    Deletes all data for the specified model in the database
    and updates the web page's correction list 
    """
    data = request.json
    model = data['model']
    
    socketio.emit('clear-status', {'message': 'Clearing data...'})
    print("Clearing data...")

    # Clear data for given model
    db = get_db()
    db.execute(
        "DELETE FROM iris "
        "WHERE model LIKE ?", (model, )
    )
    db.commit()

    # Delete retrained model and plots (since there are no more corrections)
    retrained_model_path = f'classifier/models/iris/{model}_new.pkl'
    if os.path.exists(retrained_model_path):
        os.remove(retrained_model_path)
        os.remove(f'classifier/static/img/iris_{model}_sepal_new.png')
        os.remove(f'classifier/static/img/iris_{model}_petal_new.png')

    session['retrained'] = False


    socketio.emit('clear-status', {'message': 'Cleared!'})
    print("Cleared!")
    sleep(0.5)

    return jsonify({"corrections": render_template('corrections.html'),
                    "retrain_plots": render_template('iris/irisRetrainPlots.html')})

@iris_bp.route('/incorrect', methods=["POST"])
def incorrect():
    """
    Handles corrections submitted by user for when the model is wrong.

    Stores the corrections in the database (with model specified) and
    updates the corrections list seen on the web page.
    """
    # Get species index
    data = request.json
    species = SPECIES_INDEX[data['correction']]

    print(data)

    # Get model
    model = data['model']
    
    # Get inputted features from session
    iris_features = session['iris_features']

    # Add correction to database
    db = get_db()

    db.execute(
        "INSERT INTO iris (sepallen, sepalwid, petallen, petalwid, species, model)"
        "VALUES (?, ?, ?, ?, ?, ?)",
        (iris_features['sepallen'], iris_features['sepalwid'], 
         iris_features['petallen'], iris_features['petalwid'], 
         species, model)
    )
    db.commit()

    # Get all new corrections to update corrections.html
    new_corrections = db.execute(
        "SELECT sepallen, sepalwid, petallen, petalwid, species "
        "FROM iris "
        "WHERE model LIKE ?", (model, )
    ).fetchall()

    # Model can now be retrained
    session['retrained'] = False

    # Convert to list of dicts for JSON serialization
    all_corrections = [dict(row) for row in new_corrections]

    return render_template('corrections.html', 
                           corrections=all_corrections, 
                           index=SPECIES)

def heatmap_visualization(model, features = None):
    """
    Creates heatmaps to visualize models
    
    There are two heatmaps: the sepal and petal plane.
    The sepal plane keeps the sepal features constant, while
    the petal plane keeps the petal features constant.

    By default, these planes use the average sepal/petal length
    and width for constant values. However, if the features
    parameter is specified, it will use those instead.
    """
    
    # Handle correctly sending status messages for socketio
    if features is not None:
        status = 'classify-status'
    else:
        status = 'retraining-status'

    if features is not None:
        plane_sepallen = features[0]
        plane_sepalwid = features[1]
        plane_petallen = features[2] 
        plane_petalwid = features[3]
    else:
        plane_sepallen, plane_sepalwid, plane_petallen, plane_petalwid = get_averages('knn')

    ## We can only plot 2 dimensions at a time!
    from seaborn import set_theme, heatmap
    from matplotlib import use
    from matplotlib.pyplot import clf, figure
    from matplotlib.patches import Rectangle


    # Select matplotlib backend to allow for plot creation
    use('agg')

    ####### Sepal Plane #######

    VERTICAL = arange(0,8,.1) # array of vertical input values
    HORIZONT = arange(0,8,.1) # array of horizontal input values
    PLANE = zeros( (len(HORIZONT),len(VERTICAL)) ) # the output array

    socketio.emit(status, {'message': 'Working on sepal plane...'})
    print("Working on sepal plane...")

    col = 0
    row = 0
    for petalwid in VERTICAL: # for every petal width
        for petallen in HORIZONT: # for every petal length
            Features = [ plane_sepallen, plane_sepalwid, petallen, petalwid ]
            output = model.predict([Features])
            PLANE[row,col] = int(round(output[0]))
            row += 1
        row = 0
        col += 1

    figure(figsize=(12,8))
    sepal = heatmap(PLANE, cbar_kws={'ticks': [0, 1, 2]})
    sepal.invert_yaxis() # to match our usual direction
    sepal.set(xlabel="Petal Width (mm)", ylabel="Petal Length (mm)")

    sepal.set_xticks(sepal.get_xticks()[::4])
    sepal.set_yticks(sepal.get_yticks()[::4])
    if features is not None:
        sepal.set_title(
            f"Model Predictions with Sepal Length {plane_petallen:.2f} cm and Sepal Width {plane_petalwid:.2f} cm")
    else:
        sepal.set_title(
            f"Model Predictions with the Average Sepal Length ({plane_sepallen:.2f} cm) and Sepal Width ({plane_sepalwid:.2f} cm)")

    # Modify color bar
    cbar = sepal.collections[0].colorbar # Get color bar from heatmap
    cbar.set_label('Species', labelpad=-95)  # Set the label for the color bar
    cbar.set_ticklabels(['setosa', 'versicolor', 'virginica'])  # Set the tick labels

    # Highight inputted iris features from web form
    if features is not None:
        highlight_x, highlight_y = plane_petalwid*10, plane_petallen*10

        # Make sure features are on the heatmaps
        if highlight_x < 80 and highlight_y < 80:
            rect = Rectangle((highlight_x, highlight_y), 1, 1, linewidth=2, edgecolor='#e399f2', facecolor='none')
            sepal.add_patch(rect)

            # Add text
            sepal.text(highlight_x, highlight_y - 2, 'Your iris', color='#e399f2', ha='center', va='center', fontsize=16)
        else:
            # Tell user their iris is not on the plot (message shown in center of plot)
            sepal.text(40, 40, 'Iris outside of plot', 
                       color='#e399f2', ha='center', va='center', fontsize=16)


    # Save Plot
    if features is not None:
        sepal_path = 'img/this_iris_sepal.png'
        sepal.get_figure().savefig(f'classifier/static/{sepal_path}')
    else:
        sepal_path = 'img/iris_avg_sepal_new.png'
        sepal.get_figure().savefig(f'classifier/static/{sepal_path}')
    
    clf() # clear figure to create next plot


    ####### Petal Plane #######

    socketio.emit(status, {'message': 'Working on petal plane...'})
    print("Working on petal plane...")
    
    col = 0
    row = 0
    for sepalwid in VERTICAL: # for every petal length
        for sepallen in HORIZONT: # for every sepal length
            Features = [ sepallen, sepalwid, plane_petallen, plane_petalwid ]
            output = model.predict([Features])
            PLANE[row,col] = int(round(output[0]))
            row += 1
        row = 0
        col += 1

    figure(figsize=(12,8))
    petal = heatmap(PLANE, cbar_kws={'ticks': [0, 1, 2]})
    petal.invert_yaxis() # to match our usual direction
    petal.set(xlabel="Sepal Width (mm)", ylabel="Sepal Length (mm)")
    petal.set_xticks(petal.get_xticks()[::4])
    petal.set_yticks(petal.get_yticks()[::4])
    if features is not None:
        petal.set_title(
            f"Model Predictions with Petal Length {plane_petallen:.2f} cm and Petal Width {plane_petalwid:.2f} cm")
    else:
        petal.set_title(
            f"Model Predictions with the Average Petal Length ({plane_petallen:.2f} cm) and Petal Width ({plane_petalwid:.2f} cm)")

    # Highight inputted iris features from web form
    if features is not None:
        highlight_x, highlight_y = plane_sepalwid*10, plane_sepallen*10

        # Make sure features are on the heatmaps
        if highlight_x < 80 and highlight_y < 80:
            rect = Rectangle((highlight_x, highlight_y), 1, 1, linewidth=2, edgecolor='#e399f2', facecolor='none')
            petal.add_patch(rect)

            # Add text
            petal.text(highlight_x, highlight_y - 2, 'Your iris', color='#e399f2', ha='center', va='center', fontsize=16)
        else:
            # Tell user their iris is not on the plot (message shown in center of plot)
            petal.text(40, 40, 'Iris outside of plot', 
                        color='#e399f2', ha='center', va='center', fontsize=16)
    # Modify color bar
    cbar = petal.collections[0].colorbar # Get color bar from heatmap
    cbar.set_label('Species', labelpad=-95)  # Set the label for the color bar
    cbar.set_ticklabels(['setosa', 'versicolor', 'virginica'])  # Set the tick labels

    # Save Plot
    if features is not None:
        petal_path = 'img/this_iris_petal.png'
        petal.get_figure().savefig(f'classifier/static/{petal_path}')
    else:
        petal_path = 'img/iris_avg_petal_new.png'
        petal.get_figure().savefig(f'classifier/static/{petal_path}')
    clf()

    ### NOTE: Look into using BytesIO and base64 for sending heatmaps to server
    socketio.emit('retraining-status', {'message': 'Created plots!'})
    print('Created plots!')

    return [sepal_path, petal_path]
