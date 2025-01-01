import os
from time import sleep

from flask import (
    Blueprint, render_template, request, jsonify, session)

from classifier import socketio  # to give loading messages when clicking buttons

from classifier.commands import clean_files
from classifier.cancer import model_loader
from classifier.database.db import get_db
import numpy as np
from sklearn.datasets import load_breast_cancer
from matplotlib import use
from matplotlib.pyplot import suptitle, tight_layout, subplots, clf
from seaborn import heatmap
from matplotlib.patches import Patch, Rectangle

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
    clean_files(classify='cancer', model='knn')

    db = get_db()
    corrections = db.execute(
        "SELECT * "
        "FROM cancer_data "
        "WHERE model LIKE 'knn'"
    ).fetchall()

    # Keep track if the model can be retrained or not
    if os.path.exists('classifier/models/cancer/knn_new.pkl'):
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
    Features = np.asarray([data['radius'],
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
        heatmap_visualization(knn_model, get_all_data('knn', 'classify-status')[0], Features)
        return jsonify({"cell_type": prediction, 
                        "images": render_template('cancer/cancerInstancePlots.html',
                                                    instancePlotExists=True)})
    
    return jsonify({"cell_type": prediction})

@cancer_bp.route('/knn/retrain_and_visualize', methods=['POST'])
def knn_retrain_and_visualize():
    """
    Retrains a KNN model and creates heatmaps to visualize the new model

    Takes all original breast ancer data and all corrections for the KNN model
    to retrain the KNN model
    """
    session['retrained'] = True

    # Get data
    features, targets = get_all_data('knn', 'retraining-status')

    # Retrain model
    new_model = knn_retrain(features, targets)

    # Create plots for visualization
    image_paths = heatmap_visualization(new_model, features)

    return render_template('cancer/cancerRetrainPlots.html', images = image_paths)

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
    # X_train, X_test, y_train, y_test = train_test_split(all_features, all_targets)
    
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
    dump(new_knn_model, 'classifier/models/cancer/knn_new.pkl')
    socketio.emit('retraining-status', {'message': 'Saved!'})
    print("Saved!")

    return new_knn_model


#####################
## DTREE FUNCTIONS
#####################
@cancer_bp.route('/dtree', methods=['GET'])
def dtree_classifier():
    """View the page for the breast cancer decision tree classifier"""

    clean_files(classify='cancer', model='dtree')

    db = get_db()
    corrections = db.execute(
        "SELECT * "
        "FROM cancer_data "
        "WHERE model LIKE 'dtree'"
    ).fetchall()

    # Keep track if the model can be retrained or not
    if os.path.exists('classifier/models/cancer/dtree_new.pkl'):
        session['retrained'] = True
    else:
        session['retrained'] = False

    return render_template('cancer/dtree.html', 
                           corrections=corrections, 
                           index=TARGET)

@cancer_bp.route('/dtree/predict', methods=['POST'])
def dtree_predict():
    """
    Make a breast cancer prediction using the decision tree.
    
    Also creates tree plot visualizations for the given features
    submitted on the web form if specified on the web page
    """
    # Load model
    socketio.emit('classify-status', {'message': 'Loading model...'})
    print('Loading model...')
    sleep(0.1)
    dtree = model_loader.load_dtree_model()
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
    Features = np.asarray([data['radius'],
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

    prediction = dtree.predict([Features])
    prediction = int(round(prediction[0]))  # unpack the extra brackets
    prediction = TARGET[prediction]  # change to string
    
    if visualize:
        heatmap_visualization(dtree, get_all_data('dtree', 'classify-status')[0], Features)
        return jsonify({"cell_type": prediction, 
                        "images": render_template('cancer/cancerInstancePlots.html', 
                                                    instancePlotExists=True)})
    
    return jsonify({"cell_type": prediction})

@cancer_bp.route('/dtree/retrain_and_visualize', methods=['POST'])
def dtree_retrain_and_visualize():
    """
    Retrains a dtree model and creates a tree plot to visualize it

    Takes all original cancer data and all corrections for the dtree model
    to retrain it
    """
    session['retrained'] = True

    # Get data
    features, targets = get_all_data('dtree', 'retraining-status')

    # Retrain model
    new_model = dtree_retrain(features, targets)

    # Create plots for visualization
    image_path = dtree_visualization(new_model)

    return render_template('cancer/cancerRetrainPlots.html', images=image_path)

def dtree_retrain(features, targets):
    from numpy.random import permutation
    from sklearn.tree import DecisionTreeClassifier
    from joblib import dump

    # Scramble data to remove (potential) dependence on ordering
    indices = permutation(len(targets))
    features = features[indices]
    targets = targets[indices]    
    
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
    dump(new_dtree_model, 'classifier/models/cancer/dtree_new.pkl')
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

    fig = figure(figsize=(25,20))
    plot_tree(model, feature_names=FEATURES,
                class_names=TARGET,
                filled=True)
    
    # Save figure
    filename = "img/cancer_dtree.png"
    fig.savefig(f'classifier/static/{filename}', bbox_inches='tight')
    clf()

    socketio.emit('retraining-status', {'message': 'Created plot!'})
    print('Created plot!')

    return [filename]


###################
### MLP FUNCIONS
###################
@cancer_bp.route('/mlp', methods=['GET'])
def mlp_classifier():
    """View the page for the multilayer perceptron breast cancer classifier"""

    clean_files(classify='cancer', model='mlp')

    db = get_db()
    corrections = db.execute(
        "SELECT * "
        "FROM cancer_data "
        "WHERE model LIKE 'mlp'"
    ).fetchall()

    # Keep track if the model can be retrained or not
    if os.path.exists('classifier/models/cancer/mlp_new.pkl'):
        session['retrained'] = True
    else:
        session['retrained'] = False

    return render_template('cancer/mlp.html', 
                           corrections=corrections, 
                           index=TARGET)

@cancer_bp.route('/mlp/predict', methods=['POST'])
def mlp_predict():
    """
    Make an breast cancer prediction using the multilayer perceptron.
    
    Also creates a heatmap visualization for the given features
    submitted on the web form if specified on the web page
    """
    # Load model
    socketio.emit('classify-status', {'message': 'Loading model...'})
    print('Loading model...')
    sleep(0.1)
    mlp = model_loader.load_mlp_model()
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
    Features = np.asarray([data['radius'],
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

    prediction = mlp.predict([Features])
    prediction = int(round(prediction[0]))  # unpack the extra brackets
    prediction = TARGET[prediction]  # change to string
    
    if visualize:
        heatmap_visualization(mlp, get_all_data('mlp', 'classify-status')[0], Features)
        return jsonify({"cell_type": prediction, 
                        "images": render_template('cancer/cancerInstancePlots.html', 
                                                    instancePlotExists=True)})
    
    return jsonify({"cell_type": prediction})


@cancer_bp.route('/mlp/retrain_and_visualize', methods=['POST'])
def mlp_retrain_and_visualize():
    """
    Retrains an mlp model and creates a heatmap to visualize it

    Takes all original breast cancer data and all corrections for the mlp model
    to retrain it
    """
    session['retrained'] = True

    # Get data
    features, targets = get_all_data('mlp', 'retraining-status')

    # Retrain model
    new_model = mlp_retrain(features, targets)

    # Create plots for visualization
    image_path = heatmap_visualization(new_model, features)

    return render_template('cancer/cancerRetrainPlots.html', images=image_path)

def mlp_retrain(features, targets):
    from numpy.random import permutation
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from joblib import dump

    # Scramble data to remove (potential) dependence on ordering
    indices = permutation(len(targets))
    features = features[indices]
    targets = targets[indices]

    # Define training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, 
                                                        targets, 
                                                        test_size=model_loader.TEST_PERCENT)
    
    # Scale data
    socketio.emit('retraining-status', {'message': 'Scaling data...'})
    print("Scaling data...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    
    # Train new model
    socketio.emit('retraining-status', {'message': 'Retraining model...'})
    print("Retraining model...")

    # Same settings as base model
    new_mlp_model = MLPClassifier(hidden_layer_sizes=(6,7),
                                  max_iter=500,
                                  shuffle=True,
                                  learning_rate_init=1,
                                  learning_rate='adaptive')
    
    new_mlp_model.fit(X_train_scaled, y_train)
    socketio.emit('retraining-status', {'message': 'Trained!'})
    print("Trained!")
    
    # Save new model
    socketio.emit('retraining-status', {'message': 'Saving retrained model...'})
    print("Saving retrained model...")
    dump(new_mlp_model, 'classifier/models/cancer/mlp_new.pkl')
    socketio.emit('retraining-status', {'message': 'Saved!'})
    print("Saved!")

    return new_mlp_model


#####################
## GENERAL FUNCTIONS
#####################

@cancer_bp.route('/incorrect', methods=["POST"])
def incorrect():
    """
    Handles corrections submitted by user for when the model is wrong.

    Stores the corrections in the database (with model specified) and
    updates the corrections list seen on the web page.
    """
    # Get target indices (malignant, benign)
    data = request.json
    cell_type = TARGET_INDEX[data['correction']]

    # Get model
    model = data['model']
    
    # Get inputted features from session
    cancer_features = session['cancer_features']

    # Add correction to database
    db = get_db()

    db.execute(
        "INSERT INTO cancer (radius, texture, perimeter, area, smoothness, "
        "compactness, concavity, concave_points, symmetry, fractal, "
        "cell_type, model)"
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (cancer_features['radius'], cancer_features['texture'], 
         cancer_features['perimeter'], cancer_features['area'], 
         cancer_features['smoothness'], cancer_features['compactness'], 
         cancer_features['concavity'], cancer_features['concave_points'], 
         cancer_features['symmetry'], cancer_features['fractal_dimension'], 
         cell_type, model)
    )
    db.commit()

    # Get all new corrections to update corrections.html
    new_corrections = db.execute(
        "SELECT * "
        "FROM cancer_data "
        "WHERE model LIKE ?", (model, )
    ).fetchall()

    # Model can now be retrained
    session['retrained'] = False

    # Convert to list of dicts for JSON serialization
    all_corrections = [dict(row) for row in new_corrections]

    return render_template('cancer/corrections.html', 
                           corrections=all_corrections, 
                           index=TARGET)

@cancer_bp.route('/clear_corrections', methods = ['POST'])
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
        "DELETE FROM cancer "
        "WHERE model LIKE ?", (model, )
    )
    db.commit()

    # Delete retrained model and plots (since there are no more corrections)
    retrained_model_path = f'classifier/models/cancer/{model}_new.pkl'
    if os.path.exists(retrained_model_path):
        os.remove(retrained_model_path)
        os.remove(f'classifier/static/img/cancer_{model}_new.png')

    session['retrained'] = False


    socketio.emit('clear-status', {'message': 'Cleared!'})
    print("Cleared!")
    sleep(0.5)

    return jsonify({"corrections": render_template('cancer/corrections.html'),
                    "retrain_plots": render_template('cancer/cancerRetrainPlots.html')})


def get_all_data(model, status):
    """
    Gets all available cancer data.

    Includes the base cancer dataset and any corrections for model
    found in the database
    """
    socketio.emit(status, {'message': 'Obtaining data...'})
    print('Obtaining data...')

    db = get_db()
    cancer_data = load_breast_cancer()

    # Get pre-existing feature and target data from dataset
    feature_data = cancer_data['data'][:, :10] # only get the data for means
    target_data = cancer_data['target']

    # Get new features and target data from database
    new_features = db.execute(
        "SELECT radius, texture, perimeter, area, smoothness, "
        "compactness, concavity, concave_points, symmetry, fractal "
        "FROM cancer "
        "WHERE model LIKE ?", (model, )
    ).fetchall()
    new_targets = db.execute(
        "SELECT cell_type FROM cancer "
        "WHERE model LIKE ?", (model, )
    ).fetchall()

    # Format as numpy array
    new_features = np.array(new_features, dtype='float64')
    new_targets = np.array(new_targets, dtype='float64')
    new_targets = np.reshape(new_targets, -1)  # convert to 1D array

    ### Combine all features and target data
    #
    # if there's no new features, all_features is just feature_data
    # NOTE: trying to concatenate when there are no new features gives an error
    all_features = np.concatenate((feature_data, new_features)) if new_features.size != 0 else feature_data
    all_targets = np.concatenate((target_data, new_targets))

    socketio.emit(status, {'message': 'Obtained all data!'})
    print("Obtained all data!")

    return all_features, all_targets

def heatmap_visualization(model, data, features = None):
    """
    Creates a 5x5 grid of pairwise heatmaps to visualize models,
    thus 25 planes are visualized
    
    Certain features were selected for the x- and y-axes:
    x-axis: mean radius, mean perimeter, mean area, mean texture, mean smoothness
    y-axis: mean compactness, mean concavity, mean concave points, mean symmetry, mean fractal dimension

    By default, each plane uses the average values as constants for the features that 
    aren't on the axis.
    If the features parameter is specified, it will use those instead.
    """
    
    if features is not None:
         # Handle correctly sending status messages for socketio
        status = 'classify-status'
    else:
         # Handle correctly sending status messages for socketio
        status = 'retraining-status'

    socketio.emit(status, {'message': 'Working on heatmaps...'})
    print("Working on heatmaps...")


    # Define axes features
    x_features = ['mean radius', 'mean perimeter', 'mean area', 'mean texture', 'mean smoothness']
    y_features = ['mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension']

    # Find indices of the features
    x_indices = [list(FEATURES).index(f) for f in x_features]
    y_indices = [list(FEATURES).index(f) for f in y_features]


    # Select matplotlib backend to allow for plot creation
    use('agg')

    # Generate pairwise plots
    nrows, ncols = len(y_features), len(x_features)
    cancer, axes = subplots(nrows, ncols, figsize=(15, 12), sharex='col', sharey='row')
    grid_resolution = 100

    # Create a grid of plots
    for i, y_idx in enumerate(y_indices):
        for j, x_idx in enumerate(x_indices):
            
            # Get feature ranges
            x_min, x_max = data[:, x_idx].min(), data[:, x_idx].max()
            y_min, y_max = data[:, y_idx].min(), data[:, y_idx].max()
            x_range = np.linspace(x_min, x_max, grid_resolution)
            y_range = np.linspace(y_min, y_max, grid_resolution)
            
            # Generate grid for the selected features
            grid_x, grid_y = np.meshgrid(x_range, y_range)
            grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

            # Create a synthetic dataset with other features held constant
            if features is not None:
                X_const = np.full((grid_points.shape[0], data.shape[1]), features)
                X_const[:, x_idx] = grid_points[:, 0]
                X_const[:, y_idx] = grid_points[:, 1]
            else:
                X_const = np.full((grid_points.shape[0], data.shape[1]), data.mean(axis=0))
                X_const[:, x_idx] = grid_points[:, 0]
                X_const[:, y_idx] = grid_points[:, 1]

            # Predict on the grid and reshape
            Z = model.predict(X_const).reshape(grid_resolution, grid_resolution)
        
            # Plot heatmap
            ax = axes[i, j]
            c = heatmap(Z, cmap=["#e03434", "#21de5d"], ax=ax, cbar=False, vmin=0, vmax=1)

            # Modify axes
            ax.invert_yaxis()

            # Set tick labels to actual feature ranges (instead of grid indices)
            x_ticks = np.linspace(0, len(x_range) - 1, 5)
            x_ticklabels = np.linspace(x_min, x_max, 5).round(2)
            y_ticks = np.linspace(0, len(y_range) - 1, 5)
            y_ticklabels = np.linspace(y_min, y_max, 5).round(2)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels, rotation=0)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticklabels, rotation=0)

            # Set axis titles
            if i == nrows - 1:
                ax.set_xlabel(x_features[j])
            if j == 0:
                ax.set_ylabel(y_features[i])

            # Create legend (on top right subplot)
            if i == 0 and j == ncols - 1:
                legend_labels = [Patch(color="#e03434", label="Malignant"),
                                Patch(color="#21de5d", label="Benign")]
                axes[0, nrows-1].legend(handles=legend_labels, loc="upper right")

            # Highight inputted cancer features from web form
            if features is not None:
                highlight_x, highlight_y = features[x_idx], features[y_idx]

                # Ensure features are in bounds
                if x_min <= highlight_x <= x_max and y_min <= highlight_y <= y_max:
                    # Map to grid indices
                    highlight_x_idx = int((highlight_x - x_min) / (x_max - x_min) * (len(x_range) - 1))
                    highlight_y_idx = int((highlight_y - y_min) / (y_max - y_min) * (len(y_range) - 1))

                    # Add rectangle to highlight the point
                    rect = Rectangle((highlight_x_idx - 0.5, highlight_y_idx - 0.5), 1, 1, 
                                    linewidth=2, edgecolor='#e399f2', facecolor='none')
                    ax.add_patch(rect)

                    # Add label
                    ax.text(highlight_x_idx, highlight_y_idx - 5, 'Your cell', color='#e399f2', 
                            ha='center', va='center', fontsize=10)
                else:
                    # Indicate the point is out of bounds
                    ax.text(len(x_range) // 2, len(y_range) // 2, 'Tumor cell outside plot', 
                            color='#e399f2', ha='center', va='center', fontsize=10)

    # Save Plot
    if features is not None:
        suptitle("Model Predictions with Given Feature Values", fontsize=16)
        tight_layout()

        cancer_path = 'img/this_cancer.png'
        cancer.get_figure().savefig(f'classifier/static/{cancer_path}')
    else:
        suptitle("Model Predictions with Average Feature Values", fontsize=16)
        tight_layout()

        cancer_path = 'img/cancer_knn_new.png'
        cancer.get_figure().savefig(f'classifier/static/{cancer_path}')
    
    clf() # clear figure to create next plot

    ### NOTE: Look into using BytesIO and base64 for sending heatmaps to server
    socketio.emit('retraining-status', {'message': 'Created plots!'})
    print('Created plots!')

    return [cancer_path]