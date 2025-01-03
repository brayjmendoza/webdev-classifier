from os.path import exists

from joblib import load

#### CONSTANTS ####
TEST_PERCENT = 0.2
BEST_K = 15      # best k for knn model taken from previous cross-validation
BEST_DEPTH = 5  # best depth for dtree model taken from previous cross-validation

def load_knn_model():
    """Load the cancer KNN model"""
    path="classifier/models/cancer/knn.pkl"

    # Use retrained model if it exists
    if exists("classifier/models/cancer/knn_new.pkl"):
        path = 'classifier/models/cancer/knn_new.pkl'

    print(f"Using model: {path}")

    return load(path)

def load_dtree_model():
    """Load the cancer dtree model"""
    path="classifier/models/cancer/dtree.pkl"

    # Use retrained model if it exists
    if exists("classifier/models/cancer/dtree_new.pkl"):
        path = 'classifier/models/cancer/dtree_new.pkl'

    print(f"Using model: {path}")

    return load(path)

def load_mlp_model():
    """Load the cancer mlp classifier"""
    path="classifier/models/cancer/mlp.pkl"

    # Use retrained model if it exists
    if exists("classifier/models/cancer/mlp_new.pkl"):
        path = 'classifier/models/cancer/mlp_new.pkl'

    print(f"Using model: {path}")

    return load(path)