from os.path import exists

from joblib import load

#### CONSTANTS ####
TEST_PERCENT = 0.2
BEST_K = 9     # best k for knn model taken from previous cross-validation (hw5)

def load_knn_model():
    path="predictor/models/iris/knn.pkl"

    # Use retrained model if it exists
    if exists("predictor/models/iris/knn_new.pkl"):
        path = 'predictor/models/iris/knn_new.pkl'

    print(f"Using model: {path}")

    return load(path)

def load_dtree_model():
    path="predictor/models/iris/dtree.pkl"

    # Use retrained model if it exists
    if exists("predictor/models/iris/dtree_new.pkl"):
        path = 'predictor/models/iris/dtree_new.pkl'

    print(f"Using model: {path}")

    return load(path)