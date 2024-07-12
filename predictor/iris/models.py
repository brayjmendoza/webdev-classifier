from os.path import exists

from joblib import load

#### CONSTANTS ####
TEST_PERCENT = 0.2
BEST_K = 9     # best k for knn model taken from previous cross-validation (hw5)

def load_knn_model():
    path="predictor/iris/models/iris_knn.pkl"

    # Use retrained model if it exists
    if exists("predictor/iris/models/iris_knn_new.pkl"):
        path = 'predictor/iris/models/iris_knn_new.pkl'

    print(f"Using model: {path}")

    return load(path)

def load_dtree_model(path="predictor/iris/models/iris_dtree.pkl"):
    path="predictor/iris/models/iris_dtree.pkl"

    # Use retrained model if it exists
    if exists("predictor/iris/models/iris_knn_new.pkl"):
        path = 'predictor/iris/models/iris_dtree_new.pkl'

    print(f"Using model: {path}")

    return load(path)