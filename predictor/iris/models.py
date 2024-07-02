from joblib import load

def load_knn_model(path="predictor/iris/models/iris_knn.pkl"):
    return load(path)

def load_dtree_model(path="predictor/iris/models/iris_dtree.pkl"):
    return load(path)