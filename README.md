### Overview
This project combines web development with machine learning.

This is a Flask app that can classify a variety of different things. As of now, it can only classify different iris species. On the web application, you just have to input feature data and it will return a prediction based on a model. If it is incorrect, you can tell the program, so it can retrain when desired.

### Classification
For iris classification, the scikit-learn's iris dataset was used to train base models. As of now, there is a k-Nearest Neighbors classifier, and a decision tree classifier. More models can be easily added, and I plan on adding random forests next.

In the future, I will continue to add additional models to classify the irises. I will also add different things to classify, such as digits and births. I will primarily be using popular datasets that people encounter when learning about data science and machine learning. 

### Model Correction
Models are never always correct. So, if the model makes a wrong prediction, you can correct it. The program will then store
the correction into a database. This database includes the feature and target data of the correction as well as the model that the correction is for. A corrections list is updated real-time to reflect changes in the database. A "retrain" button will also appear, allowing you to retrain the model with the new corrections.

### Model Visualization
It can be hard to truly visualize how a model makes its classifications. So, graphics are used to visualize the models and gain more insight on how it makes its predictions. At the bottom of each page, there are heatmap visualizations of the base models. After retraining, new graphics are shown side-by-side for direct comparison between the new and old models. Also, when entering features in the form to make a prediction, you can specify if you'd like the classification itself to be visualized. 

For example let's use the KNN model for irises. The iris models have four-dimensions (sepal length, sepal width, petal length, petal width). Since we can't visualize two dimensions, two plots are used to visualize them. One plot is a heatmap for the sepal plane, where sepal length and width are held constant; and the other heatmap is for the petal plane, where the petal length and width are held constant. The model visualizations at the bottom of the page choose the average values of all the data as constants. However, by toggling the visualize button in the classification form, the feature values that you inputted will be used as constants for these planes. Furthermore, thse heatmaps will highlight your iris on the generated heatmaps. This allows you to visualize your iris in context, making it easier to understand how the models are deciding.

### Commands
I also has a number of utility commands for this project

`init-db` - Initializes the database. MUST BE RUN BEFORE RUNNING THE APP FOR THE VERY FIRST TIME
`clean` - Cleans all files created at runtime
`reset-app` - Re-initalizes the database and cleans up all files created at runtime
`reset-iris`- Clears the iris table in the database and cleans all iris-related files created a runtime
