### Overview
This project combines web development with machine learning and data visualization.

This is a Flask app that can classify a variety of different things. As of now, it can only classify different iris species. On the web application, you just have to input feature data and it will return a prediction based on a model. If it is incorrect, you can tell the program, so it can retrain when desired.

This project was done for an NSF-funded research project, *Scripting For All*, by Professor Dodds of Harvey Mudd College. The overall goal of the research is to expand the understanding of computer science to disciplines that traditionally don't use it. Computation is becoming an increasingly useful and necessary skill, and we'd love to make it more accessible to everyone. This webdev-classifier project is an example of a final (albeit large) project that students should have the ability to do after taking HMC's CS35 *Computing for Insight* course.

### Classification
For iris classification, the scikit-learn's iris dataset was used to train base models. As of now, there are three different classifiers: k-Nearest Neighbors, decision tree, multi-layer perceptron. More models can be easily added.

In the future, I will continue to add additional models to classify the irises. I will also add different things to classify, such as digits and births. I will primarily be using popular datasets that people encounter when learning about data science and machine learning.

The training for all of the initial models was done in initial_models.ipynb. That being said, some of the values for the iris models were done separately in a homework assignment from Harvey Mudd College's CS35 taught by Professor Dodds.

### Model Correction
Models are never always correct. So, if the model makes a wrong prediction, you can correct it. The program will then store
the correction into a database. This database includes the feature and target data of the correction as well as the model that the correction is for. A corrections list is updated real-time to reflect changes in the database. A "retrain" button will also appear, allowing you to retrain the model with the new corrections.

### Model Visualization
It can be hard to truly visualize how a model makes its classifications. So, graphics are used to visualize the models and gain more insight on how it makes its predictions. At the bottom of each page, there are visualizations of the base models. After retraining, new graphics are shown side-by-side for direct comparison between the new and old models. Also, when entering features in the form to make a prediction, you can specify if you'd like the classification itself to be visualized. 

For example, let's use the KNN model for irises. The iris models have four-dimensions (sepal length, sepal width, petal length, petal width). Since we can't visualize two dimensions, two plots are used to visualize them. One plot is a heatmap for the sepal plane, where sepal length and width are held constant; and the other heatmap is for the petal plane, where the petal length and width are held constant. The model visualizations at the bottom of the page choose the average values of all the data as constants. However, by toggling the visualize button in the classification form, the feature values that you inputted will be used as constants for these planes. Furthermore, these heatmaps will highlight your iris on the generated heatmaps. This allows you to visualize your iris in context, making it easier to understand how the models are deciding.

### Commands
I also have a number of utility commands for this project

`init-db` - Initializes the database. **MUST BE RUN BEFORE RUNNING THE APP FOR THE VERY FIRST TIME**
`clean` - Cleans all files created at runtime
`reset-app` - Re-initalizes the database and cleans up all files created at runtime
`reset-iris`- Clears the iris table in the database and cleans all iris-related files created a runtime

### Running the Flask App
First, install the necessary dependencies found in requirements.txt (either in a global or virtual environment). Then simply run the following command to run the flask app:

```shell
flask --app classifier run
```

You can optionally choose which port to use with the `--port` flag.

We can also run the custom commands in a similar way:
```shell
flask --app classifier <command>
```

This will run `<command>` as explained above. Be sure to run the `init-db` command before running the app for the first time.