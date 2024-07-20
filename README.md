### Overview
This project combines web development with machine learning.

It uses Flask to create a web application that can classify a variety of different things. As of now, it can only classify different iris species.
On the web application, you just have to input feature data and it will return a prediction based on a model. If it is incorrect, you can tell
the program, so it can retrain when desired.

### Classification
For iris classificaiton, the iris dataset is used to train multiple models. These include the nearest neighbors algorithm and decision trees.
In the future, I will continue to add additional models to classify the irises. 

A key feature is if the model makes a wrong prediction, you can correct it. The program will then store
the correction into a database. A "retrain" button will also appear, allowing you to retrain the model with the new corrections.

### Model Visualization
Another aspect of this project is visualizing a selected model to gain more insight on how the program is making its classifications. 
After retraining, there is a side-by-side comparison between the old and new models.

Also, after making a prediction, you can specify if you'd like the classification plane to be visualized. 
