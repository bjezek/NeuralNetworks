# NeuralNetworks are Awesome!

# Venture Capital Fund yes/no binary classification model

# Imported Libaries
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder


# TARGET[IS_SUCCESSFUL], 1 +

# Review, Edit & Analyse data
Split the data into training & testing set for X, y remove two columns the where not needed for the module and would have slowed it down when running the epochs
use OneHotEncoder and StandardScaler to change the format to fit the binary model and all value types to floats 
so 1 and above is consideration while 0 is failure of firm 

#  Evaluate a Binary Classification Model Using a Neural Network
 looks at an input and predicts which of two possible classes it belongs to. lending is a 1 and above class 0 is a failure not giving no money 
 
# Fit the model using the binary_crossentropy loss function, the adam optimizer, and the accuracy evaluation metric
categorical crossentropy when your classes are mutually exclusive aka 1 or 0 (e.g. when each sample belongs exactly to one class) and adam optimizer
has RMSProp which  is designed to accelerate the optimization process, e.g. decrease the number of function evaluations required to reach the optima, or to improve the capability of the optimization algorithm, e.g. result in a better final result https://machinelearningmastery.com/gradient-descent-with-rmsprop-from-scratch/

#  three new deep neural network models (resulting in the original plus 3 optimization attempts). Improve on my first model’s predictive accuracy.

# Original Model Results
268/268 - 0s - loss: 0.5545 - accuracy: 0.7299 - 415ms/epoch - 2ms/step
Loss; 0.5544908046722412, Accuracy: 0.729912519454956

#  Alternative Model 1 Results
268/268 - 0s - loss: 0.5542 - accuracy: 0.7314 - 346ms/epoch - 1ms/step
Loss; 0.5542263984680176, Accuracy: 0.7314285635948181

# Alternative Model 2 Results
268/268 - 1s - loss: 0.5505 - accuracy: 0.7315 - 591ms/epoch - 2ms/step
Loss; 0.5504850149154663, Accuracy: 0.7315452098846436



# Report
Alternative Model 2 Results had the best accuracy/loss score due to more hidden
layers and nodes within the hidden layers also  less passes through the algorithim (epochs)
I also used # Sigmoid for the last model As you can see, the sigmoid is a function that only occupies the range from 0 to 1 and it asymptotes both values. This makes it very handy for binary classification with 0 and 1 as potential output values
https://programmathically.com/the-sigmoid-function-and-binary-logistic-regression/#:~:text=As%20you%20can%20see%2C%20the,1%20as%20potential%20output%20values.

 I believe if I was able to add more input features to the algorithim and kept out_putlayers neuron low say to 1 with 1 hidden layer 
 and 25 to 50 epochs I could have got closer to 1 but for this practice it was not possible to do so. 
