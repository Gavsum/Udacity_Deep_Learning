# Project 1 – Predicting Bike Rental Patterns & Neural Network Basics
This project demonstrates the use of a simple neural network architecture applied to predicting bike rental patterns. It utilizes weather & seasonal information to try and predict the level of demand for bicycles at various points throughout the year.

Data Set: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset

Several key concepts are utilized in this project as follows...

#### Logistic Regression
Logistic regression forms the basis for the implementation of all machine learning and deep learning models. Logistic regression is a method used to solve binary classification problems whereby we attempt to find the coefficients of an equation that best describes this binary classification. 

![Good Func](https://github.com/Gavsum/Udacity_Deep_Learning/blob/master/project1/goodfunc.png "Accurate Classification Function") ![Bad Func](https://github.com/Gavsum/Udacity_Deep_Learning/blob/master/project1/badfunc.png "Inaccurate Classification Function")


We can see that the bad classification function misclassifies two points. If the function got 48/50 points correct, and two wrong, we have an error rate of 0.04.  The ability to measure “how wrong” a function is at describing the data is what allows us to create models that learn through tracking a “wrongness” variable and attempting to reduce it over time. 

#### Neurons
With the above data, optimizing a linear function by reducing the error works well to classify the data. However, what if we need to classify data that doesn’t look as nice? What if we have data that contains a lot of noise, or is perhaps is just difficult to classify with a linear function? This type of data can still be processed with the use of neurons.

![Combined Functions](https://github.com/Gavsum/Udacity_Deep_Learning/blob/master/project1/tensplayground.png "Combining 4 functions to accurately classify noisy data")


Each neuron is a simple computational unit.  It computes the inner product of an input vector and a matching weight vector of trainable parameters. With the function given below (i), a neuron can approximate any general function. 

![Formula](https://github.com/Gavsum/Udacity_Deep_Learning/blob/master/project1/Formula.png "Neuron Formula")

![Formula Visualization](https://github.com/Gavsum/Udacity_Deep_Learning/blob/master/project1/Formula_Visualization.png "Neuron Formula Visualization")

Using both the function definition as well as the visualization, we can see that a neuron will generate a single output value 

#### Weights & Summation
Once the input is fed into the neuron, it gets multiplied by the weight value for this particular type of input. As the network learns, it modifies these weight values such that the final output of the neuron or network classifies the data as accurately as possible.

E.g.: Rainy weather might end up being negatively correlated with bike rental volumes

After each input to the neuron has been multiplied by its relevant weighting, all of the resulting values are summed and then (maybe) passed through an activation function to generate the neuron’s output signal. 

#### Activation Functions & Bias
The activation function used will shape the final output of the neuron. There are many types of activation functions such as Heaviside step, tanh, relu, sigmoid, etc., and they all shape the output in slightly different ways. These functions can be used to help “regularize” our 
network output by scaling possible output values, eliminating negative values, etc. 

![Activation Functions](https://github.com/Gavsum/Udacity_Deep_Learning/blob/master/project1/Activation_Funcs.png "Activation Functions")

TODO: Bias Explanation 
    
List of concepts to explain
* Logistic Regression
* Perceptron
* Gradient Descent
* Multi-layer perceptron
* Backpropagation
* Scaling target variables
* Splitting the data into training, testing, and validation sets
* Choosing some hyper parameters (modify LR .1 to .005 - .02)

Links
* Tensor Flow demo - http://playground.tensorflow.org/




