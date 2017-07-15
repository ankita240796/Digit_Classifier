# MNIST Handwrritten Digit Classifier
This is the python implementation of digit classifer using numpy library.
# Intro to Machine Learning
Before starting with neural networks here is a small intro to machine learning.Machine learning, as the name suggests is a way to make the machine learn and make right decisions on its own. A machine is said to learn from its experience E with respect to some task T and some performance P,if its performance P on T improves with experience E.There are two braod categories of machine learning algorithms:-

>Supervised Learning Algorthims are the ones which learn when given the right answer for each example.When a real value output is produced it comes under regression problem and a discrete value output defines classification problem.

>Unsupervised Learning Algorthims are used to draw inferences from datasets consisting of input data without any labels. The most common unsupervised learning method is cluster analysis which divides data into different clusters.

# Neural Networks
Neural Network is a system which is modelled on human brain and nervous system.Similar to neurons in human nervous system there are neurons in this system which recieve a set of input data and produce some output which can be used as input for further neurons or as the final output for the input data.

##### Neurons
Neurons are the computational unit of a neural network. A neuron takes several inputs and produces a single output.
![N|Solid](http://i.imgur.com/cPfELbJ.png)

This is a simple neuron where x1,x2,....xn are the inputs with respective weights w1,w2,...wn.These weights and inputs are used to produce the output of neuron.Further there is a bias unit with weight w0 and input as 1.w0 can also be denoted as b to represent the bias.The neuron which I am describing here is a sigmoid neuron which uses the sigmoid function to generate the output.

##### Sigmoid function

![N|Solid](http://i.imgur.com/kFQz36H.png)

![N|Solid](http://i.imgur.com/Q4l5Fz5.png)

We use a sigmoid function to produce the output for a neuron rather than a step function so that a small change in a weight (or bias) causes only a small change in output, then we could use this fact to modify the weights and biases to get our neural network to behave more in the manner we want.For example, we design a network to classify digits and the network was mistakenly classifying an image as an "8" when it should be a "9". We could figure out how to make a small change in the weights and biases so the network gets a little closer to classifying the image as a "9".

##### Output of Sigmoid Neuron

![N|Solid](http://i.imgur.com/abA1fB7.png)

![N|Solid](http://i.imgur.com/I7BqHmB.png)

##### Significance of Weights and Bias
The weights here determine the respective weightage of the input to produce the desired output.Bias can be thought of as a measure of how easy it is to get the neuron to output a 1.For a neuron with a really big bias, it's extremely easy for the perceptron to output a 1 but if the bias is very negative, then it's difficult for the neuron to output a 1.For example let's consider a simple neuron which produces ouput as follows:- 

![N|Solid](http://i.imgur.com/yugDgJ7.png)

Now if we want our neuron to decide that whether to go for a fest or not by using two inputs x1 and x2.x1=1 denotes good weather and x1=0 bad weather.x2=1 denotes that the fest is near public transit and x2=0 otherwise.If weather is bad as well as fest is awat from public transit, you don't want to attend the fest i.e output y=0.If the fest is away from public transit but the weather is good you decide to attend the fest i.e y=1.On the other hand in case of bad weather and availability of public transit you decide not to attend the fest i.e y=0. It is clear from this example that weather has more weightage for a positive decision. So we can assign the weights as w1=4,w2=2 and bias b=-3. 

Now let's see the various cases:-
>x1=1, x2=1 => Good weather , public transit available
w.x+b=4+2-3=3>0 => y=1

>x1=1 x2=0 => Good weather , public transit unavailbale
w.x+b=4+0-3=1>0 => y=1

>x1=0, x2=1 => Bad weather , public transit available
w.x+b=0+2-3=-1<0 => y=0

>x1=0, x2=0 => Bad weather , public transit unavailable
w.x+b=0+0-3=-3<0 => y=0

On the other hand if we choose values of weights and bias as w1=2,w2=4 and b=-3 we would have given more importance to public transit factor.Similarly choosing a bias b=-1 would have resulted in positive output even in case of bad weather and availability of public transit.This simple example clearly shows the relevance of weights and bias.
##### Layers in Neural Network
![N|Solid](http://i.imgur.com/MivPGt7.jpg)

This is a simple neural network.The first layer is the input layer and the last layer is output layer.Any layers in between are hidden layers.The design of the input and output layers in a network is often straightforward. For example, suppose we're trying to determine whether a handwritten image depicts a "9" or not. A natural way to design the network is to encode the intensities of the image pixels into the input neurons. If the image is a 64 by 64 greyscale image, then we'd have 4,096=64×64 input neurons, with the intensities scaled appropriately between 0 and 1. The output layer will contain just a single neuron, with output values of less than 0.5 indicating "input image is not a 9", and values greater than 0.5 indicating "input image is a 9 ".While the design of the input and output layers of a neural network is often straightforward, there can be quite an art to the design of the hidden layers. In particular, it's not possible to sum up the design process for the hidden layers with a few simple rules of thumb. Instead, neural networks researchers have developed many design heuristics for the hidden layers, which help people get the behaviour they want out of their nets.

##### Neural Network Architecture

S(l) denotes the number of neurons in lth layer.Weights connecting Layer l of size S(l) and layer l+1 say of S(l+1) form a matrix of dimension S(l+1)*S(l) denoted by weight(l). Wl(ij) denotes weight connecting ith neuron in (l+1) th layer with jth neuron in lth layer. Bias(l) is a vector containing bias values corresponding to each neuron in (l+1)th layer.

![N|Solid](http://i.imgur.com/YFfVQ0g.png)

##### Gradient Descent to Train Neural Network
Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function (cost).
We define the cost function for our neural networks as follows:-

![N|Solid](http://i.imgur.com/h79owj1.png)

Here, w denotes the collection of all weights in the network, b all the biases, nn is the total number of training inputs, a is the vector of outputs from the network when x is input, and the sum is over all training inputs, x.The notation ||v|| just denotes the usual length function for a vector v. We'll call C the quadratic cost function.

![N|Solid](http://i.imgur.com/xpHoyvr.png)

Think of a large bowl like what you would eat cereal out of or store fruit in. This bowl is a plot of the cost function (f).A random position on the surface of the bowl is the cost of the current values of the coefficients (cost).The bottom of the bowl is the cost of the best set of coefficients, the minimum of the function.The goal is to continue to try different values for the coefficients, evaluate their cost and select new coefficients that have a slightly better (lower) cost.Repeating this process enough times will lead to the bottom of the bowl and you will know the values of the coefficients that result in the minimum cost.

Gradient Descent updates the variable w and b in each iteration so that the cost keeps on decreasing until it reaches a minimum.

![N|Solid](http://i.imgur.com/WWkLUj0.png)

∂C/∂wk and ∂C/∂bl are derivatives of the cost function with respect to weight w(k) and bias b(l) and eta is the learning rate. By repeatedly applying this update rule we can "roll down the bowl", and hopefully find a minimum of the cost function. 
To compute the gradient ∇C we need to compute the gradients ∇Cx separately for each training input, x, and then average them,  ∇C=1/n∑∇Cx.Unfortunately, when the number of training inputs is very large this can take a long time, and learning thus occurs slowly. An idea called mini batch gradient descent can be used to speed up learning. The idea is to estimate the gradient ∇C by computing ∇Cx for a small sample of randomly chosen training inputs. By averaging over this small sample it turns out that we can quickly get a good estimate of the true gradient ∇C, and this helps speed up gradient descent, and thus learning.

##### Backpropagation
Backpropagation is a method to calculate the gradient of the cost function with respect to the weights and biases in an artificial neural network. It is commonly used as a part of algorithms that optimize the performance of the network by adjusting the weights, for example in the gradient descent algorithm. 

The algorithm is as follows:-
- ##### Note :-
    The sigmoid-prime derivative terms can also be written out as:
    "g" here denotes sigmoid function
    ![N|Solid](http://i.imgur.com/H0UMVox.png)
    
    where a(l)=sigmoid(z(l))
    ".*" represents element wise multiplication
- Given training set {(x(1),y(1))⋯((x(n),y(n))}
- Set Δ(l)i,j := 0 for all (l,i,j), (hence you end up having a matrix full of zeros)

- For training example t =1 to n:
>Set a(1):=x(t)

>Perform forward propagation to compute a(l) for l=2,3,…,L

![N|Solid](http://i.imgur.com/L0LkJzg.jpg)
{theta denotes weight matrix here}

>Using y(t), compute δ(L)=(a(L)−y(t)).*g'(z(L))
Where L is our total number of layers and a(L) is the vector of outputs of the activation units for the last layer.

>Compute δ(L−1),δ(L−2),…,δ(2) using  
![N|Solid](http://i.imgur.com/2XaeZoo.png)
The delta values of layer l are calculated by multiplying the delta values in the next layer with the theta matrix of layer l. We then element-wise multiply that with a function called g', or g-prime, which is the derivative of the activation function g evaluated with the input values given by z(l).


 >![N|Solid](http://i.imgur.com/LAfZ2pM.png) or with vectorization, ![N|Solid](http://i.imgur.com/56poaub.png)
Hence we update our new Δ matrix.

>![N|Solid](http://i.imgur.com/oJocVHd.png)

>The capital-delta matrix D is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative. Thus we get
![N|Solid](http://i.imgur.com/URO3QXl.png)

I have not described the mathematical details of the algortihm here. To get into the mathematical details , you can refer this link :-http://neuralnetworksanddeeplearning.com/chap2.html

# My implemenentation of Digit Classifier
The code is written in python using numpy library.You can read about numpy library here:-http://cs231n.github.io/python-numpy-tutorial/#numpy.
There are two files:-
- network.py- to train the neural network to classify digits with great accuracy
- load.py- to load the training,test and validation data in proper format
>Note :-Training data is the the data set to train our network validadtion data can be used for setting right values for parameters like learning rate and test data is used to finally check the accuracy of our system. 

>In this implementation :- Training data for the network will consist of many 28 by 28 pixel images of scanned handwritten digits, and so the input layer contains 784=28×28 neurons.The second layer of the network is a hidden layer which consists of 30 neurons.The output layer of the network contains 10 neurons. If the first neuron fires, i.e., has an output ≈1, then that will indicate that the network thinks the digit is a 0. If the second neuron fires then that will indicate that the network thinks the digit is a 1. And so on. 

I got an accuracy of about 95% using this implementation. This high accuracy may exceed the human accuracy for certain difficult digit recognition.Here is an screenshot showing the accuracy for 30 epoch(In epoch the implementation trains the network using mini batch gradient descent.)After each epoch accuracy on test data will be displayed on running the code.

The code can be executed by launching the python shell in the source code directory and giving the following commands as in the screenshot:-

![N|Solid](http://i.imgur.com/BTjxCXu.png)



     















