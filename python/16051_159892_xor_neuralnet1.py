import numpy as np #import important libraries. 
from matplotlib import pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')

from visualise_neural_network import NeuralNetwork

network = NeuralNetwork() #create neural network object
network.add_layer(2,['Input A','Input B'],['Weight A','Weight B']) #create the input layer which has two neurons.
#Each input neuron has a single line extending to the next layer up
network.add_layer(1,['Output']) #create output layer - a single output neuron
network.draw() #draw the network

Inputs = np.array([0, 1])
Weights = np.array([0.25, 0.5])

net_Output = np.dot(Inputs,Weights)
print net_Output

def logistic(x): #each neuron has a logistic activation function
    return 1.0/(1+np.exp(-x))

x = np.arange(-5,5,0.1) #create vector of numbers between -5 and 5
plt.plot(x,logistic(x))
plt.ylabel('Activation')
plt.xlabel('Input');

Output_neuron = logistic(net_Output)
print Output_neuron
plt.plot(x,logistic(x));
plt.ylabel('Activation')
plt.xlabel('Input')
plt.plot(net_Output,Output_neuron,'ro');

Inputs = [[1,0],[0,1]]
Answers = [0,1,]

Guesses = [logistic(np.dot(x,Weights)) for x in Inputs] #loop through inputs and find logistic(sum(input*weights))
plt.plot(Guesses,'bo')
plt.plot(Answers,'ro')
plt.axis([-0.5,1.5,-0.5,1.5])
plt.ylabel('Activation')
plt.xlabel('Input #')
plt.legend(['Guesses','Answers']);
print Guesses

alpha = 0.5

def delta_Output(target,Output):
    return -(target-Output)*Output*(1-Output) #find the amount of error and derivative of activation function

def update_weights(alpha,delta,unit_input):
    return alpha*np.outer(delta,unit_input) #multiply delta output by all the inputs and then multiply these by the learning rate 

def network_guess(Input,Weights):
    return logistic(np.dot(Input,Weights.T)) #input by weights then through a logistic

def back_prop(Input,Output,target,Weights):
    delta = delta_Output(target,Output) #find delta portion
    delta_weight = update_weights(alpha,delta,Input) #find amount to update weights
    Weights = np.atleast_2d(Weights) #convert weights to array
    Weights += -delta_weight #update weights
    return Weights

from random import choice, seed
seed(1) #seed random number generator so that these results can be replicated

Weights = np.array([0.25, 0.5])

Error = []
while True:
    
    Trial_Type = choice([0,1]) #generate random number to choose between the two inputs
    
    Input = np.atleast_2d(Inputs[Trial_Type]) #choose input and convert to array
    Answer = Answers[Trial_Type] #get the correct answer
    
    Output = network_guess(Input,Weights) #compute the networks guess
    Weights = back_prop(Input,Output,Answer,Weights) #change the weights based on the error
    
    Error.append(abs(Output-Answer)) #record error
    
    if len(Error) > 6 and np.mean(Error[-5:]) < 0.05: break #tell the code to stop iterating when mean error is < 0.05 in the last 5 guesses

Error_vec = np.array(Error)[:,0]
plt.plot(Error_vec)
plt.ylabel('Error')
plt.xlabel('Iteration #');

Inputs = [[1,0],[0,1]]
Answers = [0,1,]

Guesses = [logistic(np.dot(x,Weights.T)) for x in Inputs] #loop through inputs and find logistic(sum(input*weights))
plt.plot(Guesses,'bo')
plt.plot(Answers,'ro')
plt.axis([-0.5,1.5,-0.5,1.5])
plt.ylabel('Activation')
plt.xlabel('Input #')
plt.legend(['Guesses','Answers']);
print Guesses

