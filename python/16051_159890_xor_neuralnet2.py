import numpy as np #import important libraries. 
from matplotlib import pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')

plt.plot([0,1],[0,1],'bo')
plt.plot([0,1],[1,0],'ro')
plt.ylabel('Value 2')
plt.xlabel('Value 1')
plt.axis([-0.5,1.5,-0.5,1.5]);

from visualise_neural_network import NeuralNetwork

network = NeuralNetwork() #create neural network object
network.add_layer(2,['Input 1','Input 2']) #input layer with names
network.add_layer(2,['Hidden 1','Hidden 2']) #hidden layer with names
network.add_layer(1,['Output']) #output layer with name
network.draw()

np.random.seed(seed=10) #seed random number generator for reproducibility

Weights_2 = np.random.rand(1,3)-0.5*2 #connections between hidden and output
Weights_1 = np.random.rand(2,3)-0.5*2 #connections between input and hidden

Weight_Dict = {'Weights_1':Weights_1,'Weights_2':Weights_2} #place weights in a dictionary

Train_Set = [[1.0,1.0],[0.0,0.0],[0.0,1.0],[1.0,0.0]] #train set

network = NeuralNetwork()
network.add_layer(2,['Input 1','Input 2'],
                  [[round(x,2) for x in Weight_Dict['Weights_1'][0][:2]],
                   [round(x,2) for x in Weight_Dict['Weights_1'][1][:2]]])
#add input layer with names and weights leaving the input neurons
network.add_layer(2,[round(Weight_Dict['Weights_1'][0][2],2),round(Weight_Dict['Weights_1'][1][2],2)],
                  [round(x,2) for x in Weight_Dict['Weights_2'][0][:2]])
#add hidden layer with names (each units' bias) and weights leaving the hidden units
network.add_layer(1,[round(Weight_Dict['Weights_2'][0][2],2)])
#add output layer with name (the output unit's bias)
network.draw()

Input = np.array([0,1])
net_Hidden = np.dot(np.append(Input,1.0),Weights_1.T) #append the bias input
print net_Hidden

def logistic(x): #each neuron has a logistic activation function
    return 1.0/(1+np.exp(-x))

Hidden_Units = logistic(net_Hidden)
print Hidden_Units

net_Output = np.dot(np.append(Hidden_Units,1.0),Weights_2.T)
print 'net_Output'
print net_Output
Output = logistic(net_Output)
print 'Output'
print Output

def layer_InputOutput(Inputs,Weights): #find a layers input and activity
    Inputs_with_bias = np.append(Inputs,1.0) #input 1 for each unit's bias
    return logistic(np.dot(Inputs_with_bias,Weights.T))

def neural_net(Input,Weights_1,Weights_2,Training=False): #this function creates and runs the neural net    
        
    target = 1 #set target value
    if np.array(Input[0])==np.array([Input[1]]): target = 0 #change target value if needed
    
    #forward pass
    Hidden_Units = layer_InputOutput(Input,Weights_1) #find hidden unit activity
    Output = layer_InputOutput(Hidden_Units,Weights_2) #find Output layer actiity
        
    return {'output':Output,'target':target,'input':Input} #record trial output

Train_Set = [[1.0,1.0],[0.0,1.0],[1.0,0.0],[0.0,0.0]] #the four input patterns
tempdict = {'output':[],'target':[],'input':[]} #data dictionary
temp = [neural_net(Input,Weights_1,Weights_2) for Input in Train_Set] #get the data
[tempdict[key].append([temp[x][key] for x in range(len(temp))]) for key in tempdict] #combine all the output dictionaries

plotter = np.ones((2,2))*np.reshape(np.array(tempdict['output']),(2,2))
plt.pcolor(plotter,vmin=0,vmax=1,cmap=plt.cm.bwr)
plt.colorbar(ticks=[0,0.25,0.5,0.75,1]);
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.xticks([0.5,1.5], ['0','1'])
plt.yticks([0.5,1.5], ['0','1']);

alpha = 0.5 #learning rate
target = 1 #target outpu

error = target - Output #amount of error
delta_out = np.atleast_2d(error*(Output*(1-Output))) #first two terms of error by weight derivative

Hidden_Units = np.append(Hidden_Units,1.0) #add an input of 1 for the bias
print Weights_2 + alpha*np.outer(delta_out,Hidden_Units) #apply weight change

delta_hidden = delta_out.dot(Weights_2)*(Hidden_Units*(1-Hidden_Units)) #find delta portion of weight update
                       
delta_hidden = np.delete(delta_hidden,2) #remove the bias input
print Weights_1 + alpha*np.outer(delta_hidden,np.append(Input,1.0)) #append bias input and multiply input by delta portion 

def neural_net(Input,Weights_1,Weights_2,Training=False): #this function creates and runs the neural net    
        
    target = 1 #set target value
    if np.array(Input[0])==np.array([Input[1]]): target = 0 #change target value if needed
    
    #forward pass
    Hidden_Units = layer_InputOutput(Input,Weights_1) #find hidden unit activity
    Output = layer_InputOutput(Hidden_Units,Weights_2) #find Output layer actiity
        
    if Training == True:
        alpha = 0.5 #learning rate
        
        Weights_2 = np.atleast_2d(Weights_2) #make sure this weight vector is 2d.
        
        error = target - Output #error
        delta_out = np.atleast_2d(error*(Output*(1-Output))) #delta between output and hidden
        
        Hidden_Units = np.append(Hidden_Units,1.0) #append an input for the bias
        delta_hidden = delta_out.dot(np.atleast_2d(Weights_2))*(Hidden_Units*(1-Hidden_Units)) #delta between hidden and input
               
        Weights_2 += alpha*np.outer(delta_out,Hidden_Units) #update weights
        
        delta_hidden = np.delete(delta_hidden,2) #remove bias activity
        Weights_1 += alpha*np.outer(delta_hidden,np.append(Input,1.0))  #update weights
            
    if Training == False: 
        return {'output':Output,'target':target,'input':Input} #record trial output
    elif Training == True:
        return {'Weights_1':Weights_1,'Weights_2':Weights_2,'target':target,'output':Output,'error':error}

from random import choice
np.random.seed(seed=10) #seed random number generator for reproducibility

Weights_2 = np.random.rand(1,3)-0.5*2 #connections between hidden and output
Weights_1 = np.random.rand(2,3)-0.5*2 #connections between input and hidden
                      
Weight_Dict = {'Weights_1':Weights_1,'Weights_2':Weights_2}

Train_Set = [[1.0,1.0],[0.0,0.0],[0.0,1.0],[1.0,0.0]] #train set

Error = []
while True: #train the neural net
    Train_Dict = neural_net(choice(Train_Set),Weight_Dict['Weights_1'],Weight_Dict['Weights_2'],Training=True)
    
    Error.append(abs(Train_Dict['error']))
    if len(Error) > 6 and np.mean(Error[-10:]) < 0.025: break #tell the code to stop iterating when recent mean error is small

Error_vec = np.array(Error)[:,0]
plt.plot(Error_vec)
plt.ylabel('Error')
plt.xlabel('Iteration #');

Weights_1 = Weight_Dict['Weights_1']
Weights_2 = Weight_Dict['Weights_2']

Train_Set = [[1.0,1.0],[0.0,1.0],[1.0,0.0],[0.0,0.0]] #train set

tempdict = {'output':[],'target':[],'input':[]} #data dictionary
temp = [neural_net(Input,Weights_1,Weights_2) for Input in Train_Set] #get the data
[tempdict[key].append([temp[x][key] for x in range(len(temp))]) for key in tempdict] #combine all the output dictionaries

plotter = np.ones((2,2))*np.reshape(np.array(tempdict['output']),(2,2))
plt.pcolor(plotter,vmin=0,vmax=1,cmap=plt.cm.bwr)
plt.colorbar(ticks=[0,0.25,0.5,0.75,1]);
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.xticks([0.5,1.5], ['0','1'])
plt.yticks([0.5,1.5], ['0','1']);

Weight_Dict = {'Weights_1':Weights_1,'Weights_2':Weights_2}

network = NeuralNetwork()
network.add_layer(2,['Input 1','Input 2'],
                  [[round(x,2) for x in Weight_Dict['Weights_1'][0][:2]],
                   [round(x,2) for x in Weight_Dict['Weights_1'][1][:2]]])
network.add_layer(2,[round(Weight_Dict['Weights_1'][0][2],2),round(Weight_Dict['Weights_1'][1][2],2)],
                  [round(x,2) for x in Weight_Dict['Weights_2'][:2][0]])
network.add_layer(1,[round(Weight_Dict['Weights_2'][0][2],2)])
network.draw()

Weights_2 = np.array([-4.5,5.3,-0.8]) #connections between hidden and output
Weights_1 = np.array([[-2.0,9.2,2.0],
                     [4.3,8.8,-0.1]])#connections between input and hidden

Weight_Dict = {'Weights_1':Weights_1,'Weights_2':Weights_2}

network = NeuralNetwork()
network.add_layer(2,['Input 1','Input 2'],
                  [[round(x,2) for x in Weight_Dict['Weights_1'][0][:2]],
                   [round(x,2) for x in Weight_Dict['Weights_1'][1][:2]]])
network.add_layer(2,[round(Weight_Dict['Weights_1'][0][2],2),round(Weight_Dict['Weights_1'][1][2],2)],
                  [round(x,2) for x in Weight_Dict['Weights_2'][:2]])
network.add_layer(1,[round(Weight_Dict['Weights_2'][2],2)])
network.draw()

Train_Set = [[1.0,1.0],[0.0,0.0],[0.0,1.0],[1.0,0.0]] #train set

Error = []
while True:
    Train_Dict = neural_net(choice(Train_Set),Weight_Dict['Weights_1'],Weight_Dict['Weights_2'],Training=True)
    
    Error.append(abs(Train_Dict['error']))
    if len(Error) > 6 and np.mean(Error[-10:]) < 0.025: break
        
Error_vec = np.array(Error)[:]
plt.plot(Error_vec)
plt.ylabel('Error')
plt.xlabel('Iteration #');

Weights_1 = Weight_Dict['Weights_1']
Weights_2 = Weight_Dict['Weights_2']

Train_Set = [[1.0,1.0],[0.0,1.0],[1.0,0.0],[0.0,0.0]] #train set

tempdict = {'output':[],'target':[],'input':[]} #data dictionary
temp = [neural_net(Input,Weights_1,Weights_2) for Input in Train_Set] #get the data
[tempdict[key].append([temp[x][key] for x in range(len(temp))]) for key in tempdict] #combine all the output dictionaries

plotter = np.ones((2,2))*np.reshape(np.array(tempdict['output']),(2,2))
plt.pcolor(plotter,vmin=0,vmax=1,cmap=plt.cm.bwr)
plt.colorbar(ticks=[0,0.25,0.5,0.75,1]);
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.xticks([0.5,1.5], ['0','1'])
plt.yticks([0.5,1.5], ['0','1']);

Weights_1 = Weight_Dict['Weights_1']
Weights_2 = Weight_Dict['Weights_2']

Weight_Dict = {'Weights_1':Weights_1,'Weights_2':Weights_2}

network = NeuralNetwork()
network.add_layer(2,['Input 1','Input 2'],
                  [[round(x,2) for x in Weight_Dict['Weights_1'][0][:2]],
                   [round(x,2) for x in Weight_Dict['Weights_1'][1][:2]]])
network.add_layer(2,[round(Weight_Dict['Weights_1'][0][2],2),round(Weight_Dict['Weights_1'][1][2],2)],
                  [round(x,2) for x in Weight_Dict['Weights_2'][:2]])
network.add_layer(1,[round(Weight_Dict['Weights_2'][2],2)])
network.draw()

