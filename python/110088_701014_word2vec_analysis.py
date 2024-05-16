#Import graphing utilities
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# Import useful mathematical libraries
import numpy as np
import pandas as pd

# Import useful Machine learning libraries
import gensim

# Import utility files
from utils import save_object,load_object

model_name = "model6"

model = gensim.models.Word2Vec.load('models/'+model_name+'.model')

model.most_similar(positive=["heartbreak"])

model.most_similar(positive=["pills"])

model.most_similar(positive=["knife"])

model.most_similar(positive=["kitten"])

model.most_similar(positive=["puppy"])

model.most_similar(positive=["abusive","words"],negative =["physical"])

model.most_similar(positive=["suicide","self"])

model.most_similar(positive=["family","obligation"],negative = ["love"])

model.most_similar(positive=["father","woman"],negative=["man"])

model.most_similar(positive=["kitten","dog"],negative=["cat"])

model.most_similar(positive=["i"])

