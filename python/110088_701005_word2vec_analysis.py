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

model_name = "PTSD_model"

model = gensim.models.Word2Vec.load('models/'+model_name+'.model')

model.most_similar(positive=["abuse"])

model.most_similar(positive=["military"])

model.most_similar(positive=["medication"])

model.most_similar(positive=["victim"])

model.most_similar(positive=["ptsd"])

model.most_similar(positive=["his"])

model.most_similar(positive=["suicide","self"])

model.most_similar(positive=["family","obligation"],negative = ["love"])

model.most_similar(positive=["brother","girl"],negative = ["boy"])

model.most_similar(positive=["father","woman"],negative=["man"])

model.most_similar(positive=["kitten","dog"],negative=["cat"])

model.most_similar(positive=["veteran","trauma"])

model.most_similar(positive=["law","family"],negative=["love"])

