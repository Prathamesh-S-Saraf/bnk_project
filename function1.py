import pickle
import config1
import numpy as np

model = pickle.load(open(config1.MODEL_PATH,'rb'))



def prediction(data):

    result = model.predict(data)
    

    if result[0] == 1:
        return "End as bank offer subscription "

    else:
        return  "Not bank offer subscription"
    
    