import pandas
import numpy
from LiarLiar import arePantsonFire

import seaborn
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize

from torch.utils.data import DataLoader
import torch

def create_glove_dict(path_to_text:str): # 0.75 Marks
    """
    Create the dictionary containing word and corresponding vector. 
    :param path_to_text: Path to Glove embeddings.  
    """
    embeddings = {}
    with open(path_to_text,"r",encoding="utf8") as f:
        while True:
            tokens=[]
            line=f.readline()
            if not line:
                break
            tokens=line.split()
            vector=numpy.array(list(map(float,tokens[1:])))
            embeddings[tokens[0]]=vector
    #print(embeddings)
    #del tokens, vector, lines,key
    return embeddings

def get_max_length(dataframe: pandas.DataFrame, column_number: int): # 0.75 Marks
    """
    :param dataframe: Pandas Dataframe
    :param column_number: Column number you want to get max value from
    :return: max_length: int
    """
    max_length = 0
    #print(len(dataframe))
    
    for i in range(0, len(dataframe)):
        length= len(word_tokenize(dataframe.iloc[i][column_number].lower()))
        if(length>max_length):
            max_length=length
    #print(max_length)
    
    return max_length


def visualize_Attenion(attention_matrix: numpy.ndarray):
    """
    Visualizes multihead attention. Expected input shape to [n_heads, query_len, key_len]
    :param attention_matrix:
    :return:
    """
    assert len(attention_matrix.shape) == 3

    for head in range(attention_matrix.shape[0]):
        seaborn.heatmap(attention_matrix[head])
    plt.show()

def infer(model: arePantsonFire, dataloader:DataLoader):
    """
    Use for inferencing on the trained model. Assumes batch_size is 1.
    :param model: trained model.
    :param dataloader: Test Dataloader
    :return:
    """
    labels = {0: "true", 1: "mostly true", 2: "half true", 3: "barely true" , 4: "false", 5: "pants on fire"}
    model.eval()
    correct = 0
    wrong = 0
    for _, data in enumerate(dataloader):
        statement = data['statement']
        justification = data['justification']
        credit_history = data['credit_history']
        label = data['label'][0]
        print(label)
        label=label.cuda()
        statement=statement.cuda()
        justification=justification.cuda()
        credit_history=credit_history.cuda()
        prediction = model(statement, justification, credit_history)
        if torch.argmax(prediction).item() == label.item():
            print("Correct Prediction")
            correct+=1
        else:
            print("wrong prediction")
            wrong+=1

        print(labels[torch.argmax(prediction).item()])

        print('-------------------------------------------------------------------------------------------------------')
    print(correct/_)
    print(wrong/_)
    
