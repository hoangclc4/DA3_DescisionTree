import numpy as np
import pandas as pd
#get the smallest number to make denominator not have value=0
from numpy import log2
import pickle

import json
def calculate_entropy(df):
    #get the last column
    last_column = df.keys()[-1]  
    entropy = 0
    #get unique value in the entire values in column in dataframe
    values = df[last_column].unique()
    for value in values:
        fraction = df[last_column].value_counts()[value]/len(df[last_column])#  fraction= number of sub values / total number
        entropy += -fraction*np.log2(fraction)
    return entropy
  
  
def calculate_entropy_attribute(df,attribute):
  last_column = df.keys()[-1]  
  target_variables = df[last_column].unique()  
  variables = df[attribute].unique()   
  remainder = 0
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          # all rows match var name and match target var name
          num = len(df[attribute][df[attribute]==variable][df[last_column] ==target_variable])
          denominator = len(df[attribute][df[attribute]==variable])
          sub_fraction = num/(denominator+eps)
          entropy += -sub_fraction*log2(sub_fraction+eps)
      fraction = denominator/len(df)
      remainder += -fraction*entropy 
  return abs(remainder)


def find_max_gain(df):
    information_gain = []
    ## for each column except last column
    for key in df.keys()[:-1]:
        information_gain.append(calculate_entropy(df)-calculate_entropy_attribute(df,key))
    ## return column with is index max of IG
    max_IG_column_name=df.keys()[:-1][np.argmax(information_gain)]
    print("max Information Gain : ",end=max_IG_column_name)
    print("\n")

    return max_IG_column_name
  
  
def get_subtable(df, node,value):
    column_value=df[node] ## 'patron' column
    column_boolean=column_value == value ## true false
    print(df[column_boolean].reset_index(drop=True),"\n")
    return df[column_boolean].reset_index(drop=True)


def buildTree(df,tree=None): 
    last_column = df.keys()[-1]   

    #Get attribute with maximum information gain
    node = find_max_gain(df)
    
    #Get distinct value of that attribute 
    att_value = np.unique(df[node])
    
    #Create an empty object to create tree    
    if tree is None:                    
        tree={}
        tree[node] = {}
    
    #In this we check if the subset is pure and stops if it is pure. 

    for value in att_value:
        
        subtable = get_subtable(df,node,value)
        check_value = np.unique(subtable[last_column])                        
        
        if len(check_value)==1:#Checking purity of subset
            tree[node][value] = check_value[0] 
            print("----------------------",end="\n")
        else:        
            tree[node][value] = buildTree(subtable) #Calling the function recursively 
    return tree
def test_data(train,test): 
    values = list(train.keys())
    check=False
    for value in values:
        check_value = test[value][test[value].index[0]]
        if(np.unique(train[value][check_value]).dtype=="O"):
            check=test_data(train[value][check_value],test)
        else:        
                return train[value][check_value] 
    return check

eps = np.finfo(float).eps

#save tree  
train_data = pd.read_csv(r'hotel_bookings.csv')

train=1
if train==1:
    tree=buildTree(train_data[50:],None)
    with open('DescisionTree.pickle', 'wb') as handle:
        pickle.dump(tree, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('DescisionTree.pickle', 'rb') as handle:
    tree = pickle.load(handle)
print(tree)
# create test data
print("------------------------------test here -----------------------------------------")
for i in range(50):
    print(test_data(tree,train_data[i:i+1]))






a=1