'''
	Check.py is for evaluating your model. 
	Function eval() will print out the accuracy of training and testing data. 
	To call:
        	import Check
        	Check.eval(o_train, p_train, o_test, p_test)
        
	At the end of this file, it also contains how to read data from a file.
'''
import numpy as np
import pandas as pd
import Solution as sl
#eval:
#   Input: original training labels list, predicted training labels list,
#	       original testing labels list, predicted testing labels list.
#   Output: print out training and testing accuracy
def eval(o_train, p_train, o_test, p_test):
    print('\nTraining Result!')
    accuracy(o_train, p_train)
    print('\nTesting Result!')
    accuracy(o_test, p_test)


#accuracy:
#   Input: original labels list, predicted labels list
#   Output: print out accuracy
def accuracy(orig, pred):
    
    num = len(orig)
    if(num != len(pred)):
        print('Error!! Num of labels are not equal.')
        return
    match = 0
    for i in range(len(orig)):
	    o_label = orig[i]
	    p_label = pred[i]
	    if(o_label == p_label):
	        match += 1
    print('***************\nAccuracy: ' + str(float(match)/num) + '\n***************')

def convert_dtype(data):
    """"
    Function to replace the '?' with Nan
    Correct the datatypes
    """
    data.replace('?', np.nan, inplace=True)
    data[1] = pd.to_numeric(data[1], downcast='float')
    data[2] = pd.to_numeric(data[2], downcast='float')
    data[7] = pd.to_numeric(data[7], downcast='float')
    data[10] = pd.to_numeric(data[10], downcast='float')
    data[13] = pd.to_numeric(data[13], downcast='float')
    data[14] = pd.to_numeric(data[14], downcast='float')
    return data


def readfile(filename):
    """
    Reads .txt file and converts '?' to Nan and returns the dataset
    Input: filename
    Output: pandas dataframe
    """
    dataset = pd.read_csv(filename, header=None, delimiter='\t')
    dataset = convert_dtype(dataset)

    return dataset

def run(df_train, df_test):
    """
    Funtion to fit the 
    Input: training dataframe, testing dataframe
    Output: labels in the prescribed format
    """
    X_train = df_train.iloc[:, :-1].values
    Y_train = df_train.iloc[:, -1].values.reshape(-1,1)
    X_test = df_test.iloc[:, :-1].values
    Y_test = df_test.iloc[:, -1].values.reshape(-1,1)
    classifier = sl.DecisionTree(max_depth=8)
    classifier.fit(X_train, Y_train)
    train_pred = classifier.predict(X_train)
    test_pred = classifier.predict(X_test)
    labels = [[i for sub in Y_train for i in sub], train_pred, [i for sub in Y_test for i in sub], test_pred] 

    return labels

if __name__ == '__main__':
    train_file = 'train.txt'
    test_file = 'test.txt'
    df_train = readfile(train_file)
    df_test = readfile(test_file)
    labels = run(df_train, df_test)
    if labels == None or len(labels) != 4:
        print('\nError: DecisionTree Return Value.\n')
    else:
        eval(labels[0],labels[1],labels[2],labels[3])
