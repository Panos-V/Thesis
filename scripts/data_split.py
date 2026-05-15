import pandas as pd
import os

def split_data(input_file,train_name,test_name, test_size=0.2):

    data = pd.read_csv(input_file)
    test = data.sample(frac=test_size)
    train = data.drop(test.index)
    train.to_csv(train_name,index=False)
    test.to_csv(test_name,index=False)

    matching = pd.merge(train,test,on=['new_img_name'])
    if len(matching) > 0:
        print("Warning: There are {} duplicate entries in the train and test sets.".format(len(matching)))
    else:
        print("No duplicate entries found in the train and test sets.")

if __name__ == "__main__":
    input_file = 'raw_data/Fitzpatrick17k_full.csv'
    train_name = 'raw_data/Fitzpatrick17k_train.csv'
    test_name = 'raw_data/Fitzpatrick17k_val.csv'
    split_data(input_file, train_name, test_name)