from prepare_data import *
from sklearn.model_selection import train_test_split
import sys

import tensorflow as tf

import keras
from keras import layers
from keras.layers import Embedding
from keras.layers import Dense, Flatten
from keras import models
from keras.models import Sequential
from keras import utils
#from keras.utils.np_utils import to_categorical

from predict import *
from lime_explainer import *
from DiCE_explain import *
from optAINet_explain import *
from shap_explain import *
from create_db import *
from write_to_db import *

import sqlite3
import requests

import operator

def get_line_columns(dataset):
    line_columns = dataset['X_columns']
    if dataset['use_dummies'] == True:
        line_columns = dataset['X_columns_with_dummies']
    return line_columns

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """

    #from sqlite3 import Error
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return None

def make_save_keras_model_2(X_train, y_train, filename, dataset,
                            shuffle=False,batch_size=384):
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience= 10
        ),
        keras.callbacks.ModelCheckpoint(
            filepath= filename,
            monitor = 'val_loss',
            save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor = 0.1,
            patience = 5
        )
    ]

    uv = np.unique(y_train,return_counts=True)
    weight_dict = {}

    weight_dict[uv[0][0]] = (len(y_train)/uv[1][0])
    weight_dict[uv[0][1]] = (len(y_train)/uv[1][1])
    #print('weight_dict: ',weight_dict)

    model = models.Sequential()
    num_rows = X_train.shape[0]
    num_columns = X_train.shape[1]
    #num_results = len(dataset['possible_outcomes'])
    #add embedding and flatten layers
    #vocabulary size = 23 *12 = 276
    embedding_layer = Embedding(300,8,input_length =num_columns)
    model.add(embedding_layer)
    model.add(Flatten())

    #change to set layer sizes to compare if number of parameters of information in attributes affects performance
    model.add(layers.Dense(1, activation='sigmoid'))
    #model.add(layers.Dense(1, activation='sigmoid', input_shape= (num_columns,)))
    print(model.summary())

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])


    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=100,
        class_weight=weight_dict,
        callbacks= callbacks_list,
        validation_split=0.2
    )
    model.save('models/'+filename)
    return model

def make_save_lending_non_linear(X_train, y_train, filename, dataset,
                            shuffle=False,batch_size=384):
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience= 10
        ),
        keras.callbacks.ModelCheckpoint(
            filepath= filename,
            monitor = 'val_loss',
            save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor = 0.1,
            patience = 5
        )
    ]

    uv = np.unique(y_train,return_counts=True)
    weight_dict = {}

    weight_dict[uv[0][0]] = (uv[1][1]/len(y_train)) #weight is proportion of other class
    weight_dict[uv[0][1]] = (uv[1][0]/len(y_train)) #weight is proportion of other class

    num_rows = X_train.shape[0]
    num_columns = X_train.shape[1]

    model = models.Sequential()
    # 3 layers n_columns then 0.5 n_columns then 0.1 n_columns the output sigmoid only 1 linear layer

    model.add(layers.Dense(40, activation = 'relu', input_shape= (X_train.shape[1],)))
    model.add(layers.Dense(8, activation = 'relu', input_shape= (40,)))
    model.add(layers.Dense(1, activation='sigmoid',input_shape=(8,)))


    print(model.summary())

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=100,
        class_weight=weight_dict,
        callbacks= callbacks_list,
        validation_split=0.2
    )

    #model.save('models/'+filename)
    model.save(filename)
    return model

def make_save_keras_model(X_train, y_train, filename, dataset,
                            shuffle=False,batch_size=384):
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience= 10
        ),
        keras.callbacks.ModelCheckpoint(
            filepath= filename,
            monitor = 'val_loss',
            save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor = 0.1,
            patience = 5
        )
    ]

    uv = np.unique(y_train,return_counts=True)
    weight_dict = {}

    weight_dict[uv[0][0]] = (uv[1][1]/len(y_train)) #weight is proportion of other class
    weight_dict[uv[0][1]] = (uv[1][0]/len(y_train)) #weight is proportion of other class
    model = models.Sequential()

    num_rows = X_train.shape[0]
    num_columns = X_train.shape[1]
    #only 1 linear layer
    model.add(layers.Dense(1, activation='sigmoid',input_shape=(num_columns,)))
    #num_results = len(dataset['possible_outcomes'])
    #add embedding and flatten layers
    #vocabulary size = 23 *12 = 276

    print(model.summary())

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    #model.fit(X_train,y_train,batch_size,epochs=50,class_weight=weight_dict)
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=100,
        class_weight=weight_dict,
        callbacks= callbacks_list,
        validation_split=0.2
    )
    model.save('models/'+filename)
    return model
def main(argv):
    #arg 1 = method (string) arg2 = dbfile (string) will be in dir 'explanation_store/'
    #start executing code here


    #dataset_cat = process_lending()
    #dataset_cat = process_adult()
    #dataset_cat = process_compas()



    db_file = 'explanation_store/'+argv[2]
    conn = create_db(db_file)



    dataset_cat = pickle.load(open('pickled_data/lending_pickled_data_MinMax_27_06_22.p','rb'))#using normalized continuous features for use with dice.
    #dataset_cat = pickle.load(open('pickled_data/adult_pickled_data_MinMax_4_07_22.p','rb'))
    #dataset_cat = pickle.load(open('pickled_data/compas_pickled_data_MinMax_6_07_22.p','rb'))

    #keras_model_filename = 'keras_model_compas_test_cf.h5'
    keras_model_filename = 'models/keras_model_lending_new_data_2.h5' #most models are this one for experiment
    #keras_model_filename = '../keras_models/keras_model_lending_17_11_20.h5'
    #keras_model_filename = 'keras_model_compas_test_cf.h5'
    #keras_model_filename = 'keras_model_adult_test_cf.h5'

    X_train, X_test, y_train, y_test = train_test_split(dataset_cat['X'], dataset_cat['y'], test_size = 0.2, random_state=0)#with dummies

    def get_dce_test_train(dataset):#without dummies
        df_dce = dataset['df_dce']
        dce_X =  df_dce[dataset['X_columns']].values
        dce_y= get_possible_outcomes(df_dce,dataset['class_name'])
        #dce_y is a tuple of length 3
        dce_y = dce_y[0][dataset['class_name']].values
        dce_X_train, dce_X_test, dce_y_train, dce_y_test = train_test_split(dce_X, dce_y, test_size = 0.2, random_state=0)
        return dce_X_train, dce_X_test, dce_y_train, dce_y_test

    dce_X_train, dce_X_test, dce_y_train, dce_y_test = get_dce_test_train(dataset_cat)




    #model = make_save_lending_non_linear(X_train, y_train, keras_model_filename, dataset_cat)
    #breakpoint()
    #blackbox = keras.models.load_model('models/'+keras_model_filename)#experiment model
    blackbox = keras.models.load_model(keras_model_filename)

    y_predict_one = predict_single(blackbox,X_train[0])
    y_predict_all = predict_batch(blackbox,X_train)

    #get predicted values for test set

    #new shape reqd here
    #Invalid input shape for input Tensor("data:0", shape=(32, 32), dtype=float32). Expected shape (None, None, 32), but input has incompatible shape (32, 32)

    y_predicted = predict_batch(blackbox,X_test)
    dice_predicted = copy.deepcopy(y_predicted)




    for i in range(len(y_predicted)):
        if y_predicted[i]<0.5:
            y_predicted[i] = 0
        else:
            y_predicted[i] = 1
    y_predicted = y_predicted.astype(int)
    y_predicted = np.ravel(y_predicted)




    def descriptive_stats(y_test, y_predicted):
        #write descriptive statistics out for model.
        TP=0
        FP=0
        TN=0
        FN=0
        total= len(y_test)
        for i in range(total):
            if y_predicted[i] == 1:
                if y_predicted[i] == y_test[i]:
                    TP = TP+1
                else:
                    FP = FP+1
            else:
                if y_predicted[i] == y_test[i]:
                    TN = TN+1
                else:
                    FN=FN+1

        #accuracy
        accuracy = (TP+TN)/total
        #precision
        precision = TP/(TP+FP)
        #recall
        recall = TP/(TP+FN)
        #FN rate
        TN_rate = TN/(TN+FN)
        print('TP: ',TP,' TN: ',TN,' FP: ',FP,' FN: ',FN)
        print('accuracy: ',accuracy,' precision: ',precision,' recall: ',recall)

    def compare_y (y_test,y_predicted): #dce added from csv
        print('descriptive stats for y_test vs y_predicted')
        descriptive_stats(y_test,y_predicted)
        #get_dce prediction from csv
        y_dce_df = pd.read_csv('dcf_output/dcf_diverse_coherent_explanation_long_2_layers_hidden.csv')
        y_dce = y_dce_df['score'].values
        #y_dce = y_dce.reshape(-1,1)#results are 'Good' and 'Bad' translate to 0 and 1

        for i in range(len(y_dce)):
            y = y_dce[i].strip("''")
            if y == dataset_cat['labels'][0]:#labels is the tuple ('Bad','Good')
                y_dce[i] = 0
            else:
                y_dce[i] = 1

        print('descriptive stats for y_test vs y_dce')
        descriptive_stats(y_test,y_dce)

        print('descriptive stats for y_predicted vs y_dce')
        descriptive_stats(y_predicted,y_dce)
        breakpoint()





    def get_weight_dict(y):
        uv = np.unique(y,return_counts=True)
        weight_dict = {}
        weight_dict[uv[0][0]] = (len(y)/uv[1][0])
        weight_dict[uv[0][1]] = (len(y)/uv[1][1])
        return weight_dict
    #line_number  = 0
    number_of_lines=10 #test cf needs 500, changed to 100 for dice as dice hangs at around 200+ lines


    #line_arr = [x,y,z]#for experiment model with normalized continuous data


    if argv[1] == 'print':#print line to check their values

        #thsi section pickles the first 100 lines of X_test and the predictions form the model for checking the marking tool
        hundred_lines = X_test[0:100]
        predictions = blackbox.predict(hundred_lines)
        pickle_dict = {'X_test':hundred_lines, 'predictions':predictions}
        pickle_filename = 'pickled_data/X_test_100.p'
        outfile = open(pickle_filename,'wb')
        pickle.dump(pickle_dict,outfile)
        outfile.close()
        breakpoint()
    if argv[1] == 'shap':
        for line in range(number_of_lines):
            get_shap_exp(line, blackbox,X_train, X_test, dataset_cat, db_file)
    elif argv[1] == 'lime':
        for line in range(number_of_lines):
            get_lime_exp( line, blackbox, dataset_cat, X_test, db_file )
    elif argv[1] == 'ais':
        for line in range(number_of_lines):
            get_ais_exp(line, blackbox,X_train, y_train, X_test,dataset_cat,keras_model_filename, db_file)
    elif argv[1] == 'dice':
        for line in range(number_of_lines):
        #for line in range(317,318):
            #blackbox = keras.models.load_model('models/'+keras_model_filename)
            blackbox = keras.models.load_model(keras_model_filename)
            print('line: ',line)
            get_dice_exp( X_train, y_train,dce_X_train, X_test,dce_X_test, line, dice_predicted, blackbox, dataset_cat, keras_model_filename, db_file)
            #keras.backend.clear_session()  # clearing up keras tf models to prevent mem leak
            #get_dice_exp( line, blackbox, X_test, dce_X_test, dataset_cat, keras_model_filename, db_file )#uses dce_X_test instead of the dummied X_test
    elif argv[1] == 'dice_arr':
        for line_no in line_arr:
            #blackbox = keras.models.load_model('models/'+keras_model_filename)
            blackbox = keras.models.load_model(keras_model_filename)
            print('line: ',line_no)
            get_dice_exp( X_train, y_train,dce_X_train, X_test,dce_X_test, line_no, dice_predicted, blackbox, dataset_cat, keras_model_filename, db_file)


    elif argv[1] == 'ia':

        for line_no in range(number_of_lines):
            get_ia_exp(line_no, blackbox,X_train, y_train, X_test,dataset_cat,keras_model_filename, db_file)

    import pdb; pdb.set_trace()


def get_prob_dict(blackbox, X, line_no,dataset):
    prob_dict = {}
    prob_pos = 0
    #prob = blackbox.predict_proba(X[line_no].reshape(1,-1))[0][0]
    prob = predict_single(blackbox,X[line_no])
    prob_array = np.asarray([1-prob,prob],dtype=float).reshape(1,-1)#2 classes for binary decsion
    for outcome in dataset['possible_outcomes']:
        prob_dict[outcome] = prob_array[0][prob_pos]
        prob_pos = prob_pos+1
    return prob_dict # output in style of {'Bad':1-prob,'Good:prob'}

    #BB explainers
    """
    for line in range(20):
        prob_dict = get_prob_dict(blackbox,X_test,line,dataset_cat)
        print(line,prob_dict)
    """
def get_lime_exp( line, blackbox, dataset, X_test, db_file ):
    """
    print('*******************************************************************************')
    print('LIME')
    print('*******************************************************************************')
    print('line number: ',line)
    """
    prob_dict = get_prob_dict(blackbox,X_test,line,dataset)
    #new section get lime to only predict +ve outcome for lending only, remove to regeneralise
    #prob_dict.pop('Charged Off')
    #end of new section
    #top_labels used to explain both +ve and -ve labels https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_tabular
    get_lime(blackbox,dataset,X_test,line,db_file,prob_dict,num_features=X_test.shape[1],diagram=False)#num features = regularisation term


#get_lime(blackbox,dataset_cat,X_test,line_number,db_file,prob_dict,num_features=X_test.shape[1],diagram=False)#num features = regularisation term
#write_importances_to_csv(blackbox,dataset_cat,X_test)
# write importances to a csv for all of X_test. To get idea of importances scales for NLG.
#next steps get dce and opt-AINet

def get_dice_exp(X_train, y_train,dce_X_train,X_test, dce_X_test, line_no, dice_predicted,blackbox, dataset, keras_model_filename, db_file ):
    """
    print('*************************************************************************************')
    print('DiCE')
    print('*************************************************************************************')
    print('line number: ',line)
    """

    # hardwire invariant as
    prob_dict = get_prob_dict(blackbox,X_test,line_no,dataset)
    print('main prob dict: ',prob_dict)
    print('line number: ',line_no)


    for i in range(1):# no incremental algorithm# (len(dataset['X_columns'])):#-1 because removing 1 before calling method, will underflow otherwise
        #pop first item on invariants
        #change here for temporaery fix to keep loan-amount as invariant only for lending data
        #first_invariant = dataset['invariants'].pop(0)#side effect is to reduce invariants by first element


        columns_to_vary = dataset['X_columns'] # for test_cf columns_to_vary =
        #if columns_to_vary != []:#redundant in test_cf
            #num_cfs or k is ten in paper test, changes with dataset COMPAS is less
            #used to be indented when following if statement above
        dice_explain(X_train, y_train, dce_X_train,X_test,dce_X_test,line_no, dice_predicted,dataset,keras_model_filename, db_file,columns_to_vary,num_cfs=5)
        #keras.backend.clear_session()  # clearing up keras tf models to prevent mem leak


def get_ais_exp(line, blackbox,X_train, y_train, X_test,dataset, keras_model_filename,db_file):
    """
    print('*******************************************************************************')
    print('opt-AINet')
    print('*******************************************************************************')
    print('line number: ',line)
    """


    def get_invariants_lime(blackbox,dataset,X,line_number,num_features):
        feature_names = dataset['X_columns_with_dummies']
        invariants = list()
        def predict_fn(x):#x is 2d array
            prob = blackbox.predict_proba(x[0,:].reshape(1,-1))[0][0]
            prob_array = np.asarray([1-prob,prob],dtype=float).reshape(1,-1)#2 classes for binary decsion
            for row in range(1,x.shape[0]):
                prob = blackbox.predict_proba(x[row,:].reshape(1,-1))[0][0]
                prob_vals = np.asarray([1-prob,prob],dtype=float).reshape(1,-1)#2 classes for binary decsion
                prob_array = np.append(prob_array, prob_vals, axis = 0)
            return prob_array
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X,
            class_names=dataset['class_name'],
            feature_names=feature_names,
            categorical_features = range(len(feature_names)),#all columns are categorical
            discretize_continuous=False)

        exp = explainer.explain_instance(
            X[line_number],
            predict_fn,
            num_features = num_features
            )
        def truncate_k(k):
            l = len(k)
            new_k = ''
            for i in range(l):
                if k[i] == '=':
                    return new_k
                else:
                    new_k = new_k + k[i]
            return new_k
        for k,v in exp.as_list():#invariants are k, need to be truncates where cat because =X will be in k
            k = truncate_k(k)
            invariants.append(k)
        #print("End of lime picking invariants")
        return invariants


    LR = LinearRegression()


    LR.fit(X_train,y_train,sample_weight=None)

    prob_dict = get_prob_dict(blackbox,X_test,line,dataset)
    LR_coef = LR.coef_#importances for a linear model for lending

    #old line
    #init_var_optAINet(blackbox,X_test, line, dataset,prob_dict,db_file,LR_coef)
    #new line
    df_out = pd.DataFrame(columns=get_line_columns(dataset))
    lime_coeffs_reorder = list() # new line
    result, df_out = init_var_optAINet(blackbox, X_test, line, dataset, prob_dict, db_file, lime_coeffs_reorder, df_out)

def get_ia_exp(line, blackbox,X_train, y_train, X_test,dataset,keras_model_filename, db_file):

    print('*******************************************************************************')
    print('Incremetal Algorithm')
    print('*******************************************************************************')

    dataset['invariants'], imp_dict =get_dataset_invariants(blackbox,dataset,X_train,X_test,line)
    dataset['invariants'] = list()
    lime_coeffs_reorder = list()
    for i in range (X_train.shape[1]):
        lime_coeffs_reorder.append(imp_dict[get_line_columns(dataset)[i]])
    prob_dict = get_prob_dict(blackbox,X_test,line,dataset)
    print('main prob dict: ',prob_dict)
    print('line number: ',line)

    df_out = pd.DataFrame(columns = get_line_columns(dataset))

    result, df_out = init_var_optAINet(blackbox,X_test, line, dataset,prob_dict,db_file,lime_coeffs_reorder,df_out)
    print('df_out of length: ',df_out.values.shape[0])

def get_shap_exp(line, blackbox,X_train, X_test, dataset, db_file):
    """
    print('*******************************************************************************')
    print('SHAP')
    print('*******************************************************************************')
    """
    prob_dict = get_prob_dict(blackbox,X_test,line,dataset)
    #needed paramters (dataset, model, X_train, X_test, line, db_file, prob_dict)
    get_shap(dataset,blackbox,X_train,X_test,line, db_file, prob_dict)

def get_invariants_lime(blackbox,dataset,X,line_number,num_features):
    lime_coeffs = list()
    feature_names = get_line_columns(dataset)
    invariants = list()
    cat_index = list()
    for i in range(len(feature_names)):
        f_name = feature_names[i]
        if f_name not in dataset['continuous']:
            cat_index.append(i)

    def predict_fn(x):#x is 2d array
        #prob = blackbox.predict_proba(x[0,:].reshape(1,-1))[0][0] old line for keras 2
        #new line keras 3

        #prob = predict_batch(blackbox,x)
        #prob_array = np.asarray([1-prob,prob],dtype=float).reshape(2,-1)#2 classes for binary decsion
        prob_array = np.empty([1,2],dtype=float)
        for row in range(1,x.shape[0]):
            #prob = blackbox.predict_proba(x[row,:].reshape(1,-1))[0][0]old keras 2
            #new keras 3
            prob = predict_single(blackbox,x[row,:])
            prob_vals = np.asarray([1-prob,prob],dtype=float).reshape(-1,2)#2 classes for binary decsion
            prob_array = np.append(prob_array, prob_vals, axis = 0)

        return prob_array
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X,
        class_names=dataset['class_name'],
        feature_names=feature_names,
        categorical_features = cat_index,#all columns are categorical
        discretize_continuous=False)
    #if line_number == 4:        breakpoint()
    exp = explainer.explain_instance(
        X[line_number],
        predict_fn,
        num_features = num_features
        )
    def truncate_k(k):
        l = len(k)
        new_k = ''
        for i in range(l):
            if k[i] == '=':
                return new_k
            else:
                new_k = new_k + k[i]
        return new_k

    for k,v in exp.as_list():#invariants are k, need to be truncates where cat because =X will be in k
        k = truncate_k(k)
        invariants.append(k)
        lime_coeffs.append(v)
    #print("End of lime picking invariants")
    return invariants, lime_coeffs

def get_dataset_invariants(blackbox,dataset,X_train,X_test,line):
    #calls get_invariants_lime()
    #print('num_features: ' ,X_train.shape[1])

    invariants_list,lime_coeffs =  get_invariants_lime(blackbox,dataset,X_train,line,X_train.shape[1])
    #dataset['invariants'],lime_coeffs =  get_invariants_lime(blackbox,dataset,X_train,line,X_train.shape[1])
    #need to list importances in line_columns order currently lime coeffs in importance order
    #create dict of key = attribute, value = importance
    imp_dict ={}

    for i in range(len(invariants_list)):
        imp_dict[invariants_list[i]] = lime_coeffs[i]
        print(i,invariants_list[i],imp_dict[invariants_list[i]])
    lime_dict = {}
    for col in dataset['X_columns']:#not using dummies
        lime_dict[col]=list()

    #continuous atrributes use value, non_continuous use MAD of all values.
    #loop through X_columns  if in continuous take value else add value to list
    for i in range (X_test.shape[1]):#loop through get_line_columns and lime_coeffs
        if invariants_list[i] in dataset['continuous']:
            lime_dict[invariants_list[i]].append(abs(lime_coeffs[i]))

        else:#is non_continuous
            for dummied in dataset['dummy']:
                if invariants_list[i] in dataset['dummy'][dummied]:
                    lime_dict[dummied].append(abs(lime_coeffs[i]))
        #print('i',i,'feature: ', invariants_list[i] ,' lime coeffs: ',lime_coeffs[i])

    #for att in lime_dict:
        #there is an argument to be made that contini=uous and non_continupous are not comparable because non_contiuos are 1*1 and many*0 where continuous are afloat that multiplies ascaled attribute

    #sort lime_dict by value
    sorted_d = sorted(lime_dict.items(), key=operator.itemgetter(1), reverse=True)
    invariants = list()
    for pair in sorted_d:
        invariants.append(pair[0])
    return invariants,imp_dict
"""
MAIN below
"""
if __name__ == '__main__':#arg 1 method name (string)  arg 2 db_file_name (string)
    main(sys.argv)
