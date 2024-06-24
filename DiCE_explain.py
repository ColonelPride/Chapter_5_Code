# import DiCE
import dice_ml
from dice_ml.utils import helpers # helper functions

# supress deprecation warnings from TF
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import pandas as pd
#import pickle
from  sklearn.neighbors import KNeighborsClassifier
from dcf_explain import *
from sklearn.preprocessing import MinMaxScaler
#X_train, y_train, X_test,dce_X_test,line,dataset,keras_model_filename, db_file,num_cfs=5
#def explain(dataset,X_test, db_file, prob_dict, line, keras_model_filename, num_cfs=10):


def dice_explain(X_train, y_train,dce_X_train, X_test,dce_X_test,line_no,dice_predicted,dataset,keras_model_filename, db_file, columns_to_vary,num_cfs=5):
    """£
    def get_line_columns(dataset):
        line_columns = dataset['X_columns']
        if dataset['use_dummies'] == True:
            line_columns = dataset['X_columns_with_dummies']
        return line_columns
    """
    """
    def get_prob_dice(df,dataset):
        prob_dict = {}
        prob_pos = 0
        prob = df[dataset['class_name']].values[0]
        prob_array = np.asarray([1-prob,prob],dtype=float).reshape(1,-1)#2 classes for binary decsion
        for outcome in dataset['possible_outcomes']:
            prob_dict[outcome] = prob_array[0][prob_pos]
            prob_pos = prob_pos+1
        return prob_dict # output in style of {'Bad':1-prob,'Good:prob'}
    """
    def get_prob_dice_2(score,cf_score,pred):
        prob_dict = {}
        if pred < 0.5:
            prob_dict[score] =(1-pred)[0]
            prob_dict[cf_score] = pred[0]
        else:
            prob_dict[score]=pred[0]
            prob_dict[cf_score]=(1-pred)[0]
        return prob_dict
    #d = dice_ml.Data(dataframe=dataset['df'], continuous_features=['annual_inc', 'open_acc','years_of_credit', 'emp_length'], outcome_name='loan_status'
    #make a new dataset of y_train and X_train as df with use_dummies
    def get_new_df(dataset, y_train, X_train):
        dice_dict = {}
        dice_dict[dataset['class_name']] = y_train
        for i in range(len(dataset['X_columns'])):
            if dataset['X_columns'][i] in dataset['continuous']:
                values = X_train[:,i].astype('int32')
                dice_dict[dataset['X_columns'][i]] =values
        for i in range(len(dataset['X_columns'])):
            if dataset['X_columns'][i] in dataset['non_continuous']:
                values  = X_train[:,i]
                dice_dict[dataset['X_columns'][i]] =values
        """
        #for use with dummies
        for i in range(len(dataset['X_columns_with_dummies'])):
            dice_dict[dataset['X_columns_with_dummies'][i]] = X_train[:,i]
        """
        dice_df = pd.DataFrame.from_dict(data= dice_dict)
        return dice_df
        #concatenate y_train and X_train

    dice_df = get_new_df(dataset, y_train, dce_X_train)

    d = dice_ml.Data(dataframe=dice_df,
                     continuous_features=dataset['continuous'],
                     categorical_features=dataset['non_continuous'],
                     outcome_name=dataset['class_name'])#, categorical_features=dataset['non_continuous'])

    #while it owuld be beter to pass the model into this function from main, the original code for DiCE loads the model and i dont think it would be time efficient to change this.
    #ML_modelpath = 'models/'+keras_model_filename
    ML_modelpath = keras_model_filename
    m = dice_ml.Model(model_path= ML_modelpath)

    exp = dice_ml.Dice(d, m)#exp is for explainer
    #create function to change line to dict
    """
    def array_to_dict(line_no,dataset,X,dice_df):
        arr = X[line_no]
        #input is line of X as array
        #line_columns = get_line_columns(dataset)
        output_dict = {}
        
        return output_dict
    """
    def array_to_list(line_no,dataset,X,):
        #needs achange to list in order   continous columns, non_continuous columns
        columns= dataset['X_columns']
        arr=X[line_no]
        new_arr = np.empty([1,arr.shape[0]])
        new_columns = list()
        for col in dataset['continuous']:
            new_columns.append(col)
        for col in dataset['non_continuous']:
            new_columns.append(col)
        new_df = pd.DataFrame(data= new_arr,columns = new_columns)
        for i in range(len(columns)):
            new_df[columns[i]] = arr[i]
        return new_df

    query_instance = array_to_list(line_no,dataset,dce_X_test,)
    query_instance_list = list(query_instance.values[0])


    print('query instance: ',query_instance)

    #num_cfs set as parameter in main
    #test_cf requires proximity = 0.5 diversity = 1
    #dice_exp = exp.generate_counterfactuals(query_instance_list, total_CFs=num_cfs,desired_class="opposite",proximity_weight = 0.5, diversity_weight =1.0, features_to_vary = columns_to_vary)

    #dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=num_cfs, desired_class="opposite",proximity_weight=0.5, diversity_weight=1.0,features_to_vary=columns_to_vary)
    dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=num_cfs, desired_class="opposite",
                                            proximity_weight=0.5, diversity_weight=1.0,
                                            features_to_vary=columns_to_vary)

    #trying just an array
    #dice_exp = exp.generate_counterfactuals(dce_X_test[line_no], total_CFs=num_cfs, desired_class="opposite")

    dice_exp.visualize_as_df()

    df = dice_exp.final_cfs_df

    #output is a df how to mesh with write db?
    #create list for dicts
    exp_list =  list()
    #loop through df.columns where query instance and df differ in value record dict
    #prediction = df[dataset['class_name']].values[0]change to dice_predicted[line_number]
    if dice_predicted [line_no] < 0.5:
        prediction = 0
    else:
        prediction = 1
    if prediction == 0:#assumes binary classifier
        score = dataset['possible_outcomes'][0]
        cf_score = dataset['possible_outcomes'][1]
    else:
        score = dataset['possible_outcomes'][1]
        cf_score = dataset['possible_outcomes'][0]
    exp_dict = {}
    exp_dict['score'] = score
    exp_dict['cf_score'] = cf_score

    #loop for number of cfs
    delta_list_dict = {}

    for i in range(df.values.shape[0]):
        delta_list = list()
        for col in df.columns:

            #removing class name restriction to get score into delta
            #if col != dataset['class_name']:#cannot give a delta on the class name

            if col == dataset['class_name']:
                delta_dict = {}
                delta_dict['att_name'] = col
                delta_dict['user_value'] = score
                delta_dict['cf_value'] = df[col][i]
                delta_list.append(delta_dict)
                
            else:
                if query_instance[col][0] != df[col][i]:
                    delta_dict = {}
                    delta_dict['att_name'] = col
                    delta_dict['user_value'] = query_instance[col]
                    delta_dict['cf_value'] = df[col][i]
                    delta_list.append(delta_dict)
                    print(delta_dict)


        delta_list_dict[str(i)] = delta_list
    exp_dict['deltas'] = delta_list_dict

    #write to database in same format as dce
    conn = ''
    #prob_dict = get_prob_dice(df,dataset)
    prob_dict = get_prob_dice_2(score,cf_score,dice_predicted[line_no])

    #create_dcf_explanation(conn,X_test[line_no],score,exp_dict,dataset,'DiCE',df[dataset['X_columns']],prob_dict,line_no,db_file)
    #create_dcf_explanation(conn,dce_X_test[line_no],score,exp_dict,dataset,'DiCE',df[dataset['X_columns']],prob_dict,line_no,db_file)
    create_dcf_explanation(conn,dce_X_test[line_no],score,exp_dict,dataset,'DiCE',df,prob_dict,line_no,db_file)#[dataset['X_columns']] removed from df to allow loan status into db
