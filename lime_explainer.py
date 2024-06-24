import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import write_to_db
import create_db
import pandas as pd
from predict import *
def get_line_columns(dataset):
    line_columns = dataset['X_columns']
    if dataset['use_dummies'] == True:
        line_columns = dataset['X_columns_with_dummies']
    return line_columns

def create_lime_explanation(conn,x,record_number,label,explanation,dataset,prob_dict,description):#col_names has no class label

    with(conn):
        #I have potential clash between a loop on index ant X_columns for x as array, try turning x into a dict
        def x_to_dict(x,datatset):
            dict = {}
            i=0
            line_columns = get_line_columns(dataset)
            for col in line_columns:#dataset['X_columns']:
                dict[col] = x[i]
                i=i+1
            return dict
        x = x_to_dict(x,dataset)
        explanations_id = write_to_db.db_add_explanations(conn,description,record_number)
        records_id = write_to_db.db_add_records(conn,explanations_id)
        #add record all operands will be '='
        #add class
        class_name = dataset['class_name']
        class_value = dataset['labels'][label]
        operand= '='
        #attributes_id  = db_add_attributes_bundle(conn, class_name, operand, y)
        attributes_id  = write_to_db.db_add_attributes_bundle(conn, class_name, operand, class_value)
        records_attributes_id = write_to_db.db_add_records_attributes(conn,attributes_id,records_id)
        """
        #add fidelity stat to records
        k= 'fidelity'
        v = fidelity
        attributes_id  = db_add_attributes_bundle(conn, k, operand , v)
        records_attributes_id = db_add_records_attributes(conn,attributes_id,records_id)
        """
        #add all non class columns
        #for col in dataset['columns']:
        #col_index = 0
        line_columns = get_line_columns(dataset)
        data_dict = dataset['data_human_dict']
        for col in line_columns:#dataset['X_columns']:

            att_type = col
            operand = '='
            #att_value = x[col_index]#destring to allow numerical comparison in debinning
            att_value = x[col]
            #change val by using debin and decode from write_db.py
            if dataset['use_dummies'] == True:
                if col in dataset['continuous']:
                    att_type = data_dict[col]
                    att_value = write_to_db.decode_debin_db(col,att_value,dataset)
                    attributes_id = write_to_db.db_add_attributes_bundle(conn, att_type, operand , att_value)
                    records_attributes_id = write_to_db.db_add_records_attributes(conn,attributes_id,records_id)

                else:#i scategorical only writes  records that have value 1 (not 0) to db
                    value_int = x[col]
                    if value_int == 1:
                        att_type = data_dict[col]['name']
                        att_value = data_dict[col]['value']
                        #att_value = write_to_db.decode_debin_db(col,att_value,dataset)
                        attributes_id = write_to_db.db_add_attributes_bundle(conn, att_type, operand , att_value)
                        records_attributes_id = write_to_db.db_add_records_attributes(conn,attributes_id,records_id)

            else:#not using dummies
                att_value = write_to_db.decode_debin_db(col,att_value,dataset)
                #att_value = str(att_value)
                attributes_id = write_to_db.db_add_attributes_bundle(conn, att_type, operand , att_value)
                records_attributes_id = write_to_db.db_add_records_attributes(conn,attributes_id,records_id)
                #col_index=col_index+1

        """
        for i in range(len(cf_rules)):
            #add deltas to db
            deltas_id = db_add_deltas(conn,explanations_id)
            print('deltas_id: ',deltas_id)
            operand = '='
            att_name = ''
            att_value = ''

            for k,v in (eval(cf_rules[i])).items():
                if k == 'att_name':
                    att_name = v
                elif k == 'cf_value':
                    att_value = v

            attributes_id = db_add_attributes_bundle(conn, att_name, operand ,att_value)
            rules_attributes_id = db_add_deltas_attributes(conn,deltas_id,attributes_id)
        """
        #add importances
        for k,v in explanation.as_list():#as_list not working IndexError: list index out of range use as_map instead
            #k is reading a message like ''MSinceMostRecentInqexcl7days=3'
            #truncate k at '=''
            #created specifically for lime and categorical variables where the value is included in the variable name.
            # k,v = e.g ('grade_G=0', 0.28909065192436767) truncate at '=' let record and deltas imply  if 0 or 1 is value
            #write grade_g to db as value sort it out later
            def truncate_k(k):
                l = len(k)
                new_k = ''
                for i in range(l):
                    if k[i] == '=':
                        return new_k
                    else:
                        new_k = new_k + k[i]
                return new_k
            if '=' in k:
               k = truncate_k(k)

            importances_id = write_to_db.db_add_importances(conn,explanations_id)
            operand = '='
            attributes_id = write_to_db.db_add_attributes_bundle(conn, k, operand , v)
            importances_attributes_id = write_to_db.db_add_importances_attributes(conn,importances_id,attributes_id)

        #add class class_probabilities
        for key in prob_dict:
            class_probabilities_id = write_to_db.db_add_class_probabilities(conn,explanations_id)
            operand = '='
            attributes_id = write_to_db.db_add_attributes_bundle(conn, key, operand , prob_dict[key])
            class_probabilities_attributes_id = write_to_db.db_add_class_probabilities_attributes(conn,class_probabilities_id,attributes_id)


def write_importances_to_csv(blackbox,dataset,X):
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
        feature_names=dataset['X_columns'],
        #categorical_features = feature_names,#all columns are categorical
        discretize_continuous=False)

    #empty pandas dataframe
    imp_df = pd.DataFrame(data=None,index=None,columns= dataset['X_columns'])
    for i in range(X.shape[0]):
        exp = explainer.explain_instance(
            X[i],
            predict_fn,
            labels = dataset['labels'],
            #num_features=X_train.shape[1]#num features is all
            num_features = X.shape[1]#num features regiularised to X
            )
        e = exp.as_list()
        #att_arr = np.empty((X[i].shape[0],1), dtype=float)
        att_dict = {}
        att_dict['index']=i
        #j = 0
        #change to dict or doesn't work
        for att in e:
            att_dict[att[0]]= att[1]
        #print(att_dict)
        new_df = pd.DataFrame.from_records(data=att_dict,index=['index'], columns=dataset['X_columns'])
        #print(new_df)
        #new_df = pd.DataFrame(data=att_arr.reshape(1,-1),columns=dataset['X_columns'])
        imp_df = pd.concat([imp_df,new_df],axis=0)
        print(i)
    imp_df.to_csv('importances_from_dict.csv',sep=',',columns=dataset['X_columns'])
    import pdb; pdb.set_trace()

def get_lime(blackbox,dataset,X,line_number,db_file,prob_dict,num_features,diagram):

    #feature_names = dataset['X_columns']
    feature_names = get_line_columns(dataset)#['X_columns_with_dummies']
    #feature_names = feature_names.delete(0)
    #get index on categorical features for lime tabular explainers
    cat_index = list()
    for i in range(len(feature_names)):
        f_name = feature_names[i]
        if f_name not in dataset['continuous']:
            cat_index.append(i)

    def predict_fn(x):#x is 2d array

        #prob = blackbox.predict_proba(x[0,:].reshape(1,-1))[0][0]#old keras 2 line
        #keras 3 line
        prob = predict_single(blackbox,x[0,:])
        prob_array = np.asarray([1-prob,prob],dtype=float).reshape(1,-1)#2 classes for binary decsion
        for row in range(1,x.shape[0]):
            #prob = blackbox.predict_proba(x[row,:].reshape(1,-1))[0][0]#old keras 2 line
            #keras 3 line
            prob = predict_single(blackbox,x[row])
            prob_vals = np.asarray([1-prob,prob],dtype=float).reshape(1,-1)#2 classes for binary decsion
            prob_array = np.append(prob_array, prob_vals, axis = 0)
        return prob_array #original_line
        #return prob_array[1] #testing for 1 class +ve only



    def show_diagram(explanation):
        e = explanation.as_list()
        print('e ','/', e)
        #print('original scaled line' + str(explain_line))
        print(e)
        x = []
        y = []
        colours = []
        o_value =[]
        #load conversion dataframe

        for att in e:
            lab = att[0]
            y.append(att[1])
            x.append(lab)
            if att[1] > 0:
                colours.append('green')
            else:
                colours.append('red')
        plt.title('Contribution to decision of each factor')
        plt.ylabel('Variables')
        plt.xlabel('Contribution')
        #a = plt.axes()
        #plt.invert_yaxis()
        #set(axes,'YDir','reverse')
        plt.barh(x, y, color=colours)
        plt.tight_layout()
        plt.savefig('fig.png')
        plt.show()
        plt.close()

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X,
        feature_names=feature_names,
        categorical_features =cat_index,#range(len(feature_names)),#true if all columns are categorical, this is not true but causing explanation.as_list() *** IndexError: list index out of range, should use #cat_index,
        #class_names=dataset['class_name'],
        class_names=dataset['possible_outcomes'],
        discretize_continuous=False
        )
    def predict_label(y):
        if y < 0.5:
            return 0
        else:
            return 1
    #label = predict_label(blackbox.predict_proba(X[line_number].reshape(1,-1)))#keras 2 line
    #keras 3 line
    label = predict_label(predict_single(blackbox,X[line_number]))

    exp = explainer.explain_instance(
        X[line_number],
        predict_fn,
        #labels=(1,),
        #top_labels = 1,#added 27_10_21 to explain both labels
        num_features = num_features #original
        #num_features = 5 #testing only
        )

    if diagram:
        show_diagram(exp)



    conn = write_to_db.create_connection(db_file)

    #print('lime explanation as list for line ', line_number,' with label ', label,' : ', exp.as_list())
    create_lime_explanation(conn,X[line_number],line_number,label,exp,dataset,prob_dict,'lime')

    return exp
