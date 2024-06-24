import json
import copy
import sqlite3
import requests
#import util
from  write_to_db import *
import create_db
#import lore_for_dcf
#import counterfactual as cf
import to_JSON
import pandas as pd
import numpy as np
import sklearn
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    from sqlite3 import Error
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return None
#def create_dcf_explanation(conn,x,y,cf_rules,coeffs,fidelity,dataset,description):
def create_dcf_explanation(conn,x,y,cf_rules,dataset,description,col_names,prob_dict,record_number,db_file):#col_names has no class label
    #need to take into account new data structure
    #db_file = '../explanation_store/ex_store_dce.db'
    conn = create_connection(db_file)
    def x_to_dict(x,dataset):
        dict = {}
        i=0
        for col in dataset['X_columns']:
            dict[col] = x[i]
            i=i+1
        return dict
    x = x_to_dict(x,dataset)

    with(conn):
        explanations_id = db_add_explanations(conn,description,record_number)
        records_id = db_add_records(conn,explanations_id)
        #add record all operands will be '='
        #add class
        class_name = dataset['class_name']
        class_value = y
        operand= '='
        attributes_id  = db_add_attributes_bundle(conn, class_name, operand, class_value)
        records_attributes_id = db_add_records_attributes(conn,attributes_id,records_id)
        """
        #add fidelity stat to records
        k= 'fidelity'
        v = fidelity
        attributes_id  = db_add_attributes_bundle(conn, k, operand , v)
        records_attributes_id = db_add_records_attributes(conn,attributes_id,records_id)
        """
        #add all non class columns
        #for col in dataset['columns']:
        for col in col_names:
            if col in x.keys():
                att_type = col
                operand = '='
                #att_value = str(x[col])
                att_value = x[col]
                #change val by using debin and decode from write_db.py
                #att_value = decode_debin_db(col,att_value,dataset)
                #move from mixed polytope to bins
                att_value = bin_db(col,att_value,dataset)
            else:#is score
                att_type = col
                operand = '='
                att_value = float(prob_dict[dataset['possible_outcomes'][1]])
                
                #att_value = bin_db(col,att_value,dataset)
            if description == 'DiCE':
                #debin, decode here
                #att_value = decode_debin_db(att_type,att_value,dataset)

                attributes_id = db_add_attributes_bundle(conn, att_type, operand , att_value)
                records_attributes_id = db_add_records_attributes(conn,attributes_id,records_id)
            else:
                attributes_id = db_add_attributes_bundle(conn, att_type, operand , att_value)
                records_attributes_id = db_add_records_attributes(conn,attributes_id,records_id)

        att_name_list = list()
        #for i in range(len(cf_rules)):
            #
        #add deltas to db
        #deltas_id = db_add_deltas(conn,explanations_id)
        operand = '='
        att_name = ''
        att_value = ''

        #add code here to stop repeats of same line being written to db
        #use list of att_name if k in att_name_list do not write to db
        #breakpoint()
        deltas_outer= cf_rules['deltas']

        for i in range(len(deltas_outer)):
            deltas_inner = deltas_outer[str(i)]
            #print('deltas_inner: ',deltas_inner)
            deltas_id = db_add_deltas(conn,explanations_id)
            #new loop needed for having more than one attributein a CF
            #breakpoint()
            for j in range(len(deltas_inner)):
                for k,v in (deltas_inner[j]).items():
                    if k == 'att_name':
                        att_name = v
                for k,v in (deltas_inner[j]).items():
                    if k == 'cf_value':
                        #att_value is equal to v
                        #change to allow for values now being on polytope
                        #att_value = decode_debin_db(att_name,int(v),dataset)#dependng on att_name preceeding cf_value
                        att_value = bin_db(att_name,v,dataset)
                #add code here to stop repeats of same line being written to db
                #use list of att_name if k in att_name_list do not write to db
                #if att_name not in att_name_list:#avoids repeats of same values#needs to go change to att_name AND value
                    #only allow if value in different bin ??? consider this. keeping if record avalue and delta value in same bin for now.

                #print('att_name_list: ', att_name_list,' att_name: ',att_name)

                #att_name_list.append(att_name)
                #deltas_id = db_add_deltas(conn,explanations_id) move to outer loop to stop every attribute in inner loop having own delta_id
                attributes_id = db_add_attributes_bundle(conn, att_name, operand ,att_value)
                rules_attributes_id = db_add_deltas_attributes(conn,deltas_id,attributes_id)

        """
        importances_id = db_add_importances(conn,explanations_id)
        for k,v in coeffs.items():
            operand = '='

            attributes_id = db_add_attributes_bundle(conn, k, operand , v)
            importances_attributes_id = db_add_importances_attributes(conn,importances_id,attributes_id)
        """
        #add class class_probabilities
        for key in prob_dict:
            class_probabilities_id = db_add_class_probabilities(conn,explanations_id)
            operand = '='

            attributes_id = db_add_attributes_bundle(conn, key, operand , str(prob_dict[key]))
            class_probabilities_attributes_id = db_add_class_probabilities_attributes(conn,class_probabilities_id,attributes_id)



def process_dcf(conn,idx_record2explain,dataset,X,target,X_test,prob_dict,description,db_file):
    from counterfactual import linear_explanation
    exp=linear_explanation()
    #encode X  to pd Dataframe here'
    #test if dummies used dummies shoyld not be used with dce
    col_names = dataset['X_columns']
    df2explain = pd.DataFrame(columns = col_names, data= X)
    exp.encode_pandas(df2explain)
    exp.train_logistic(target)

    #should be possible to use self.model to mark own results

    #print('dfx.values :',df2explain.iloc[idx_record2explain].values)
    df_test = pd.DataFrame(data=X_test, columns = col_names)
    text=exp.explain_entry(df2explain.iloc[idx_record2explain].values,df2explain,idx_record2explain,check_csv=False)
    """
    #section to test if DCE CFs are vailid (predict other class)

    print('score: ',text['score'],' cf_score:',text['cf_score'])
    #explain_entry(self,entry,upto=10,labels=("'good'","'bad'")):

    #trying to feed counterfactual back into own input to check that cf is valid, not working
    csv_df = pd.read_csv('dce_check.csv',sep=';')

    csv_df =csv_df.drop(columns='Unnamed: 0')

    count = csv_df.shape[0]
    csv_df = csv_df.append(df2explain,ignore_index=True)
    exp.encode_pandas(csv_df)

    for i in range(count):
        labels= ("'Good'","'Bad'")
        line = exp.explain_entry(csv_df.iloc[i].values,csv_df,i,check_csv=False)
        print('i: ',i,' result: ',line['score'])

    """
    """
    for t in text:
        print (t)
    """

    cf_rules= copy.deepcopy(text)
    #coeffs = cf.plot_coefficients(exp,filename='out.png',med_weight=False)
    #text.append(coeffs)
    with open('dcf_output/dcf_'+description+'_short.txt', 'w') as outfile:
        json.dump(text, outfile)
    #generate a full set of explanations for all data
    explain_all=False #only use if explaining one record or will repeat uselessly
    if explain_all:
        explanations=exp.explain_set(df2explain,10)
        explanations_0 = explanations[:,0]
        exp2=pd.DataFrame(columns=['score'],data = explanations_0)
        exp2.to_csv('dcf_output/dcf_'+description+'_long.csv')

    #visualise the set of linear weights ordered by their median contribution over the dataset.
    #cf.plot_coefficients(exp,filename='out.png',med_weight=False)
    """
    SQL section below
    """
    #db_file = 'explanation_store/ex_store.db'
    #conn = create_connection(db_file)
    dfx = df2explain.iloc[idx_record2explain]

    y = int(target[idx_record2explain])
    y_label = dataset['labels'][y]


    #fidelity = coeffs.pop('fidelity')
    #create_dcf_explanation(conn,dfx,y,cf_rules,coeffs,fidelity,dataset,filename)
    #towrite to db uncomment below

    #start here in morning bin results to ensure consistency, do not encode
    label_0 = dataset['possible_outcomes'][0]
    label_1 = dataset['possible_outcomes'][1]
    if prob_dict[label_0]>prob_dict[label_1]:
        if y == 0:
            create_dcf_explanation(conn,dfx,y_label,cf_rules,dataset,description,col_names,prob_dict,idx_record2explain)
        else:
            print('DCE explanation not written to db for record: ',idx_record2explain,' because explainer and model differ.')
    else:
        if y == 1:
            create_dcf_explanation(conn,dfx,y_label,cf_rules,dataset,description,col_names,prob_dict,idx_record2explain)
        else:
            print('DCE explanation not written to db for record: ',idx_record2explain,' because explainer and model differ.')
