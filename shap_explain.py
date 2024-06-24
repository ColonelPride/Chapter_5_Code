import shap
import numpy as np
import tensorflow as tf
import write_to_db
import create_db

import matplotlib.pyplot as plt
import webbrowser
#from lime_explainer import *

def get_line_columns(dataset):
    line_columns = dataset['X_columns']
    if dataset['use_dummies'] == True:
        line_columns = dataset['X_columns_with_dummies']
    return line_columns

def create_shap_explanation(conn,x,record_number,label,explanation,dataset,prob_dict,description):#col_names has no class label
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
        #this section writes the record
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

        #this section writes imp values; will change from lime to shap
        #lime uses k,v dict shap uses an array of shap values(v) and a list of line_columns(k)
        for i in range(len(line_columns)):#line_columns and shap values should be same length
            col = line_columns[i]
            val = explanation[0][0][i]#shap_values
            importances_id = write_to_db.db_add_importances(conn,explanations_id)
            operand = '='
            attributes_id = write_to_db.db_add_attributes_bundle(conn, col, operand , val)
            importances_attributes_id = write_to_db.db_add_importances_attributes(conn,importances_id,attributes_id)
        """
        #old lime section
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
        """
        #add class class_probabilities
        for key in prob_dict:
            class_probabilities_id = write_to_db.db_add_class_probabilities(conn,explanations_id)
            operand = '='
            attributes_id = write_to_db.db_add_attributes_bundle(conn, key, operand , prob_dict[key])
            class_probabilities_attributes_id = write_to_db.db_add_class_probabilities_attributes(conn,class_probabilities_id,attributes_id)

#def get_lime(blackbox,dataset,X,line_number,db_file,prob_dict,num_features,diagram): for comaprison of paprametes with shap methods
def get_shap( dataset, blackbox, X_train, X_test, line_number, db_file, prob_dict):
    def predict_label(y):
        if y < 0.5:
            return 0
        else:
            return 1
    #from make_save_keras_model
    #shape = (80,)
    #I want shape(1,80)
    #from https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor
    #arg = tf.convert_to_tensor(X, dtype=np.float32)
    line_columns = get_line_columns(dataset)

    explainer = shap.GradientExplainer(blackbox,X_train)

    shap_values = explainer.shap_values(X_test[line_number].reshape(1,-1))
    expected_value = blackbox.predict(X_test[line_number].reshape(1,-1))
    """
    #i cannot get these methods which are designed to work in notebooks to work programmatically
    force_plot = shap.force_plot(expected_value[0],shap_values[0][0],line_columns)
    force_plot.savefig('force.png')
    plt.show()
    #webbrowser.open_new_tab(force_plot)
    """
    #what parameters are required to get an exxplanantion written to db? reuse lime_explainer
    #def create_lime_explanation(conn,x,record_number,label,explanation,dataset,prob_dict,description):#col_names has no class label
    label = predict_label(blackbox.predict_proba(X_test[line_number].reshape(1,-1)))
    conn = write_to_db.create_connection(db_file)
    create_shap_explanation(conn,X_test[line_number],line_number,label,shap_values,dataset,prob_dict,'shap')
