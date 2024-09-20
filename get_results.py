import sys
import sqlite3
import numpy as np
import json
import requests
import copy
import webbrowser
import pickle
import csv
import math
import pandas as pd
from sklearn.neighbors import  KNeighborsClassifier
import tensorflow as tf
import keras

from predict import predict_single


#import present_explanation
def create_connection(db_file):
    """ create a database connection to a database that resides
        in the memory
    """
    #db_file = 'explanation_store/ex_store.db'
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    return conn

def get_record_description(conn,e):
    sql = '''SELECT description,record_number FROM explanations
    WHERE explanation_id = ?
        '''
    c = conn.cursor()
    c.execute(sql,e)
    rows = c.fetchall()
    return rows

def get_exp_ids(conn):
    sql = '''SELECT explanation_id FROM explanations'''
    c = conn.cursor()
    c.execute(sql)
    rows = c.fetchall()
    return rows

def get_deltas(conn,e):
    sql = '''SELECT explanation_id,delta_id FROM deltas
    WHERE explanation_id = ?
        '''
    c = conn.cursor()
    c.execute(sql, e,)
    rows = c.fetchall()
    return rows

def get_deltas_attributes(conn,e):
    sql = '''SELECT deltas.explanation_id,deltas.delta_id,
    attributes_deltas.attribute_id, attributes.attribute_value,
    operands.operand_value, attribute_types.attribute_name
    FROM deltas
    LEFT JOIN attributes_deltas ON deltas.delta_id = attributes_deltas.delta_id
    LEFT JOIN attributes ON attributes_deltas.attribute_id = attributes.attribute_id
    LEFT JOIN attributes_operands_types ON attributes.attributes_operands_types_id = attributes_operands_types.attributes_operands_types_id
    LEFT JOIN operands ON attributes_operands_types.operand_id = operands.operand_id
    LEFT JOIN attribute_types ON attributes_operands_types.attribute_type_id = attribute_types.attribute_type_id
    WHERE explanation_id = ?
    '''
    c = conn.cursor()
    c.execute(sql, e,)
    rows = c.fetchall()
    return rows
def get_differences_attributes(conn,e):
    sql = '''SELECT deltas.explanation_id,deltas.delta_id,
    differences_attributes.attribute_id, attributes.attribute_value,
    operands.operand_value, attribute_types.attribute_name
    FROM deltas
    LEFT JOIN differences_attributes ON deltas.delta_id = differences_attributes.delta_id
    LEFT JOIN attributes ON differences_attributes.attribute_id = attributes.attribute_id
    LEFT JOIN attributes_operands_types ON attributes.attributes_operands_types_id = attributes_operands_types.attributes_operands_types_id
    LEFT JOIN operands ON attributes_operands_types.operand_id = operands.operand_id
    LEFT JOIN attribute_types ON attributes_operands_types.attribute_type_id = attribute_types.attribute_type_id
    WHERE explanation_id = ?
    '''
    c = conn.cursor()
    c.execute(sql, e,)
    rows = c.fetchall()
    return rows


def get_rules_attributes(conn,e):
    sql = '''SELECT rules.explanation_id,rules.rule_id,
    rules_attributes.attribute_id, attributes.attribute_value,
    operands.operand_value, attribute_types.attribute_name
    FROM rules
    LEFT JOIN rules_attributes ON rules.rule_id = rules_attributes.rule_id
    LEFT JOIN attributes ON rules_attributes.attribute_id = attributes.attribute_id
    LEFT JOIN attributes_operands_types ON attributes.attributes_operands_types_id = attributes_operands_types.attributes_operands_types_id
    LEFT JOIN operands ON attributes_operands_types.operand_id = operands.operand_id
    LEFT JOIN attribute_types ON attributes_operands_types.attribute_type_id = attribute_types.attribute_type_id
    WHERE explanation_id = ?
    '''
    c = conn.cursor()
    c.execute(sql, e,)
    rows = c.fetchall()
    return rows

def get_records_attributes(conn,e):
    sql = '''SELECT records.explanation_id,records.record_id,
    records_attributes.attribute_id, attributes.attribute_value,
    operands.operand_value, attribute_types.attribute_name
    FROM records
    LEFT JOIN records_attributes ON records.record_id = records_attributes.record_id
    LEFT JOIN attributes ON records_attributes.attribute_id = attributes.attribute_id
    LEFT JOIN attributes_operands_types ON attributes.attributes_operands_types_id = attributes_operands_types.attributes_operands_types_id
    LEFT JOIN operands ON attributes_operands_types.operand_id = operands.operand_id
    LEFT JOIN attribute_types ON attributes_operands_types.attribute_type_id = attribute_types.attribute_type_id
    WHERE explanation_id = ?
    '''
    c = conn.cursor()
    c.execute(sql, e,)
    rows = c.fetchall()
    return rows

def get_importances_attributes (conn,e):
    sql = '''SELECT importances.explanation_id,importances.importance_id,
    importances_attributes.attribute_id, attributes.attribute_value,
    operands.operand_value, attribute_types.attribute_name
    FROM importances
    LEFT JOIN importances_attributes ON importances.importance_id = importances_attributes.importance_id
    LEFT JOIN attributes ON importances_attributes.attribute_id = attributes.attribute_id
    LEFT JOIN attributes_operands_types ON attributes.attributes_operands_types_id = attributes_operands_types.attributes_operands_types_id
    LEFT JOIN operands ON attributes_operands_types.operand_id = operands.operand_id
    LEFT JOIN attribute_types ON attributes_operands_types.attribute_type_id = attribute_types.attribute_type_id
    WHERE explanation_id = ?
    '''
    c = conn.cursor()
    c.execute(sql, e,)
    rows = c.fetchall()
    return rows

def get_invariants_attributes(conn,e):
    sql = '''SELECT records.explanation_id,records.record_id,
    invariants_attributes.attribute_id, attributes.attribute_value,
    operands.operand_value, attribute_types.attribute_name
    FROM records
    LEFT JOIN invariants_attributes ON records.record_id = invariants_attributes.record_id
    LEFT JOIN attributes ON invariants_attributes.attribute_id = attributes.attribute_id
    LEFT JOIN attributes_operands_types ON attributes.attributes_operands_types_id = attributes_operands_types.attributes_operands_types_id
    LEFT JOIN operands ON attributes_operands_types.operand_id = operands.operand_id
    LEFT JOIN attribute_types ON attributes_operands_types.attribute_type_id = attribute_types.attribute_type_id
    WHERE explanation_id = ?
    '''
    c = conn.cursor()
    c.execute(sql, e,)
    rows = c.fetchall()
    return rows

def get_class_probabilities_attributes(conn,e):
    sql = '''SELECT class_probabilities.explanation_id,class_probabilities.class_probability_id,
    class_probabilities_attributes.attribute_id, attributes.attribute_value,
    operands.operand_value, attribute_types.attribute_name
    FROM class_probabilities
    LEFT JOIN class_probabilities_attributes ON class_probabilities.class_probability_id = class_probabilities_attributes.class_probability_id
    LEFT JOIN attributes ON class_probabilities_attributes.attribute_id = attributes.attribute_id
    LEFT JOIN attributes_operands_types ON attributes.attributes_operands_types_id = attributes_operands_types.attributes_operands_types_id
    LEFT JOIN operands ON attributes_operands_types.operand_id = operands.operand_id
    LEFT JOIN attribute_types ON attributes_operands_types.attribute_type_id = attribute_types.attribute_type_id
    WHERE explanation_id = ?
    '''
    c = conn.cursor()
    c.execute(sql, e,)
    rows = c.fetchall()
    return rows

def create_exp_object(record,delta,rule,importance):
    att_value = {}
    for i in range(len(record)):
        att_value.update(record)
    explanation = {

    }
    return explanation

def update_delta(i,exp_deltas):
    key = i
    name = exp_deltas[0][i][5]
    operand = exp_deltas[0][i][4]
    val = exp_deltas[0][i][3]
    value = (name,operand,val)
    dict_object = {key: value}
    return dict_object
"""
#new method only return name:value
def update_delta(i,exp_deltas):
    key = i
    name = exp_deltas[0][i][5]
    operand = exp_deltas[0][i][4]
    val = exp_deltas[0][i][3]
    value = (name,operand,val)
    dict_object = {name: val}
    return dict_object
"""
def get_delta_ids(exp_deltas):
    delta_ids = list()
    for i in range(len(exp_deltas[0])):
        delta_ids.append(exp_deltas[0][i][1])
    return delta_ids

def update_rule(i,exp_rule):
    key = i
    name = exp_rule[0][i][5]
    operand = exp_rule[0][i][4]
    val = exp_rule[0][i][3]
    value = (name,operand,val)
    dict_object = {key: value}
    return dict_object

def get_rule_ids(exp_rule):
    rule_ids = list()
    for i in range(len(exp_rule[0])):
        rule_ids.append(exp_rule[0][i][1])
    return rule_ids

def update_importances(i,exp_importances):
    key = i
    if exp_importances[0][i][5]:
        name = exp_importances[0][i][5]
        operand = exp_importances[0][i][4]
        val = float(exp_importances[0][i][3])
        value = (name,operand,val)
        dict_object = {key: value}
    else:
        dict_object ={}
    return dict_object

def get_importances_ids(exp_importances):
    importances_ids = list()
    for i in range(len(exp_importances[0])):
        importances_ids.append(exp_importances[0][i][1])
    return importances_ids

def update_class_probabilities(i,exp_class_probabilities):
    key = i
    name = exp_class_probabilities[0][i][5]
    operand = exp_class_probabilities[0][i][4]
    val = float(exp_class_probabilities[0][i][3])
    value = (name,operand,val)
    dict_object = {key: value}

    return dict_object

def get_class_probabilities_ids(exp_class_probabilities):
    class_probabilities_ids = list()
    for i in range(len(exp_class_probabilities[0])):
        class_probabilities_ids.append(exp_class_probabilities[0][i][2])
    return class_probabilities_ids

def update_record(i,exp_record):
    key = i
    name = exp_record[0][i][5]
    operand = exp_record[0][i][4]
    val = exp_record[0][i][3]
    value = (name,operand,val)
    dict_object = {key: value}
    return dict_object

def get_record_ids(exp_record):
    record_ids = list()
    for i in range(len(exp_record[0])):
        record_ids.append(exp_record[0][i][1])
    return record_ids

def create_explanation(conn, e, dataset, translate = False):
        def translate_attribute(dataset,attribute):
            data_human_dict = dataset['data_human_dict']
            if attribute in data_human_dict.keys():
                attribute = data_human_dict[attribute]
            return attribute
        #get counterfactual for explanations

        exp_record = list()
        exp_record.append(get_records_attributes(conn,e))
        record_ids = get_record_ids(exp_record)
        record_list = []
        att_item = {}
        val_item = {}
        exp_invariants = list()
        exp_invariants.append(get_invariants_attributes(conn,e))

        for i in range(len(record_ids)):
            combined_item ={}
            att_item = {}
            val_item = {}
            inv_item = {}
            record_object = update_record(i,exp_record)
            attribute = str(record_object[i][0])
            value  = str(record_object[i][2])
            if translate:
                attribute = translate_attribute(dataset,attribute)
            att_item['field'] = attribute
            val_item['value'] = value
            inv_item['invariant'] = False
            #add invariant dict value true
            for inv in exp_invariants[0]:
                #import pdb; pdb.set_trace()
                if attribute == inv[5] :
                    inv_item['invariant']= True
            combined_item.update(att_item)
            combined_item.update(val_item)
            combined_item.update(inv_item)
            record_list.append(combined_item)
        record_update = {'record' : record_list}


        exp_deltas = list()
        exp_deltas.append(get_deltas_attributes(conn,e))
        delta_ids = get_delta_ids(exp_deltas)

        delta_dict = {}
        delta_id_arr = np.unique(delta_ids)
        exp_differences = list()
        exp_differences.append(get_differences_attributes(conn,e))

        exp_invariants_names = list()
        for inv in exp_invariants[0]:
            exp_invariants_names.append(inv[5])
        for delta_id in delta_id_arr:
            delta_list = list()
            for i in range(len(exp_deltas[0])):
                #add only if record invariant: False
                if exp_deltas[0][i][1] == delta_id:
                    delta_i_dict = {}
                    if exp_deltas[0][i][5] not in exp_invariants_names:#is this causing bugs? No

                        #also remove distance and cost
                        #change 28/6/22 put distance and cost back in
                        #if exp_deltas[0][i][5] not in ['distance','cost']:
                        dict_item ={}
                        dict_i_object = update_delta(i,exp_deltas)
                        field_value = str(dict_i_object[i][0])
                        if translate:
                            field_value = translate_attribute(dataset,field_value)
                        value_value = str(dict_i_object[i][2])
                        delta_i_dict['field'] = field_value
                        delta_i_dict['value'] = value_value
                        #new part here to add record value and operands
                        for rec in exp_record[0]:
                            if translate:
                                record_field = translate_attribute(dataset,rec[5])
                                if delta_i_dict['field'] == record_field:
                                    delta_i_dict['record_value'] = translate_attribute(dataset,rec[3])
                                    delta_i_dict['record_operand'] = rec[4]
                            else:
                                if delta_i_dict['field'] == rec[5]:
                                    delta_i_dict['record_value'] = rec[3]
                                    delta_i_dict['record_operand'] = rec[4]

                        #redundant for paper_one, not redundant for lending
                        #add diff if field matches
                        for diff in exp_differences[0]:

                            if field_value == diff[5] and delta_id == diff[1]:
                                delta_i_dict['difference'] = str(diff[3])
                                #add original value to delta_i_dict for nlg production

                                for rec in record_list:
                                    if rec['field'] == field_value:
                                        original = rec['value']
                                        delta_i_dict['original']=original


                        #translate from data to human

                            """
                            #this section causes problem with producing double lines in explanation
                            #only update delta_list if value is different to the record
                            #get record value from record list

                            #not worth doing this where all data is in buckets
                            for rec in record_list:
                                if rec['field'] == field_value:
                                    #problems with too much precision  '3.00000000004' != 3.0
                                    # using round(number[,digits])
                                    #what if we use strings????

                                    if round(float(rec['value']),3) != round(float(value_value),3):
                                         delta_list.append(delta_i_dict)
                            """
                            # only write if value of delta != value of record
                            #if delta_i_dict['value'] != delta_i_dict['record_value']:#is this causing error? possibly

                            #proble 1/7/22 some deltas contain repeated features need to remove these jow?
                            copy_delta_list = copy.deepcopy(delta_list)
                            if len(copy_delta_list)>0:
                                last_element = copy_delta_list.pop()
                                if last_element != delta_i_dict:
                                    delta_list.append(delta_i_dict)
                            else:
                                delta_list.append(delta_i_dict)

                delta_dict[str(delta_id)]=delta_list

        d_update = {'delta' : delta_dict}

        #get rules
        exp_rules = list()
        exp_rules.append(get_rules_attributes(conn,e))
        rule_ids = get_rule_ids(exp_rules)
        rule_dict = {}
        for i in range(len(rule_ids)):
            rule_object = update_rule(i,exp_rules)
            rule_dict.update(rule_object)
        rules_update = {'rule' : rule_dict}

        exp_importances = list()
        exp_importances.append(get_importances_attributes(conn,e))
        importances_ids = get_importances_ids(exp_importances)
        importances_dict = {}
        for i in range(len(importances_ids)):
            importances_object = update_importances(i,exp_importances)
            importances_dict.update(importances_object)
        i_update = {'importances' : importances_dict}

        #get top X importances
        #make new dict from top X with name, operand, and positive (bool)
        def top_X_imp(importances_dict,record_list,top_X):
            top_X_list = list()
            top_X_list_pos = list()
            top_X_list_neg = list()
            for x in range(top_X):
                att = importances_dict[x][0]
                #get record value for att as {'field': x_str,'value':y_str}
                att_dict = {}
                att_dict['field']=att
                for rec in record_list:
                    if rec['field'] == att:
                        att_dict['value'] = rec['value']
                #top_X_list.append(att_dict)#do not add whole dict only 'field' value aka att
                top_X_list.append(att)
                #operand = importances_dict[x][1]
                #positive = True
                if float(importances_dict[x][2]) < 0:
                    top_X_list_neg.append(att_dict)
                else:
                    top_X_list_pos.append(att_dict)

            return top_X_list,top_X_list_pos,top_X_list_neg

        #get record_number

        record_description = get_record_description(conn,e)
        r_number = {'record_number':record_description[0][1]}
        description = {'description':record_description[0][0]}
        top_X_list = list()
        top_X_list_pos = list()
        top_X_list_neg = list()
        if len(importances_dict)>0:
            top_X_list,top_X_list_pos,top_X_list_neg = top_X_imp(importances_dict,record_list,5)
        top_X_dict = {}
        top_X_dict['top_X_list']= top_X_list
        top_X_dict['top_X_list_pos']= top_X_list_pos
        top_X_dict['top_X_list_neg']= top_X_list_neg
        top_X_update = {'top_X_importances':top_X_dict}

        #get class probabilities
        exp_class_probabilities = list()
        exp_class_probabilities.append(get_class_probabilities_attributes(conn,e))
        class_probabilities_ids = get_class_probabilities_ids(exp_class_probabilities)
        class_probabilities_dict = {}
        for i in range(len(class_probabilities_ids)):
            class_probabilities_object = update_class_probabilities(i,exp_class_probabilities)
            class_probabilities_dict.update(class_probabilities_object)
        cp_update = {'class_probabilities' : class_probabilities_dict}

        exp = {}
        e_number = {'number':str(e[0])}

        exp.update(e_number)
        exp.update(r_number)
        exp.update(description)
        exp.update(d_update)
        exp.update(rules_update)
        exp.update(i_update)
        exp.update(top_X_update)
        exp.update(record_update)
        exp.update(cp_update)
        exp.update(r_number)

        #e_up = {e[0]:exp}
        #explanation.append(e_up)

        return exp
def decode_debin_db(k,val_string,dataset,decode=True):
    val_arr = [val_string] #decode needs an array
    if decode:
        if k in dataset['label_encoder']:
            val_arr = decode_db(k,val_arr,dataset)[0]
    else:
        val_arr = val_arr[0]
    if k in dataset['binner']:
        if val_arr >= 0:#all special values <0
            val_arr = debin_db(k,val_arr,dataset)
        else:
            #using definition of special values from https://community.fico.com/s/explainable-machine-learning-challenge?tabset-3158a=413df
            if val_arr == -9:
                val_arr = 'No Bureau Record or No Investigation'
            elif val_arr == -8:
                val_arr = 'No Usable Values'
            elif val_arr == -7:
                val_arr = 'Condition not Met (e.g. No Inquiries, No Late Payments)'
            else:
                val_arr = str(val_arr)
    else:#will be encoded
        if k == 'MaxDelq2PublicRecLast12M':
            if val_arr == 0:
                val_arr = 'derogatory comment'
            elif val_arr == 1:
                val_arr = '120+ days delinquent'
            elif val_arr == 2:
                val_arr =  '90 days delinquent'
            elif val_arr == 3:
                val_arr =  '60 days delinquent'
            elif val_arr == 4:
                val_arr =  '30 days delinquent'
            elif val_arr == 5 or 6:
                val_arr =  'unknown delinquency'
            elif val_arr == 7:
                val_arr =  'current and never delinquent'
            elif val_arr == 8 or 9:
                val_arr =  'all other'
            else:
                val_arr=str(val_arr)
        if k == 'MaxDelqEver':
            if val_arr == 1:
                val_arr = 'No such value'
            elif val_arr == 2:
                val_arr = 'derogatory comment'
            elif val_arr == 3:
                val_arr =  '120+ days delinquent'
            elif val_arr == 4:
                val_arr =  '90 days delinquent'
            elif val_arr == 5:
                val_arr =  '60 days delinquent'
            elif val_arr == 6:
                val_arr =  '30 days delinquent'
            elif val_arr == 7:
                val_arr =  'unknown delinquency'
            elif val_arr == 8:
                val_arr =  'current and never delinquent'
            elif val_arr == 9:
                val_arr =  'all other'
            else:
                val_arr=str(val_arr)
    return val_arr #remove array


def get_validity(explanation,delta_str,class_name,labels):#explantion is explanation[i][i+1]
    #get score from class-class_probabilities
    #need to get result from record class name

    def get_label(explanation,class_name):
        for rec in explanation['record']:
            if rec['field'] == class_name:
                return rec['value']
    def get_score( explanation,label):
        print('label: ',label)
        if explanation['description']=='optAINet':
            for i in range(len(explanation['class_probabilities'])):
                if explanation['class_probabilities'][i][0] == label:
                    return explanation['class_probabilities'][i][2]
            print('optAINet get score not working' )
            breakpoint()
        else:#is DiCE
            for rec in explanation['record']:
                if rec['field'] == class_name:
                    #for an unknown reason there are two fieds with class namwe I want the second which does not hace the label as the value
                    if rec['value']!=label:
                        return float(rec['value'])
            print('No numerical score found in record')
            breakpoint()
        """
        if label == explanation['class_probabilities'][0][0]:#assumes first label -ve second +ve
            score =  explanation['class_probabilities'][0][2]
        else:#only works for binary labels
            score = explanation['class_probabilities'][1][2]
        return explanation['class_probabilities'][1][2]
        """


    label = get_label(explanation,class_name)
    score = get_score (explanation,label)
    print('explanation record: ',explanation['record'],'explanation delta: ',delta_str,' label: ',label,' score: ',score)
    #score = explanation['class_probabilities'][0][2]#score for +ve class
    deltas = explanation['delta'][delta_str]
    if explanation['description'] == 'optAINet':
        for delta in deltas:
            if delta['field'] == 'cost':
                if float(delta['value']) > 0:
                    print('True ', delta['value'])
                    return True, delta['value']#always valid by design
                else:
                    print('False ', delta['value'])
                    return False, delta['value']
    else:# is DiCE
        for delta in deltas:
            if delta['field'] == class_name:
                db_prox = abs(0.5-float(delta['value']) )
                stop_delta = 'NA'
                if score >= 0.5:
                    if  float(delta['value']) < 0.5:
                        print('True ', delta['value'])
                        if delta_str == stop_delta:
                            breakpoint()
                        return True,  db_prox
                    else:
                        print('False ', delta['value'])
                        if delta_str == stop_delta:
                            breakpoint()
                        return False, db_prox
                else:# record is false
                    if  float(delta['value']) < 0.5:
                        print('False ', delta['value'])
                        if delta_str == stop_delta:
                            breakpoint()
                        return False, db_prox
                    else:
                        print('True ', delta['value'])
                        if delta_str == stop_delta:
                            breakpoint()
                        return True,  db_prox


        #if del['field'] == what if field is not cost?
    # if it is possible to get marking scheme in from  experiment this can be changed until then
    #returns True for valid cf else False



def get_write_diversity(dataset,explanation,csv_div_filename,index):#func gets diversity cat, cont and count for all deltas in this record

    def set_cat_dist_b(val_a,val_b,cat_dist,cat_count):
        if val_a != val_b:
            cat_dist += 1
        cat_count +=1
        return cat_dist, cat_count
    def set_cat_dist(delta_i,delta_j,cat_dist,cat_count):
        if delta_i['value'] != delta_j['value']:
            cat_dist += 1
        cat_count +=1
        return cat_dist, cat_count

    def set_count_dist_b(val_a,val_b,count_dist,count_count):
        if val_a != val_b:
            count_dist += 1
        count_count +=1
        return count_dist, count_count

    def set_count_dist(delta_i,delta_j,count_dist,count_count):
        if delta_i['value'] != delta_j['value']:
            count_dist += 1
        count_count +=1
        return count_dist, count_count

    def set_cont_dist_b(val_a,val_b,cont_dist,cont_count,dataset,data_field):
        cont_dist += abs(float(val_a) - float(val_b))/dataset['mads'][data_field]
        cont_count += 1
        return cont_dist, cont_count

    def set_cont_dist(delta_i,delta_j,cont_dist,cont_count,dataset,data_field):
        #print(delta_i, delta_j)
        #print(delta_i['value'], delta_j['value'])
        #print(delta_i['value'],delta_j['value'],dataset['mads'][data_field])
        cont_dist += abs(float(delta_i['value']) - float(delta_j['value']))/dataset['mads'][data_field]
        cont_count += 1
        #print('cont_dist:',cont_dist, ' cont_count:',cont_count)
        return cont_dist, cont_count

    def div_dist(deltas,i,j,record):
        #get 2 instance i and j for all cat features get distance

        delta_i = deltas[str(i)]
        delta_j = deltas[str(j)]
        #loop through all features get cat features
        cat_dist = 0
        cat_count = 0
        cont_dist = 0
        cont_count = 0
        count_dist = 0
        count_count = 0
        #delta_2 = list()
        def get_delta_value(deltas,r_field, r_value):
            for d_dict in deltas:
                if d_dict['field'] == r_field:
                    return d_dict['value'] #if
            return r_value #delta for field not in deltas return record value

        for r_dict in record:
            r_field = r_dict['field']
            r_value = r_dict['value']

            delta_i_val = get_delta_value(delta_i,r_field, r_value)

            delta_j_val = get_delta_value(delta_j,r_field, r_value)

            if r_field in dataset['human_data_dict']:

                data_field = dataset['human_data_dict'][r_field]
                if data_field  in dataset['non_continuous']:

                    #if data_field in delta_b use delta_b
                    cat_dist, cat_count = set_cat_dist_b(delta_i_val,delta_j_val,cat_dist,cat_count)
                    count_dist, count_count =  set_count_dist_b(delta_i_val ,delta_j_val,count_dist,count_count)
                if data_field in dataset['continuous']:
                    #if delta_2 == list():                        #breakpoint()
                    cont_dist, cont_count = set_cont_dist_b(delta_i_val,delta_j_val,cont_dist,cont_count,dataset,data_field)
                    count_dist, count_count =  set_count_dist_b(delta_i_val,delta_j_val,count_dist,count_count)

            elif r_field == 'purpose of loan':
                cat_dist, cat_count = set_cat_dist_b(delta_i_val,delta_j_val,cat_dist,cat_count)
                count_dist, count_count =  set_count_dist_b(delta_i_val,delta_j_val,count_dist,count_count)

            elif r_field == 'credit rating grade' :
                cat_dist, cat_count = set_cat_dist_b(delta_i_val,delta_j_val,cat_dist,cat_count)
                count_dist, count_count =  set_count_dist_b(delta_i_val,delta_j_val,count_dist,count_count)

        #what if difference in deltas betwween opt-AiNET proesenting all features and Dice only presenting changes from the record
        """
        for delta in delta_i:#range(len(delta_i)):
            for delta_b in delta_j:
                #if data_field in delta_b use delta_b else use record
                #what if difference in deltas betwween opt-AiNET proesenting all features and Dice only presenting changes from the record
                if delta['field'] == delta_b['field']:
                    delta_2 = delta_b
                    if delta['field'] in dataset['human_data_dict']:
                        data_field = dataset['human_data_dict'][delta['field']]
                        if data_field  in dataset['non_continuous']:
                            #if data_field in delta_b use delta_b
                            cat_dist, cat_count = set_cat_dist(delta,delta_2,cat_dist,cat_count)
                            count_dist, count_count =  set_count_dist(delta,delta_2,count_dist,count_count)
                        if data_field in dataset['continuous']:
                            if delta_2 == list():
                                breakpoint()
                            cont_dist, cont_count = set_cont_dist(delta,delta_2,cont_dist,cont_count,dataset,data_field)
                            count_dist, count_count =  set_count_dist(delta,delta_2,count_dist,count_count)

                    elif delta['field'] == 'purpose of loan':
                        cat_dist, cat_count = set_cat_dist(delta,delta_2,cat_dist,cat_count)
                        count_dist, count_count =  set_count_dist(delta,delta_2,count_dist,count_count)

                    elif delta['field'] == 'credit rating grade' :
                        cat_dist, cat_count = set_cat_dist(delta,delta_2,cat_dist,cat_count)
                        count_dist, count_count =  set_count_dist(delta,delta_2,count_dist,count_count)
        """
        cat_count, cont_count, count_count = get_counts(record,dataset)
        if cat_count>0:
            cat_dist = (1/cat_count)*cat_dist
        else:
            cat_dist = (1/1)*cat_dist
        if cont_count>0:
            cont_dist = (1/cont_count)*cont_dist
        else:
            cont_dist= (1/1)*cont_dist
        if cat_count>0:
            count_dist= (1/count_count)*count_dist
        else:
            count_dist = (1/1)*count_dist


        return cat_dist, cat_count, cont_dist, cont_count, count_dist,count_count
        #1,1 will become i and j when I have finished testing
    i = len(explanation[index][index+1]['delta']) -1
    num_of_cfs = len(explanation[index][index+1]['delta'])
    delta_ids = explanation[index][index+1]['delta'].keys() #array of str keys

    cat_dist_total = 0
    cont_dist_total = 0
    count_dist_total = 0
    cat_count_total = 0
    cont_count_total = 0
    count_count_total = 0
    while i > 0 :
        old_keys = list(explanation[index][index+1]['delta'].keys())
        new_keys = list()
        for key in old_keys:
            new_keys.append(int(key))
        j = i-1
        while j >= 0:
            cat_dist, cat_count, cont_dist, cont_count, count_dist,count_count = div_dist(explanation[index][index+1]['delta'],new_keys[i],new_keys[j],explanation[index][index+1]['record'])
            cat_dist_total += cat_dist
            cont_dist_total += cont_dist
            count_dist_total += count_dist
            cat_count_total += cat_count
            cont_count_total += cont_count
            count_count_total += count_count
            cat_dist = 0
            cont_dist = 0
            count_dist = 0
            j = j-1

        i = i -1

    cat_count, cont_count, count_count = get_counts(explanation[i][i+1]['record'],dataset)
    if cat_count_total > 0:
        #cat_diversity = cat_dist_total * (1/(num_of_cfs**2))
        fac_top = math.factorial(num_of_cfs)
        fac_bottom = 2*math.factorial(num_of_cfs-2)
        cat_diversity = cat_dist_total * (1/(fac_top/fac_bottom))        #testing combinations instead of num cfs
    else:
        cat_diversity = 0#-1
    if cont_count_total > 0:
        #cont_diversity =  cont_dist_total * (1/(num_of_cfs**2))
        fac_top = math.factorial(num_of_cfs)
        fac_bottom = 2*math.factorial(num_of_cfs-2)
        cont_diversity =  cont_dist_total * (1/(fac_top/fac_bottom))  #testing combinations instead of num cfs
    else :
        cont_diversity = 0#-1

    number_of_features = len(explanation[index][index+1])-1#-1 because of class label
    if count_count_total > 0:
        #count_diversity = count_dist_total * (1/(num_of_cfs**2) )#alreadt div by num of feautres in original function
        fac_top = math.factorial(num_of_cfs)
        fac_bottom = 2*math.factorial(num_of_cfs-2)
        count_diversity = count_dist_total *(1/(fac_top/fac_bottom)) #testing combinations instead of num cfs
    else:
        count_diversity= 0

    return cat_diversity, cont_diversity, count_diversity
    #writer.writerow(cat_diversity, cont_diversity, count_diversity)

def get_counts(record,dataset):
    cont_count = 0
    cat_count = 0
    for r_dict in record:
        #print(r_dict['field'],r_dict['value'])
        if r_dict['field'] in dataset['human_data_dict']:
            data_field = dataset['human_data_dict'][r_dict['field']]
            if data_field in dataset['continuous']:
                cont_count +=1
            if data_field in dataset['non_continuous']:
                cat_count +=1
        else:
            if r_dict['field'] == 'purpose of loan':
                cat_count +=1
            if r_dict['field'] == 'credit rating grade' : #horrible fix better done in human data dict in dataset
                 cat_count +=1
    #return cat_count, cont_count, cat_count + cont_count# last varible is count_count
    return len(dataset['non_continuous']),len(dataset['continuous']),len(dataset['continuous']) + len(dataset['non_continuous'])#because count are bugged getting direct from dataset

def get_stats_write_stats(dataset,explanation,csv_prox_filename, csv_div_filename):
    #writes diversity file
    with open(csv_div_filename,'w',encoding='UTF8') as div_f:
        writer = csv.writer(div_f)
        header = ['record_id', 'cat_diversity', 'cont_diversity', 'count_diversity']
        writer.writerow(header)
        for i in range(len(explanation)):
            cat_div, cont_div, count_div = get_write_diversity(dataset,explanation,csv_div_filename,i)
            writer.writerow([i, cat_div, cont_div, count_div])
        div_f.close()
    #writes proximity file
    with open(csv_prox_filename,'w',encoding='UTF8') as f:
        writer = csv.writer(f)
        header = ['record_id', 'delta_id', 'valid', 'cost','cont_proximity', 'cat_proximity', 'sparsity']
        writer.writerow(header)

        #create blank array of len 10
        #continuous_arr = np.zeros(10)#counts number of changed cont variables
        #continuous_count_arr = np.zeros(10)#counts number of cfs
        #get-MADfrom training data for continuous data
        #Need continuous: diversity; sparsity; proximity
        #sparsity will be near impossible
        cont_div_list = list()
        cont_prox_list = list()
        for i in range(len(explanation)):
            #cat_div, cont_div, count_div = (dataset,explanation,csv_div_filename,i)
            for delta_str in explanation[i][i+1]['delta']:
                print('delta str: ',delta_str)
                cont_prox = 0
                cat_prox = 0
                diversity = 0
                cont_count = len(dataset['continuous'])
                cat_count = len(dataset['non_continuous'])
                cont_dist = 0
                cat_dist = 0
                db_prox = 'null'
                sparsity = 0
                #print('before validity')
                validity,cost = get_validity(explanation[i][i+1],delta_str, dataset['class_name'],dataset['labels'])

                def set_cat_dist(d_dict,cat_dist,cat_count):
                    if d_dict['value'] != d_dict['record_value']:
                        cat_dist += 1
                    #cat_count +=1
                    #print('cat_dist: ', cat_dist , 'cat_count: ', cat_count)
                    return cat_dist, cat_count
                def get_sparsity(d_dict):
                    #print(d_dict['field'],d_dict['value'], d_dict['record_value'])
                    if d_dict['value'] != d_dict['record_value']:
                        return 1
                    else:
                        return 0

                for d_dict in explanation[i][i+1]['delta'][delta_str]:
                    if d_dict['field'] in dataset['human_data_dict']:
                        data_field = dataset['human_data_dict'][d_dict['field']]
                        if data_field in dataset['continuous']:
                            cont_dist += abs(float(d_dict['value']) - float(d_dict['record_value']))/dataset['mads'][data_field]
                            #cont_count += 1
                            #print('cont_dist: ', cont_dist , 'cont_count: ', cont_count,'mads: ', dataset['mads'][data_field])
                            sparsity += get_sparsity(d_dict)

                        elif data_field in dataset['non_continuous']:
                            """
                            if d_dict['value'] != d_dict['record_value']:
                                cat_dist += 1#importance constant not included would be += importance*1
                            print('cat_dist: ', cat_dist , 'cat_count: ', cat_count)
                            cat_count +=1
                            """
                            cat_dist, cat_count = set_cat_dist(d_dict,cat_dist,cat_count)
                            sparsity += get_sparsity(d_dict)
                            """
                            if d_dict['value'] != d_dict['record_value']:
                                continuous_arr[delta_count]+=1
                                print(continuous_arr, delta_count, continuous_count_arr)
                            """
                    else:
                        if d_dict['field'] == 'purpose of loan':
                            cat_dist, cat_count = set_cat_dist(d_dict,cat_dist,cat_count)
                            sparsity += get_sparsity(d_dict)
                        if d_dict['field'] == 'credit rating grade' : #horrible fix better done in human data dict in dataset
                            cat_dist, cat_count = set_cat_dist(d_dict,cat_dist,cat_count)
                            sparsity += get_sparsity(d_dict)

                #cat_count and cont_count have failed for DiCE record_object, which only records features that change in deltas not all features
                cat_count, cont_count, count_count = get_counts(explanation[i][i+1]['record'],dataset)
                #print(cat_count, cont_count, count_count)

                if cont_count > 0 :
                    #cont_prox = (-1/cont_count) * cont_dist #k is used when summing all rows in csv
                    cont_prox = (-1 / len(dataset['continuous'])) * cont_dist
                else:
                    cont_prox = (-1) * cont_dist #k is used when summing all rows in csv
                if cat_count > 0:
                    cat_prox = 1-(1/len(dataset['non_continuous'])* cat_dist )#k is used when summing all rows in csv
                    print ('cat_dist: ',cat_dist)
                else:
                    cat_prox = 1-(1 * cat_dist) #k is used when summing all rows in csv
                    breakpoint()#stop if this logic breaks
                if (cat_count + cont_count) > 0:
                    sparsity = 1 - (sparsity /( len(dataset['continuous'])  + len(dataset['non_continuous'])))
                else:
                    breakpoint('error in counting features for sparsity')
                    sparsity = 1- (sparsity) #k is used when summing all rows in csv
                print('cont_prox: ', cont_prox, ' cat_prox: ', cat_prox)
                print(sparsity, cont_count, cat_count)
                if cat_prox < 0 or cat_prox > 1:
                    print ('cat_prox has invalid value')
                    breakpoint()
                if cat_count != len(dataset['non_continuous']):
                    print ('cat_count has invalid value')
                    breakpoint()
                if cont_count != len(dataset['continuous']):
                    print ('cont_count has invalid value')
                    breakpoint()
                if sparsity < 0 or sparsity > 1:
                    print ('sparsity has invalid value')
                    breakpoint()
                writer.writerow([i,delta_str, validity, cost, cont_prox, cat_prox, sparsity])
        f.close()


def set_examples_csv(explanation, dataset,model_file_name,r_constant,num_tests_per_example,num_examples, examples_file_name):#output is csv file with file_name
    #r_constant = 1 #for dividing continuous feautres in line with Mothilal et al 2020 values {0.5,1.2}
    #num_tests_per_example = 10
    #num_examples = 100
    """
    func outputs a csv
    for every line input from explanation outputs a number of lines given by num_tests_per_example
    each output line is is random :
    cat variables are random values
    cont variables are a random value modified by the mads of the value and a radius r
    these are tested elsewhere by a 1NN used in set_output_csv
    """
    #blackbox = keras.models.load_model('keras_models/keras_model_lending_MinMaxScaler_27_06_22.h5')
    blackbox = keras.models.load_model(model_file_name)
    test_df = init_row(dataset)


    #example and num_tests are int i in their for loop operations
    for example in range(num_examples ):
        for num_tests in range(num_tests_per_example):
            test_row = init_row_X(dataset)
            for column in dataset['X_columns']:
                ran = np.random.rand()#ran is a random number between 0 and 1

                if column in dataset['continuous']:
                    mad = dataset['mads'][column]#median absolute deviation
                    r_value =''
                    for r_dict in explanation[example][example+1]['record']:

                        if r_dict['field'] == column:
                            r_value = r_dict['value']
                        if dataset['name']=='LoanStats3a':
                            human_col = dataset['data_human_dict'][column]
                            if r_dict['field'] == human_col:
                                r_value = float(r_dict['value'])
                        #else: #is categorical
                            #r_value = dataset['data_human_dict'][column]['value']
                    if r_value == '':
                        print('r_value not assigned')
                        breakpoint()

                    radius = ((2*ran)-1)*mad#r_constant replaced by mad (median absolute deviation)
                    scaled_r_value = dataset['scaler'][column].transform(np.array(r_value).reshape(1,-1))
                    scaled_radius = dataset['scaler'][column].transform(np.array(radius).reshape(1,-1))
                    #go from -r to r where r is mad
                    if (scaled_r_value + scaled_radius) < 0:
                        mod_value = [0]
                    elif (scaled_r_value + scaled_radius) > 1:
                        mod_value = [1]
                    else:
                        mod_value = scaled_r_value[0] + scaled_radius[0]
                    test_row[column] = mod_value
                else:#categorical
                    dummy = dataset['dummy'][column]
                    num_dummies = len(dummy)
                    dummy_num = math.trunc(ran * num_dummies)
                    value = dummy[dummy_num]
                    for dum in range(num_dummies):
                        if dummy[dum] == value:
                            test_row[dummy[dum]] = [1]
                        else:
                            test_row[dummy[dum]] = [0]
            #get y label (from MTurk marking code)

            #test_row[dataset['class_name']] = blackbox.predict(test_row.values)#is shape (1,32) needs to be (0,0,32) needs custom method from my predict class
            test_row[dataset['class_name']] = predict_single(blackbox, test_row.values)
            test_row['test_number'] = num_tests
            test_row['example'] = example
            test_df = pd.concat([test_df,test_row])
    test_df.to_csv(examples_file_name)

def set_output_csv(explanation, dataset,file_name_in,file_name_out,row_numbers):
    """
    this tests examples provided by set_examples
    """
    input_df = pd.DataFrame(columns=dataset['X_columns_with_dummies'])
    input_df[dataset['class_name']] = []
    output_df = pd.DataFrame(columns=['example','test','keras_predict','knn_predict','original_class','number_of_deltas'])#,'original_prediction'])
    for row_number in range(row_numbers):
        input_row = init_row(dataset)
        for entry in explanation [row_number][row_number+1]['record']:#don't forget class name
            if dataset['name']=='LoanStats3a':                                    #if column in dataset['continuous']:#redundant
                if entry['field'] == dataset['class_name']:#write as 0 or 1
                    for i in range(len(dataset['labels'])):
                        if dataset['labels'][i] == entry['value']:
                            input_row[dataset['class_name']] = [i]
                            class_int = i
                            original_class = i
                else:
                    if entry['field'] in dataset['human_data_dict'].keys() and dataset['human_data_dict'][entry['field']] in dataset['continuous']:

                        data_col = dataset['human_data_dict'][entry['field']]

                        input_row[data_col]=normalise_continuous(dataset,data_col,float(entry['value']))
                    #elif entry['field'] in ['cost', 'distance']:
                        #take no action
                    else:
                        for cat_feature in dataset['dummy']:
                            if entry['field'] == 'purpose of loan':
                                entry['field']= 'purpose'
                            if entry['field'] == 'credit rating grade':
                                entry['field']= 'grade'
                            if entry['field']=='term':
                                entry['field']='term of loan'
                            if entry['field']== 'home_ownership':
                                entry['field'] = 'home ownership'
                            if dataset['human_data_dict'][entry['field']] == cat_feature:
                                for dummy in dataset['dummy'][cat_feature]:
                                    if entry['field']== 'home ownership':

                                        if entry['value'] == 'rent':
                                            dummy_name = dataset['human_data_dict'][entry['field']] + '_RENT'
                                        elif entry['value'] == 'mortgaged':
                                            dummy_name = dataset['human_data_dict'][entry['field']] + '_MORTGAGE'
                                        elif entry['value'] == 'none':
                                            dummy_name = dataset['human_data_dict'][entry['field']] + '_NONE'
                                        elif entry['value'] == 'other':
                                            dummy_name = dataset['human_data_dict'][entry['field']] + '_OTHER'
                                        elif entry['value'] == 'own':
                                            dummy_name = dataset['human_data_dict'][entry['field']] + '_OWN'
                                        else:
                                            dummy_name = dataset['human_data_dict'][entry['field']]+'_'+entry['value']

                                    elif  entry['field']== 'term of loan':
                                        if entry['value'] == '36 months':
                                            dummy_name = dataset['human_data_dict'][entry['field']] + '_ 36 months'
                                        if entry['value'] == '60 months':
                                            dummy_name = dataset['human_data_dict'][entry['field']] + '_ 60 months'
                                    elif  entry['field']== 'purpose':
                                        if entry['value']=='major purchase':
                                            dummy_name =entry['field'] + '_major_purchase'
                                        elif entry['value']=='credit card':
                                            dummy_name =entry['field'] + '_credit_card'
                                        elif entry['value']=='debt consolidation':
                                            dummy_name =entry['field'] + '_debt_consolidation'
                                        elif entry['value']=='home improvement':
                                            dummy_name =entry['field'] + '_home_improvement'
                                        elif entry['value']=='renewable energy':
                                            dummy_name =entry['field'] + '_renewable_energy'
                                        elif entry['value']=='small business':
                                            dummy_name =entry['field'] + '_small_business'
                                        else:
                                            dummy_name =entry['field'] + '_'+ entry['value']
                                    else:
                                        dummy_name = entry['field'] + '_'+ entry['value']

                                    if dummy_name == dummy:
                                        input_row[dummy] = [1]

                                    else:
                                        input_row[dummy] =[0]



            else:#not lending

                if entry['field'] in dataset['continuous']:
                    input_row[entry['field']]=normalise_continuous(dataset,entry['field'],float(entry['value']))
                elif entry['field'] == dataset['class_name']:#write as 0 or 1
                    for i in range(len(dataset['labels'])):
                        if dataset['labels'][i] == entry['value']:
                            input_row[dataset['class_name']] = [i]
                            class_int = i
                            original_class = i
                else:#dummied cat features
                    for cat_feature in dataset['dummy']:
                        if entry['field'] == cat_feature:
                            for dummy in dataset['dummy'][cat_feature]:
                                dummy_name = entry['field'] + '_ '+ entry['value']
                                if dummy_name == dummy:
                                    input_row[dummy] = [1]
                                else:
                                    input_row[dummy] =[0]

        record_row = input_row
        input_df = pd.concat([input_df,input_row])
        input_df.reset_index()
        #add deltas
        number_of_deltas = len(explanation[row_number][row_number+1]['delta'])
        for delta_num in explanation[row_number][row_number+1]['delta']:#delta_num is a str
            #because for DiCE only some features in delta first read in record as df
            input_row = record_row

            for entry in explanation[row_number][row_number+1]['delta'][delta_num]:
                if dataset['name']=='LoanStats3a':
                    if entry['field'] == dataset['class_name']:#write as 0 or 1
                        for i in range(len(dataset['labels'])):
                            if dataset['labels'][i] == entry['value']:
                                input_row[dataset['class_name']] = [i]
                                class_int = i
                                original_class = i
                    else:
                        if entry['field'] in dataset['human_data_dict'].keys() and dataset['human_data_dict'][entry['field']] in dataset['continuous']:
                            data_col = dataset['human_data_dict'][entry['field']]
                            input_row[data_col]=normalise_continuous(dataset,data_col,float(entry['value']))
                        elif entry['field'] in ['cost', 'distance']:
                            nonsense_variable = 'nonsense' #an action to be no action only a side effect on a variable found no wehere else
                        else:
                            for cat_feature in dataset['dummy']:
                                if entry['field'] == 'purpose of loan':
                                    entry['field']= 'purpose'
                                if entry['field'] == 'credit rating grade':
                                    entry['field']= 'grade'
                                if entry['field'] =='home_ownership':
                                    entry['field'] = 'home ownership'
                                if entry['field'] == 'term':
                                    entry['field'] = 'term of loan'
                                if dataset['human_data_dict'][entry['field']] == cat_feature:
                                    for dummy in dataset['dummy'][cat_feature]:
                                        if entry['field']== 'home ownership':
                                            if entry['value'] == 'rent':
                                                dummy_name = dataset['human_data_dict'][entry['field']] + '_RENT'
                                            elif entry['value'] == 'mortgaged':
                                                dummy_name = dataset['human_data_dict'][entry['field']] + '_MORTGAGE'
                                            elif entry['value'] == 'none':
                                                dummy_name = dataset['human_data_dict'][entry['field']] + '_NONE'
                                            elif entry['value'] == 'other':
                                                dummy_name = dataset['human_data_dict'][entry['field']] + '_OTHER'
                                            elif entry['value'] == 'own':
                                                dummy_name = dataset['human_data_dict'][entry['field']] + '_OWN'
                                            else:
                                                dummy_name = dataset['human_data_dict'][entry['field']] + '_' + entry['value']

                                        elif  entry['field']== 'term of loan':
                                            if entry['value'] == '36 months':
                                                dummy_name = dataset['human_data_dict'][entry['field']] + '_ 36 months'
                                            if entry['value'] == '60 months':
                                                dummy_name = dataset['human_data_dict'][entry['field']] + '_ 60 months'
                                        elif  entry['field']== 'purpose':
                                            if entry['value']=='major purchase':
                                                dummy_name =entry['field'] + '_major_purchase'
                                            elif entry['value']=='credit card':
                                                dummy_name =entry['field'] + '_credit_card'
                                            elif entry['value']=='debt consolidation':
                                                dummy_name =entry['field'] + '_debt_consolidation'
                                            elif entry['value']=='home improvement':
                                                dummy_name =entry['field'] + '_home_improvement'
                                            elif entry['value']=='renewable energy':
                                                dummy_name =entry['field'] + '_renewable_energy'
                                            elif entry['value']=='small business':
                                                dummy_name =entry['field'] + '_small_business'
                                            else:
                                                dummy_name =entry['field'] + '_'+ entry['value']
                                        else:
                                            dummy_name = entry['field'] + '_'+ entry['value']
                                        if dummy_name == dummy:
                                            input_row[dummy] = [1]
                                        else:
                                            input_row[dummy] =[0]

                else:#not lending
                    if entry['field'] in dataset['continuous']:

                        input_row[entry['field']]=normalise_continuous(dataset,entry['field'],float(entry['value']))[0]
                    else:#dummied cat features
                        for cat_feature in dataset['dummy']:
                            if entry['field'] == cat_feature:
                                for dummy in dataset['dummy'][cat_feature]:
                                    dummy_name = entry['field'] + '_ '+ entry['value']
                                    if dummy_name == dummy:
                                        input_row[dummy] = [1]
                                    else:
                                        input_row[dummy] = [0]
            input_row[dataset['class_name']] = [get_class_int(class_int)]#is a cf so therefore other class to record
            input_df = pd.concat([input_df,input_row])

            #have df next create 1NN classifier for local points
        X = input_df[dataset['X_columns_with_dummies']]#.values
        y = input_df[dataset['class_name']]#.values

        knn_model = KNeighborsClassifier(n_neighbors=1)
        knn_model.fit(X,y)
        #get relevant rows from csv of generated local points
        local_df = pd.read_csv(file_name_in)
        example_df = local_df.loc[local_df['example']==row_number]
        keras_predict = example_df[dataset['class_name']].reset_index(drop=True)

        number_of_tests = len(example_df['test_number'])
        example_df = example_df[dataset['X_columns_with_dummies']]
        knn_predict = knn_model.predict(example_df.values)

        for row in range(number_of_tests):#['example','test','keras_predict','knn_predict']
            output_row = pd.DataFrame(columns=['example','test','keras_predict','knn_predict'])
            output_row['example'] = [row_number]
            output_row['test'] = [row]
            output_row['keras_predict'] = [keras_predict[row]]
            output_row['knn_predict'] = [knn_predict[row]]
            output_row['original_class']= original_class
            output_row['number_of_deltas']=number_of_deltas
            output_row['number_of_deltas']=number_of_deltas
            output_df = pd.concat([output_df,output_row])
    output_df.to_csv(file_name_out)

def knn_stats(output_csv):

    marking_df = pd.read_csv(output_csv)

    # need p,r,f1
    # 'We use precision, recall, and F1 for the counterfactual outcome class (Figure 3) as our main evaluation metrics because of the class imbalance in data points near the original input. To evaluate the sensitivity of these metrics to varying distance from the original input, we show these metrics for points sampled within varying distance thresholds'
    # precision TP/TP+FP
    # recall is TP/TP+FN
    # want cf class for  knn  this is the opposite of keras_pred
    # p=0
    # r=0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    number_wanted_deltas = 'NA'#8  # int or 'NA'
    for row in range(marking_df.values.shape[0]):
        if number_wanted_deltas == 'NA':
            if marking_df['original_class'][row] == 0:
                if marking_df['keras_predict'][row] < 0.5 and int(marking_df['knn_predict'][row]) < 0.5:
                    TN += 1
                elif marking_df['keras_predict'][row] < 0.5 and int(marking_df['knn_predict'][row]) >= 0.5:
                    FP += 1
                elif marking_df['keras_predict'][row] >= 0.5 and int(marking_df['knn_predict'][row]) < 0.5:
                    FN += 1
                elif marking_df['keras_predict'][row] >= 0.5 and int(marking_df['knn_predict'][row]) > 0.5:
                    TP += 1
                else:
                    print('logic error in p,r calcs')
                    breakpoint()
            else:  # original class == 1
                if marking_df['keras_predict'][row] < 0.5 and int(marking_df['knn_predict'][row]) < 0.5:
                    TP += 1
                elif marking_df['keras_predict'][row] < 0.5 and int(marking_df['knn_predict'][row]) >= 0.5:
                    FN += 1
                elif marking_df['keras_predict'][row] >= 0.5 and int(marking_df['knn_predict'][row]) < 0.5:
                    FP += 1
                elif marking_df['keras_predict'][row] >= 0.5 and int(marking_df['knn_predict'][row]) > 0.5:
                    TN += 1
                else:
                    print('logic error in p,r calcs')
                    breakpoint()

        else:
            if number_wanted_deltas == marking_df['number_of_deltas'][row]:
                if marking_df['original_class'][row] == 0:
                    if marking_df['keras_predict'][row] < 0.5 and int(marking_df['knn_predict'][row]) < 0.5:
                        TN += 1
                    elif marking_df['keras_predict'][row] < 0.5 and int(marking_df['knn_predict'][row]) >= 0.5:
                        FP += 1
                    elif marking_df['keras_predict'][row] >= 0.5 and int(marking_df['knn_predict'][row]) < 0.5:
                        FN += 1
                    elif marking_df['keras_predict'][row] >= 0.5 and int(marking_df['knn_predict'][row]) > 0.5:
                        TP += 1
                    else:
                        print('logic error in p,r calcs')
                        breakpoint()
                else:  # original class == 1
                    if marking_df['keras_predict'][row] < 0.5 and int(marking_df['knn_predict'][row]) < 0.5:
                        TP += 1
                    elif marking_df['keras_predict'][row] < 0.5 and int(marking_df['knn_predict'][row]) >= 0.5:
                        FN += 1
                    elif marking_df['keras_predict'][row] >= 0.5 and int(marking_df['knn_predict'][row]) < 0.5:
                        FP += 1
                    elif marking_df['keras_predict'][row] >= 0.5 and int(marking_df['knn_predict'][row]) > 0.5:
                        TN += 1
                    else:
                        print('logic error in p,r calcs')
                        breakpoint()

    if TP == 0 or FP == 0 or TN == 0 or FN == 0:
        print('one of the metrics == 0, potential div by 0 error')
        breakpoint()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = (2 * precision * recall) / (precision + recall)  # f1 for +ve predictions
    other_precision = TN / (TN + FN)
    other_recall = TN / (TN + FP)
    other_f1 = (2 * other_precision * other_recall) / (other_precision + other_recall)  # f1 for -ve redictions
    accuracy = (TP + TN) / (TP + TN + FP + FN)  # accuracy is best for measuring both classes
    print('TP: ', TP, ' TN: ', TN, ' FP: ', FP, ' FN: ', FN)
    print('p: ', precision, ' r: ', recall, ' f1: ', f1)
    print('-ve p: ', other_precision, ' -ve r: ', other_recall, '-ve f1: ', other_f1)
    print('acc: ', accuracy)
    breakpoint()



def init_row(dataset):
    input_row = init_row_X(dataset)
    input_row[dataset['class_name']] = []
    return input_row
def init_row_X(dataset):
    input_row = pd.DataFrame(columns = dataset['X_columns_with_dummies'])
    return input_row
def get_class_int(class_int):
    if class_int == 0:
        return 1
    else:
        return 0

def normalise_continuous(dataset, field, value):
    scaler = dataset['scaler'][field]
    return scaler.transform(np.array(value).reshape(1,-1))

def main(argv):#argv[1] = db_file (string) #argv[2] = pickled dataset (string) if entered as '' will default to lending dataset
    if argv[1] == '':#default db file
        db_file = ('explanation_store/demo_10.db')
    else:
        db_file = ('explanation_store/' + argv[1])
    conn = create_connection(db_file)
    exp_ids = get_exp_ids(conn) #an array of tuples
    explanation = list()
    #change exp_ids from array of tuples to list
    translate = True
    #get dataset from pickle
    if argv[2] =='':
        pickle_filename = 'pickled_data/lending_pickled_data_MinMax_27_06_22.p'
    else:
        pickle_filename = 'pickled_data/'+argv[2]
    infile = open(pickle_filename,'rb')
    dataset = pickle.load(infile)

    #get counterfactual for explanations
    for e in exp_ids:
        exp = create_explanation(conn, e,dataset,translate)
        e_up = {e[0]:exp}
        explanation.append(e_up)
    with open('JSON_out/explanation.json','w') as outfile:
       json.dump(explanation,outfile)
    with open('JSON_out/explanation.p','wb') as pf:
        pickle.dump(explanation,pf)

    if argv[3] == 'create_stats':
        csv_prox_filename = input('Enter Proximity csv file name (CR gives default name): ')#'lending_ia_prox.csv'
        if csv_prox_filename == '':#default for ease of testing
            csv_prox_filename = 'csv/proximity.csv'
        csv_div_filename = input('Enter Diversity csv file name (CR gives default name) : ')#'lending_ia_div.csv'
        if csv_div_filename == '':#default for ease of testing
            csv_div_filename = 'csv/diversity.csv'

        get_stats_write_stats(dataset, explanation, csv_prox_filename, csv_div_filename)
        breakpoint()
    elif argv[3] == 'set_examples_csv':
        print('set_examples_csv() called, to create the examples used in INN test')
        r_constant = 1
        num_tests_per_example = 10
        num_examples = 10
        model_file_name = input('Enter model file name (CR for lending model)')
        if model_file_name == '':
            model_file_name = 'keras_models/keras_model_lending_MinMaxScaler_27_06_22.h5'
        examples_file_name = input('Enter examples file name: (CR for default')
        if examples_file_name == '':
            examples_file_name = 'csv/examples.csv'
        set_examples_csv(explanation, dataset, model_file_name, r_constant, num_tests_per_example, num_examples, examples_file_name)

        breakpoint()


    elif argv[3] =='set_output_csv':
        print('set_output_csv() called')
        file_name_in = input('file name to be read (CR gives default)')
        if file_name_in == '':
            file_name_in = 'csv/examples.csv'
        file_name_out = input('file name for output (CR gives default)')
        if file_name_out == '':
            file_name_out = 'csv/outputs.csv'
        row_numbers = 10
        set_output_csv(explanation, dataset, file_name_in, file_name_out, row_numbers)

        breakpoint()

    elif argv[3] == 'knn_stats':
        print('knn test called')
        file_name_out = input('file name for outputs (CR gives default)')
        if file_name_out == '':
            file_name_out = 'csv/outputs.csv'
        knn_stats(file_name_out)
        breakpoint()
    else:
        print('no relevant commands given in arg3')

        breakpoint()
    breakpoint('outside of if else statement for commands')#used if only wanting stats












if __name__ == "__main__":#arg1 = db_name arg2 = pickled data name arg3 = function wanted
    main(sys.argv)
    #function names are for stats
