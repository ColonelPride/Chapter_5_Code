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
    with open(csv_div_filename,'w',encoding='UTF8') as div_f:
        writer = csv.writer(div_f)
        header = ['record_id', 'cat_diversity', 'cont_diversity', 'count_diversity']
        writer.writerow(header)
        for i in range(len(explanation)):
            cat_div, cont_div, count_div = get_write_diversity(dataset,explanation,csv_div_filename,i)
            writer.writerow([i, cat_div, cont_div, count_div])
        div_f.close()

    with open(csv_prox_filename,'w',encoding='UTF8') as f:
        writer = csv.writer(f)
        header = ['record_id', 'delta_id', 'valid', 'cost','cont_proximity', 'cat_proximity', 'sparsity']
        writer.writerow(header)
        print('prox writer')
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
                #cont_count = 0
                cont_count = len(dataset['continuous'])
                #cat_count = 0
                cat_count = len(dataset['non_continuous'])
                cont_dist = 0
                cat_dist = 0
                db_prox = 'null'
                sparsity = 0
                #print('before validity')
                validity,cost = get_validity(explanation[i][i+1],delta_str, dataset['class_name'],dataset['labels'])

                #print('after validity')
                #print('validity: ', validity, ' cost: ', cost)
                #for d_dict in explanation[i][i+1]['delta'][delta_str]:#change to loop through record

                #print('d_dict: ', d_dict)
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

                #cat_count and cont_count have failed for DiCE record_object, which only records features that chnage in deltas not all features
                cat_count, cont_count, count_count = get_counts(explanation[i][i+1]['record'],dataset)
                print(cat_count, cont_count, count_count)

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
                    breakpoint()
                if (cat_count + cont_count) > 0:
                    #sparsity = 1- (sparsity /(cat_count + cont_count)) #k is used when summing all rows in csv
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



def main(argv):#argv[1] = db_file (string) #argv[2] = pickled dataset (string) if entered as '' will default to lending dataset
    if argv[1] == '':
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
    if argv[3] == 'stats':
        csv_prox_filename = input('Enter proximity csv file name: ')
        csv_div_filename = input('Enter Diversity csv file name: ')
        get_stats_write_stats(dataset, explanation, csv_prox_filename, csv_div_filename)
        #get_stats_write_stats(dataset, explanation, 'search_by_cost/compas/IA/mads/compas_ia_prox.csv','search_by_cost/compas/IA/mads/compas_ia_div.csv')
        breakpoint()
    elif argv[3] == 'get_prox_csv':

    #breakpoint()#used if only wanting stats


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

    def set_examples_csv(file_name,r_constant,num_tests_per_example,num_examples):#output is csv file with file_name
        #r_constant = 1 #for dividing continuous feautres in line with Mothilal et al 2020 values {0.5,1.2}
        #num_tests_per_example = 10
        #num_examples = 100
        """
        func outputs a csv
        for every line input from explanation outputs a number of lines given by num_tests_per_example
        each output line is is random :
        cat variables are random values
        cont variables are a random value modified b the mads of the value and a radius r
        these are tested elsewhere by a 1NN used in set_output_csv
        """
        blackbox = keras.models.load_model('keras_models/keras_model_lending_MinMaxScaler_27_06_22.h5')
        test_df = init_row(dataset)
        for example in range(num_examples):
            for num_tests in range(num_tests_per_example):
                test_row = init_row_X(dataset)
                for column in dataset['X_columns']:
                    ran = np.random.rand()#ran is a random number between 0 and 1
                    """
                    if dataset['name']=='LoanStats3a':
                        if column in dataset['continuous']:
                            column = dataset['data_human_dict'][column]
                        else: #is categorical
                            column = dataset['data_human_dict'][column]['name']
                    """
                    #if r_dict['field'] == column:

                    if column in dataset['continuous']:
                        mad = dataset['mads'][column]#median absolute deviation
                        r_value =''
                        for r_dict in explanation[example][example+1]['record']:
                            print('r_dict: ',r_dict)
                            if r_dict['field'] == column:
                                r_value = r_dict['value']
                            if dataset['name']=='LoanStats3a':
                                #if column in dataset['continuous']:#redundant
                                human_col = dataset['data_human_dict'][column]
                                if r_dict['field'] == human_col:
                                    r_value = float(r_dict['value'])
                            #else: #is categorical
                                #r_value = dataset['data_human_dict'][column]['value']
                        if r_value == '':
                            print('r_value not assigned')
                            breakpoint()
                        #r_value = explanation [0][1]['record'][column]#get value from record
                        radius = ((2*ran)-1)*mad#(mad*2*ran)-mad
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

                test_row[dataset['class_name']] = blackbox.predict(test_row.values)
                test_row['test_number'] = num_tests
                test_row['example'] = example
                test_df = pd.concat([test_df,test_row])
        test_df.to_csv(file_name)



    def set_output_csv(file_name_in,file_name_out,row_numbers):
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
                            #print(entry)
                            data_col = dataset['human_data_dict'][entry['field']]
                            #print(data_col)
                            input_row[data_col]=normalise_continuous(dataset,data_col,float(entry['value']))
                        elif entry['field'] in ['cost', 'distance']:
                            print(entry['field'])
                            #take no action
                        else:
                            for cat_feature in dataset['dummy']:
                                if entry['field'] == 'purpose of loan':
                                    entry['field']= 'purpose'
                                if entry['field'] == 'credit rating grade':
                                    entry['field']= 'grade'
                                if entry['field']=='term':
                                    entry['field']='term of loan'
                                #bodge temp fix below is it a rogue instance of the underscore in home_ownership or a deeper problem
                                #print('record_A',dataset['human_data_dict'][entry['field']], cat_feature)
                                print('entry[field]: ',entry['field'])
                                if entry['field']== 'home_ownership':
                                    entry['field'] = 'home ownership'
                                if dataset['human_data_dict'][entry['field']] == cat_feature:
                                    for dummy in dataset['dummy'][cat_feature]:
                                        if entry['field']== 'home ownership':
                                            print('entry field: ',entry['field'],' entry value: ',entry['value'])
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
                                                print(dummy_name)
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
                                        #print('record_B',entry, cat_feature)
                                        if dummy_name == dummy:
                                            input_row[dummy] = [1]
                                            print(dummy_name,1)
                                        else:
                                            #if input_row[dummy][0] != 1:
                                            input_row[dummy] =[0]
                                            print(dummy_name,0)


                else:#not lending
                    #print('not lending dataset')
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
                #print( delta_num, explanation [row_number][row_number+1]['delta'][delta_num])
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
                                #print(entry)
                                data_col = dataset['human_data_dict'][entry['field']]
                                #print(data_col)
                                input_row[data_col]=normalise_continuous(dataset,data_col,float(entry['value']))
                            elif entry['field'] in ['cost', 'distance']:
                                print(entry['field'])
                                #take no action
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
                                    #print('record_A',dataset['human_data_dict'][entry['field']], cat_feature)
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
                                                    print(dummy_name)
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
                                            #print('record_B',entry, cat_feature)
                                            if dummy_name == dummy:
                                                input_row[dummy] = [1]
                                                print(dummy_name,1)
                                            else:
                                                #if input_row[dummy][0] != 1:
                                                input_row[dummy] =[0]
                                                print(dummy_name,0)
                    else:#not lending
                        if entry['field'] in dataset['continuous']:
                            #print(float(entry['value']))
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
            #original_prediction = example_df[dataset['class_name']]
            number_of_tests = len(example_df['test_number'])
            example_df = example_df[dataset['X_columns_with_dummies']]
            knn_predict = knn_model.predict(example_df.values)
            #print('example_df: ',example_df)
            #print('knn_predict: ',knn_predict)
            #print('keras_predict: ',keras_predict)
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
    #breakpoint()
    #start of knn code here (funcs above)
    #need a marking scheme on csv
    #set_examples_csv('test_examples_knn_lending_distance.csv',1,10,99)
    #set_output_csv('test_examples_knn_lending_distance.csv','test_outputs_knn_lending_distance.csv',99)
    marking_df = pd.read_csv('test_outputs_knn_lending_distance.csv')
    #need p,r,f1
    #'We use precision, recall, and F1 for the counterfactual outcome class (Figure 3) as our main evaluation metrics because of the class imbalance in data points near the original input. To evaluate the sensitivity of these metrics to varying distance from the original input, we show these metrics for points sampled within varying distance thresholds'
    #precision TP/TP+FP
    #recall is TP/TP+FN
    #want cf class for  knn  this is the opposite of keras_pred
    #p=0
    #r=0
    TP=0
    FP=0
    TN=0
    FN=0
    number_wanted_deltas = 8# int or 'NA'
    for row in range(marking_df.values.shape[0]):
        if number_wanted_deltas == 'NA':
            if marking_df['original_class'][row] == 0:
                if marking_df['keras_predict'][row] < 0.5 and int(marking_df['knn_predict'][row]) < 0.5 :
                    TN += 1
                elif marking_df['keras_predict'][row] < 0.5  and int(marking_df['knn_predict'][row]) >= 0.5 :
                    FP += 1
                elif marking_df['keras_predict'][row] >= 0.5  and int(marking_df['knn_predict'][row]) < 0.5 :
                    FN += 1
                elif marking_df['keras_predict'][row] >= 0.5  and int(marking_df['knn_predict'][row]) > 0.5 :
                    TP += 1
                else:
                    print('logic error in p,r calcs')
                    breakpoint()
            else:#original class == 1
                if marking_df['keras_predict'][row] < 0.5 and int(marking_df['knn_predict'][row]) < 0.5 :
                    TP += 1
                elif marking_df['keras_predict'][row] < 0.5  and int(marking_df['knn_predict'][row]) >= 0.5 :
                    FN += 1
                elif marking_df['keras_predict'][row] >= 0.5  and int(marking_df['knn_predict'][row]) < 0.5 :
                    FP += 1
                elif marking_df['keras_predict'][row] >= 0.5  and int(marking_df['knn_predict'][row]) > 0.5 :
                    TN += 1
                else:
                    print('logic error in p,r calcs')
                    breakpoint()

        else:
            if number_wanted_deltas == marking_df['number_of_deltas'][row]:
                if marking_df['original_class'][row] == 0:
                    if marking_df['keras_predict'][row] < 0.5 and int(marking_df['knn_predict'][row]) < 0.5 :
                        TN += 1
                    elif marking_df['keras_predict'][row] < 0.5  and int(marking_df['knn_predict'][row]) >= 0.5 :
                        FP += 1
                    elif marking_df['keras_predict'][row] >= 0.5  and int(marking_df['knn_predict'][row]) < 0.5 :
                        FN += 1
                    elif marking_df['keras_predict'][row] >= 0.5  and int(marking_df['knn_predict'][row]) > 0.5 :
                        TP += 1
                    else:
                        print('logic error in p,r calcs')
                        breakpoint()
                else:#original class == 1
                    if marking_df['keras_predict'][row] < 0.5 and int(marking_df['knn_predict'][row]) < 0.5 :
                        TP += 1
                    elif marking_df['keras_predict'][row] < 0.5  and int(marking_df['knn_predict'][row]) >= 0.5 :
                        FN += 1
                    elif marking_df['keras_predict'][row] >= 0.5  and int(marking_df['knn_predict'][row]) < 0.5 :
                        FP += 1
                    elif marking_df['keras_predict'][row] >= 0.5  and int(marking_df['knn_predict'][row]) > 0.5 :
                        TN += 1
                    else:
                        print('logic error in p,r calcs')
                        breakpoint()

    if TP == 0 or FP == 0 or TN == 0 or FN == 0:
        print('one of the metrics == 0, potential div by 0 error')
        breakpoint()
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = (2*precision*recall)/(precision+recall)#f1 for +ve predictions
    other_precision = TN/(TN+FN)
    other_recall = TN/(TN+FP)
    other_f1 = (2*other_precision*other_recall)/(other_precision+other_recall)#f1 for -ve redictions
    #alt_p = alt_TP/(alt_TP+alt_FP)
    #alt_r = alt_TP/(alt_TP+alt_FN)
    #alt_f1 = (2*alt_p*alt_r)/(alt_p+alt_r)
    accuracy = (TP+TN)/(TP+TN+FP+FN) #accuracy is best for measuring both classes
    print('TP: ',TP,' TN: ',TN,' FP: ',FP,' FN: ', FN)
    print('p: ',precision,' r: ',recall,' f1: ',f1)
    print('-ve p: ',other_precision,' -ve r: ',other_recall, '-ve f1: ',other_f1)
    print('acc: ',accuracy)
    breakpoint()
    #keras_predict =
    #next add simulated local data
    #loop 100 local points a lot of similar code is in the Immune Algorithms




    breakpoint()#stop here for 1NN work()
    pickle.dump(explanation,open('pickled_data/test_validity.p','wb'))
    csv_prox_filename = 'csv/test_validity_compas_1cf_prox.csv'
    csv_div_filename = 'csv/test_validity_compas_1cf_div.csv'

    get_stats_write_stats(dataset, explanation, csv_prox_filename, csv_div_filename)
    breakpoint()#this here for Data/Frame project


    exp_list = list()
    for e in exp_ids:
        e = e[0]
        exp_list.append(e)
    exp_ids = exp_list
    #dump whole expalanation object to json for Data/Frame to use

    def write_to_text_file(endpoint,api_key,exp,text_file_object):
        data_dict = {}
        data_dict.update({'id':'Primary'})
        data_dict.update({'type':'json'})
        data_dict.update({'jsonData':exp})
        l = list()
        l.append(data_dict)
        data = {}
        data.update({'data':l})
        data.update({'projectArguments':None})
        data.update({'options':None})
        #temp write output to get first json for Arria Studio delta later.
        #text_file_object.write(json.dumps(exp))
        bearer = 'Bearer '+ api_key
        authorization  = {'Authorization':  bearer}
        content = {'Content-Type': 'application/json;charset=UTF-8',}
        headers = {}
        headers.update(content)
        headers.update(authorization)
        req = requests.post(endpoint, headers= headers,json=data)
        textJSON = req.text
        response_object = json.loads(textJSON)
        text = response_object[0]['result']
        print(text)
        text_file_object.write('\n**********************************************\n')
        text_file_object.write(str(e))
        text_file_object.write('\n**********************************************\n')
        #text_file_object.write('JSON')
        #text_file_object.write(json.dumps(exp))
        #text_file_object.write('**********************************************')
        text_file_object.write('TEXT\n')
        text_file_object.write(text)
        return text
        #text_file_object.close()
    #code to get explanation for row number
    row_numbers = [3]
    for e in exp_list:
        file_name = 'json_explanation.json'
        file_name = 'JSON_out/'+file_name
        text_file = 'text_out/text_and_JSON_out.txt'
        text_file_object = open(text_file,'a')
        exp = explanation[e-1][e]
        if exp['record_number'] in row_numbers:
            #add delta alterations here for all objects
            deltas_dict = exp['delta']
            delta_id_list = list()
            for delta_id in deltas_dict:
                delta_id_list.append(delta_id)
            #change dummy variables here
            #preprocess deltas to remove duplicate rows and change values

            #this is not relevant if no delta
            if len(deltas_dict) > 0:
                for i in delta_id_list:
                    first_explanation = deltas_dict[delta_id_list[0]]
                    filtered_deltas = list()#list of dicts remove rows where record and delta have same value
                    processed_deltas = list() #dict of dicts,  changed from list of dict because Studio is being .... about this being a list
                    variable_list = list() #list of used variable names for dummy values
                    for row in first_explanation:
                        variable_name = row['field']
                        record_value = row['record_value']
                        delta_value = row['value']
                        if record_value != delta_value:#only interesting and cf if record_value and delta_value are different
                            filtered_deltas.append(row)
                    #insert importances of deltas here, used to sort order of presented parts of deltas
                    #d_imp will equal:
                    #   for continuous:  |record_value - value| transformed by scaler[column]
                    #   for non continuous: unknown fill in later
                    #   for non_continuous and dummy: using filtered_deltas
                    """
                    unfinished or started functions for sorting inputs by importance here
                    for row in filtered_deltas:
                        if row in dataset['scaler']:#if in scaler is continuous
                            imp_val = np.abs(record_value - value)
                    """
                    for row in filtered_deltas:
                        variable_name = row['field']
                        record_value = row['record_value']
                        delta_value = row['value']
                        delta_len = len(processed_deltas)
                        if isinstance(variable_name,str):
                            #print('isinstance(variable_name,str')
                            processed_deltas.append({'field':variable_name,'value':delta_value,'record_value':record_value})
                            #processed_deltas[delta_len] = {'field':variable_name,'value':delta_value,'record_value':record_value}
                            #print('string')
                        else:#is dict {'name', 'value'}
                            #print('not string')
                            #print('NOT isinstance(variable_name,str')
                            if variable_name['name'] not in variable_list:
                                dict_item = ({'field':variable_name['name'],'value':'null','record_value':'null'})
                                if record_value == '1':
                                    dict_item['record_value'] = variable_name['value'] #set record
                                else:
                                    dict_item['value'] = variable_name['value']#set delta
                                processed_deltas.append(dict_item)
                                #processed_deltas[delta_len] = dict_item
                            else:
                                #alter item in list of dict
                                for i in range(len(processed_deltas)):
                                #for item in processed_deltas:
                                    item = processed_deltas[i]
                                    if item['field'] == variable_name['name']:
                                        if record_value =='1':
                                            item['record_value'] = variable_name['value'] #set record
                                        else:
                                            item['value'] = variable_name['value']#set delta
                            variable_list.append(variable_name['name'])

                    exp['delta'][i]= processed_deltas

            #JSON output
            title = str(exp['record_number'])+' : '+exp['description']
            text_file_object.write('\n**********************************************\n')
            text_file_object.write(title)
            text_file_object.write('\n**********************************************\n')
            text_file_object.write('JSON')
            text_file_object.write(json.dumps(exp))
            text_file_object.write('\n**********************************************\n')
            with open(file_name,'w') as outfile:
                json.dump(exp,outfile)



            #lime output
            #unified_explanation name
            #paper one endpoint ='https://app.studio.arria.com:443/alite_content_generation_webapp/text/DXqjA13KnJ'
            #paper one api_key = 'eyJhbGciOiJIUzUxMiJ9.eyJqdGkiOiJadzdGYzJUQzduNUFkaG1BeFJmak5DR1AiLCJpYXQiOjE1NzUzMDA1ODksImV4cCI6MTczMjk4MDU4OSwiaXNzIjoiQUxpdGUiLCJzdWIiOiJqNG93M3F1dXA5bXQiLCJBTGl0ZS5wZXJtIjpbInByczp4OkRYcWpBMTNLbkoiXSwiQUxpdGUudHQiOiJ1X2EifQ.0ulI4B7ttqHqfWf2oUHOxmnVpacJqihvy4mdC-jzHfDh9G1-BpznaqytkC80rQGpGWQbFF7moqKRKsImDr_RtQ'
            endpoint = 'https://app.studio.arria.com:443/alite_content_generation_webapp/text/kqoxbEdpxd3'
            api_key = 'eyJhbGciOiJIUzUxMiJ9.eyJqdGkiOiJGOE1abUtmaVlkVnhoYkVYRzUzU3hSeWkiLCJpYXQiOjE1ODI4MTI2MzAsImV4cCI6MTc0MDQ5MjYzMCwiaXNzIjoiQUxpdGUiLCJzdWIiOiJXa29JdUxYTUNTdDIiLCJBTGl0ZS5wZXJtIjpbInByczp4Omtxb3hiRWRweGQzIl0sIkFMaXRlLnR0IjoidV9hIn0.GEG54eaQ9ewcpLUEVNbCxbVWZRyOQ5SeBkZPZ4o7q3WkansKcnG0rq1kGgPeEwaAOtRu3ljEuRK4BItMc_mstA'
            """
            if exp['description'] == 'lime explanation':
                #old endpoint and api key
                #endpoint = 'https://app.studio.arria.com:443/alite_content_generation_webapp/text/2vzB7pvzmYP'
                #api_key = 'eyJhbGciOiJIUzUxMiJ9.eyJqdGkiOiJ1OGFvY1lTWlVrNi0xM3MxdFprV1pMVHAiLCJpYXQiOjE1NzIyMDU3NjYsImV4cCI6MTcyOTg4NTc2NiwiaXNzIjoiQUxpdGUiLCJzdWIiOiI4dS1KRzJvOWROMzgiLCJBTGl0ZS5wZXJtIjpbInByczp4OjJ2ekI3cHZ6bVlQIl0sIkFMaXRlLnR0IjoidV9hIn0.uUfb-xWhu83of62CaTmc7TCu2kkKWvq9H-G0LbBxpWsmHNdAYgZ5fKCJmSUebvcV5UFV6VR61STRnqcBkEvV1Q'

                studio_object = write_to_text_file(endpoint,api_key,exp,text_file_object)
                html_file_name= 'text_out/lime_html.htm'
                html_text = present_explanation.write_lime_to_html_file(html_file_name,exp,studio_object,dataset)
            """
            #dce output
            #elif exp['description'] == 'diverse_coherent_explanation':
            if exp['description'] == 'diverse_coherent_explanation' or exp['description'] == 'DiCE':
                #old endpoint and api key
                #endpoint = 'https://app.studio.arria.com:443/alite_content_generation_webapp/text/XgEV8BwDVdk'
                #api_key = 'eyJhbGciOiJIUzUxMiJ9.eyJqdGkiOiJFM1kzX3NiNGdHZUt3RnJhYmxLckEtWnEiLCJpYXQiOjE1NzIzNjMyOTYsImV4cCI6MTczMDA0MzI5NiwiaXNzIjoiQUxpdGUiLCJzdWIiOiJPaUxnVlhpOElzbi0iLCJBTGl0ZS5wZXJtIjpbInByczp4OlhnRVY4QndEVmRrIl0sIkFMaXRlLnR0IjoidV9hIn0.qhC2BfUBP2XX7d_MBtbq35NW9eIk6YKeupOnjDBK3a6AnBBadaUoOuw1ZW1-FqTbcn24C4JpjkbuzTjRu5_iFg'
                endpoint = 'https://app.studio.arria.com:443/alite_content_generation_webapp/text/bKzB7xBXZpz'#for DiCE_super
                api_key = 'eyJhbGciOiJIUzUxMiJ9.eyJqdGkiOiJXeU01N0ZLQXZnV3duSkZoTDdMRV9vWGwiLCJpYXQiOjE1ODM0MDI2OTQsImV4cCI6MTc0MTA4MjY5NCwiaXNzIjoiQUxpdGUiLCJzdWIiOiJxWi1HZ2s2WWU5bm8iLCJBTGl0ZS5wZXJtIjpbInByczp4OmJLekI3eEJYWnB6Il0sIkFMaXRlLnR0IjoidV9hIn0.JZYvfz5jhmEykoyXun7Q28JTF2m_z7uLJACrZVJRQD1H4gs4WM3SA1_Tm2SZQlRfutaJm6FD9zIrx0kezpRZ5g'#for DiCE super
                studio_object = write_to_text_file(endpoint,api_key,exp,text_file_object)
                html_file_name= 'text_out/dce_html.htm'
                html_text = present_explanation.write_dce_to_html_file(html_file_name,exp,studio_object,dataset)

            #opt-AInet
            """
            elif exp['description'] == 'optAINet':
                #old endpoint and api key
                #endpoint = 'https://app.studio.arria.com:443/alite_content_generation_webapp/text/kMJw7bBZgYQ'
                #api_key = 'eyJhbGciOiJIUzUxMiJ9.eyJqdGkiOiJZcUlEVEV2VlhiZzNHN0dhelVRZkNGR0YiLCJpYXQiOjE1NzI1MTgxNTEsImV4cCI6MTczMDE5ODE1MSwiaXNzIjoiQUxpdGUiLCJzdWIiOiJwZW42TndQMk5nTmkiLCJBTGl0ZS5wZXJtIjpbInByczp4OmtNSnc3YkJaZ1lRIl0sIkFMaXRlLnR0IjoidV9hIn0.0R5ngR0-zaKKnH6i7682isjL7IQm9dGsPhZQK6lDE2MjyOn3URuulfANjFlC8ZuktaQXY4LBBRuG9iZD2zMqJg'
                studio_object = write_to_text_file(endpoint,api_key,exp,text_file_object)
                html_file_name= 'text_out/opt_AINet_html.htm'
                html_text = present_explanation.write_ais_to_html_file(html_file_name,exp,studio_object,dataset)
            """
        text_file_object.close()
    import pdb; pdb.set_trace()


if __name__ == "__main__":#arg1 = db_name arg2 = pickled data name arg3 = function wanted
    main(sys.argv)
    #function names are for stats
