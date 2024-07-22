# translation and adaption of Brownlee Ruby code
# http://www.cleveralgorithms.com/nature-inspired/immune/immune_network_algorithm.html
import keras.backend
# import logging
# logging.getLogger('tensorflow').setLevel(logging.WARNING)
# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
# logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))

import pandas as pd
import numpy as np
import pdb
import copy
from operator import itemgetter
import sklearn
import matplotlib.pyplot as plt
# from astropy.stats import median_absolute_deviation
from scipy import stats
import math
# from write_to_db import *#removed for colab notebook circular dependancy
# from create_db import *
from sklearn.linear_model import LinearRegression
import csv
import gc
import predict

sort_by = 'distance'
#sort_by = 'cost'
use_mads = True  # using mads in distance calc
problem_size = 1 #1 rather than num_features because len(sorted_progeny) called in code
search_space = [0, 1]
# algorithm configuration
max_gens = 5  # for test_cf #make global
pop_size = 20  # make global
num_clones = 10  # make global

# beta = 100
beta = 1  # change to beta tested 2023.08.21 #make global
num_rand = 2
# affinity_constant = 0.35 #currently good results for adult, lending 0.35, compas 0.5/0.6
affinity_constant = 0.35  # change when not using mads
stop_condition = 0.01  # when changing distance from mads reduces dist, lower stop condition
# stop_condition  = 0.1 #2023.08.21
new_cell_rate = 0.4  # upto 0.4


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


def get_line_columns(dataset):
    # used to get columns for use with onehot encoded dummies, idf dummies used else without dummied columns in list.
    # params dataset dict object
    # returns list of columns titles
    line_columns = dataset['X_columns']
    if dataset['use_dummies'] == True:
        line_columns = dataset['X_columns_with_dummies']
    return line_columns


def keras_pred(model, line):
    # return the value the model predicts for the inputs
    # param the keras model, the input values
    # return array for [prediction for 0, perdiction for 1]
    # print('calling keras_pred')
    # print('line: ',line)
    #pred = float(model.predict(line, verbose=0)[0])#old line for keras 2
    #keras 3 new line
    pred = float(predict.predict_single(model,line))
    pred_array = [1 - pred, pred]
    # print('pred_array:',pred_array)
    return pred_array


def cost_constraint(cost):
    # return False if cost greater than value 0 for binary classifier else True
    # returning False exits a while loop in calling method
    if cost > 0:
        return False
    else:
        return True


def select_random_value(col, dataset):
    # select a random value for a a piece of data in a given column
    # 3 cases depending if a coulm is label_encoded, binned, or one-hot encoded(dummied), or scaled
    # params(column name, dataset dict object)
    # returns a randomly selected value
    ran = np.random.rand()  # ran is a random number between 0 and 1

    if col in dataset['label_encoder']:
        num_classes = len(dataset['label_encoder'][col].classes_)
        value = math.trunc(ran * num_classes)
    elif col in dataset['binner']:  # if not in label encoder then binned
        number_of_bins = dataset['number_of_bins']  # number of bins how to automate this number?
        value = trunc(ran * number_of_bins)
    elif col in dataset['dummy'].keys():
        dummy = dataset['dummy'][col]
        num_dummies = len(dummy)
        dummy_num = math.trunc(ran * num_dummies)
        value = dummy[dummy_num]
    elif col in dataset['scaler']:  # is continuous variable
        value = round(ran, 1)  # not called for continuous varibles
    else:
        value = round(ran, 1)  # not called for continuous varibles
    return value


def new_cell(dataset, model, prediction, target_vals):
    # create a new cell
    # params dataset dict object, ml model, predicted result (int), values of input instance
    # return a new cell object (dict)

    # add new constraint a count that when it reaches a limit just returns a null cell
    count = 0
    # limit=100#lower limit to 10 for mutate by dist
    limit = 10
    flag = True
    while flag:  # while combined distance == 0 flag is true , flag to false  i.e. must be different to reference cell
        count = count + 1
        # print('in new cell loop count = : ',count)
        line = []
        X_columns = dataset['X_columns']
        invariants = dataset['invariants']
        continuous = list()
        non_continuous = list()
        for col in X_columns:
            if col in dataset['continuous']:
                continuous.append(col)
            else:
                non_continuous.append(col)
        columns = list()
        # is the following block necessary?yes first 4 columns are continuous and must be fed to NN first and in order
        for c in continuous:
            columns.append(c)
        for nc in non_continuous:
            columns.append(nc)
        dummy = dataset['dummy']
        index = 0  # index for dummy columns
        for i in range(len(columns)):
            if columns[i] in continuous:
                if columns[i] in invariants:
                    line = np.append(line, target_vals[index])
                    index = index + 1
                else:  # non invariant e.g. variable
                    value = np.random.rand()
                    # value = round(value,1) no pint rounding here as effect of mutation is to move by small amounts
                    line = np.append(line, value)
                    index = index + 1
            elif columns[i] in non_continuous:
                if columns[i] in dummy.keys():
                    if columns[i] in invariants:
                        for dum in dummy[columns[i]]:
                            line = np.append(line, target_vals[index])
                            index = index + 1
                    else:  # non invariants
                        selected_value = select_random_value(columns[i], dataset)
                        for dum in dummy[columns[i]]:
                            if dum == selected_value:
                                line = np.append(line, 1)
                                index = index + 1
                            else:
                                line = np.append(line, 0)
                                index = index + 1
                else:  # non dummies in this branch
                    if columns[i] in invariants:
                        line = np.append(line, target_vals[index])
                        index = index + 1
                    else:  # non invariants
                        value = select_random_value(columns[i], dataset)
                        line = np.append(line, value)
                        index = index + 1

        line = line.reshape(1, -1)

        cost = objective_function(line, model, prediction)
        flag = cost_constraint(cost)
        # must rule out cells with a distance of 0
        if flag == False:
            if combined_distance(target_vals, line[0], dataset) == 0:
                flag = True  # do not allow cells with zero distance as these must have the same values as the  record
                breakpoint()
        if count > limit:
            # print('count > limit for new_cell')
            cell = None
            return cell
    dist_to_target = combined_distance(target_vals, line[0], dataset)
    cell = {'value': line,
            'norm_cost': 0,
            'cost': cost,
            # 'norm_distance':0,
            'distance': dist_to_target}
    # print('new_cell')
    return cell


def create_mutated_cell(value, norm_cost, model, prediction, dataset, target_vals):  # value is an array
    # take in a cells value and normalised cost and then create a mutation of that cell until it has a lower normalised cost
    # params (value array, norm_cost float, prediction int, dayset dict object, target vals array)
    # return a cell with a lower normalised cost
    # print('in create_mutated_cell value =: ', value)
    cost = objective_function(value, model, prediction)
    # print('in create_mutated_cell cost =: ', cost)
    # if cost < 0.5: cost = 1
    dist_to_target = combined_distance(target_vals, value[0], dataset)  # 'fault here some distances of clones == 0'
    # print('in create_mutated_cell distance to target =: ', cost)
    cell = {'value': value,
            'norm_cost': norm_cost,
            'cost': cost,
            'distance': dist_to_target}
    # print('exiting create_mutated_cell')
    return cell


def objective_function(line, model, prediction):
    # objective function returns a prediction of a line of values from a model
    # params line array, model ml model from keras, prediction int
    # returns  a value beteeen -0.5 to 0.5
    # if pred is 0 to get cf line[0][0] should be < 0.5
    # if pred is 1 to get cf line[0][1] should be < 0.5
    # for scikit
    # result =  ((1 - (model.predict_proba(line)[0][prediction]))-0.5)*2
    # for keras
    # print('objective function to called')
    result = 0.5 - keras_pred(model, line)[prediction]
    # print('objective function result: ',result)
    keras.backend.clear_session()  # clearing up keras tf models to prevent mem leak
    return result


def random_vector(minmax):  # minmax is a vector of two element array
    rand = (2 * np.random.rand()) - 1
    value = ((minmax[0] + (minmax[1] - minmax[0])) * rand)
    return value


# replaced  by np.random.normal()
# reinserted because np.random.normal(loc=0.0,scale=1.0,size= 1) does not work
def random_gaussian(mean=0.0, stdev=1.0):
    return np.random.normal()


"""
def random_gaussian(mean=0.0, stdev=1.0):
    u1 = 0
    u2 = 0
    w = 0
    flag = True
    while flag:
        u1 = (2 * np.random.rand()) - 1
        u2 = (2 * np.random.rand()) - 1
        w = (u1 * u1) + (u2 * u2)
        if w < 1:
          flag = False
        w = np.sqrt((-2.0 * np.log(w)) / w)
    out = mean + ((u2 * w) * stdev)
    #print('random gaussian out: ',out)
    return out
"""


def clone(parent):  # parent is an array
    clone = parent
    return clone


def mutation_rate(beta, normalized_cost):
    return (1.0 / beta) * np.exp(-normalized_cost)


def mutate(beta, child, normalized_cost, dataset, model, prediction):  # child is a 2d array [i][v]
    count = 0
    limit = 10  # if count = limit terminate process
    alpha = mutation_rate(beta, normalized_cost)
    # columns = dataset['X_columns']
    columns = list()
    for c in dataset['continuous']:
        columns.append(c)
    for nc in dataset['non_continuous']:
        columns.append(nc)
    flag = True
    line = copy.deepcopy(child['value'])
    dummy = dataset['dummy']
    # to accommadate dummies this will be easier to do in a dataframe as column names will be easier to manipulate than column numbers
    """
    line_columns = dataset['X_columns']
    if dataset['use_dummies'] == True:
        line_columns = dataset['X_columns_with_dummies']
    """
    line_columns = get_line_columns(dataset)
    df_line = pd.DataFrame(data=line, columns=line_columns)
    while flag:
        # potential for counter to stop long loops if more than given number return original cell
        for i in range(len(columns)):  # -1 because class name not in line
            if columns[i] in dataset['continuous']:
                if columns[i] not in dataset['invariants']:
                    """
                    #line[0][i-columns_offset] = line[0][i-columns_offset]
                    df_line[columns[i]] = df_line[columns[i]]# no change
                    else:#tab indented to avoid error at compile
                    line[0][i-columns_offset] = line[0][i-columns_offset]+(alpha*random_gaussian(mean=0.0, stdev=1.0))#-1 because class name not in line
                    if line[0][i-columns_offset] < 0:
                        line[0][i-columns_offset] = 0
                    if line[0][i-columns_offset] >1:
                        line[0][i-columns_offset] = 1
                    """
                    # print('get continuous mutation')
                    # add a constraint for mutate by distance to ensure value of feature is between 0 & 1
                    feature_value = df_line[columns[i]] + (alpha * random_gaussian(mean=0.0, stdev=1.0))
                    if feature_value[0] < 0:
                        # print('feature_value[0] constrained to be 0, was: ',feature_value[0])
                        feature_value[0] = 0
                    elif feature_value[0] > 1:
                        # print('feature_value[0] constrained to be 1, was: ', feature_value[0])
                        feature_value[0] = 1
                    df_line[columns[i]] = feature_value

            if columns[i] in dataset['non_continuous']:
                if columns[i] not in dataset['invariants']:
                    """
                    line[0][i-columns_offset] = line[0][i-columns_offset]
                    else:#tab indented to avoid error at compile
                    """
                    r_line = np.random.rand()  # picks a value to change to may be the original
                    r_alpha = np.random.rand()  # chance of mutation happening
                    # num_classes = len(dataset['label_encoder'][columns[i]].classes_)
                    # mutate to random value if r < alpha
                    # below is weird alpha is prop to 1/beta why multiply beta out?
                    # print('get non-continuous mutation')
                    if r_alpha < 10 * beta * alpha:  # 10*alpha to makea mutation more likely
                        # print('alpha: ', alpha,' beta: ',beta)
                        # print('select random value')
                        value = select_random_value(columns[i], dataset)
                        # print('value: ',value)
                        # line[0][i-1] = round((r_line * num_classes)-0.5,0)#-1 because class name not in line
                        if columns[i] in dummy.keys():
                            # print('columns[i] in dummy.keys()')
                            for dum in dummy[columns[i]]:
                                if dum == value:
                                    df_line[dum] = 1
                                else:

                                    df_line[dum] = 0
                        else:
                            # print('columns[i] not in dummy.keys()')
                            # line[0][i-columns_offset] = value
                            df_line[columns[i]] = value

        # flag = cost_constraint(objective_function(line,model,prediction))

        # print('flag = cost_constraint')
        # print('df_line.values: ',df_line.values,' prediction: ',prediction)
        # print('df_line.value example =  [[0.27914655 0.67903022 0.1515629  0.55941183 0. 1.  0.  0.  0.  0.  0.  0. 1.  0. 0. 0. 0.   0.   0.  0.  0.   0.  0.  0. 0.  0. 0.  1.  0.  0.  0.   1. ]] ')
        # print('len(df_line.valuess[0]): ',len(df_line.values[0]), ' should be len 32')
        flag = cost_constraint(objective_function(df_line.values, model, prediction))

        if flag == False:  # check distance if 0 not allowed why is cost constraint allowing this?
            # print('flag==False')
            if combined_distance(child['value'][0], df_line.values, dataset) == 0:
                # print('combined_distance(child[value][0], df_line.values, dataset) == 0 ')
                flag = True
                breakpoint()
        # else:#flag=True
        # print('flag==True')

        count = count + 1
        if count > limit:
            # print('count > limit, count:',count,' limit:',limit)
            # print('end of mutation of cell due to exceeding limit')
            # return child['value'] # change to return #trminate process as taking too long
            # line = df_line.values problem with returning invalid cell omewhere try here by return parent value rather than line
            # return line
            return child['value']
    # child['value']= line
    line = df_line.values
    # print('end of mutation of cell')
    # print('gc_count before: ', gc.get_count())
    gc.collect()
    # print('gc_count after: ', gc.get_count())
    return line  # child['value']


def sort_by_dist(list_of_dict):
    """
    for dict_item in list_of_dict:
        print('cell distance:',dict_item['distance'])
        if dict_item['distance']==None:
            print('distance is None')
            breakpoint()
    """
    sorted_list = sorted(list_of_dict, key=itemgetter('distance', 'cost'))
    # sort by cost added to differentiate between categorical rich data which tends to have similar distances
    return sorted_list


def sort_by_cost(list_of_dict):
    sorted_list = sorted(list_of_dict, key=itemgetter('cost'))
    return sorted_list


def clone_cell(beta, num_clones, parent, search_space, dataset, model, prediction, target_vals):
    clones = list()
    clones.append(parent)
    if sort_by == 'cost':
        normalized_cost = parent['norm_cost']
    elif sort_by == 'distance':
        normalized_cost = parent['norm_cost']
    else:
        print('fail on clone cell to pick sorting by cost/distance')
        breakpoint()
    for i in range(num_clones):
        # print('mutate cell')
        v = mutate(beta, parent, normalized_cost, dataset, model, prediction)
        # print('in clone_cell have mutated cell')
        child = create_mutated_cell(v, normalized_cost, model, prediction, dataset, target_vals)
        # print('in clone_cell created mutated cell')
        if child['distance'] == 0:
            print('child is at fact')
            breakpoint()
        clones.append(child)
    # clones = sorted(clones,key = itemgetter('distance'))
    # change here to return clones by cost instead of distance 2020/09/16
    # sort_method = 'cost'#'distance'#potential change here
    # sort_method =  'distance'#changing to sort by distance to see if prox improves
    # clones = sort_by_dist(clones)
    if sort_by == 'cost':
        # print('sorting clones by cost')
        clones = sort_by_cost(clones)
    elif sort_by == 'distance':
        # print('sorting clones by distance')
        clones = sort_by_dist(clones)
    if len(clones) > 0:
        # print('end of clone cell returning clone[0]')
        # print('gc_count before: ', gc.get_count())
        gc.collect()
        # print('gc_count after: ', gc.get_count())
        return clones[0]
    else:
        # print('end of clone cell returning parent')
        # print('gc_count before: ', gc.get_count())
        gc.collect()
        # print('gc_count after: ', gc.get_count())
        return parent


def calculate_normalized_cost(pop):
    # change to normalized_cost for german currently all cost between 0.5 and 1
    # therefore subtract 1/2 then * 2
    last = len(pop) - 1
    pop = sorted(pop, key=itemgetter('cost'))
    rg = pop[last]['cost']  # - pop[0]['cost'] #r changed from range to avoid confusion in loops
    if rg == 0:
        for cell in pop:
            cell['norm_cost'] = cell['cost']
    else:
        for cell in pop:
            cell['norm_cost'] = 1 - (cell['cost'] / rg)
    return pop


def calculate_normalized_distance(pop):
    for p in pop:
        if p == None:
            pop.remove(p)
    last = len(pop) - 1
    pop = sorted(pop, key=itemgetter('distance'))
    rg = pop[last]['distance']  # - pop[0]['cost'] #r changed from range to avoid confusion in loops
    if rg == 0:
        for cell in pop:
            cell['norm_cost'] = cell['distance']
    else:
        for cell in pop:
            cell['norm_cost'] = 1 - (cell['distance'] / rg)
    return pop


def average_cost(pop):
    sum = 0.0
    for cell in pop:
        sum = sum + cell['cost']
    return sum / len(pop)


def average_distance(pop):
    sum = 0.0
    for cell in pop:
        sum = sum + cell['distance']  # changed to distance as metric for mutation
    return sum / len(pop)


"""
#obsolete
def distance(c1, c2):
  sum = np.square(c1['value']-c2['value'])
  return np.sqrt(sum)
"""


def get_offset(class_name, columns):  # gets an offset depending if the class name is in class name or not
    if class_name in columns:  # assumes class_name is first in columns
        offset = 1
    else:
        offset = 0
    return offset


def combined_distance(reference_cell, cell, dataset):
    # distance = 0
    # count = 0
    columns = get_line_columns(dataset)  # dataset['columns']
    d_array = []
    d_cont_arr = []
    d_cat_arr = []
    for i in range(len(cell)):
        if columns[i] in dataset['continuous']:  # abs distance
            ref_val = reference_cell[i]
            cell_val = cell[i]
            dist = abs(ref_val - cell_val)
            # count = count +1
            # distance = distance + dist
            # d_array = np.append(d_array,dist)
            # changing to mad for dist
            d_cont_arr = np.append(d_cont_arr, dist)
            # print('target: ', ref_val ,'cell: ', cell_val,'dist: ',d,'count:',count)
        else:  # columns[i] in dataset['non_continuous']:#hamming distance
            ref_val = reference_cell[i]
            cell_val = cell[i]
            # if using dummies a change of 1 category results in two changes a change of 1 to 0 and a change of 0 to 1 this means that we need to change the cost of a change from 1 to 0.5
            if dataset['use_dummies']:
                if ref_val == cell_val:
                    dist = 0
                else:
                    dist = 0.5
            else:
                if ref_val == cell_val:
                    dist = 0
                else:
                    dist = 1
            # distance = distance + dist
            # count = count + 1
            # d_array = np.append(d_array,dist)
            # change to using normalised hamming distance
            d_cat_arr = np.append(d_cat_arr, dist)
            # print('target: ', ref_val ,'cell: ', cell_val,'dist: ',d,'count:',count)
    # if np.sum(d_array)==0:  breakpoint()
    """
    #implementingMedian Absolute Deviation
    #scipy stats.median_absolute_deviation not working
    #pseudo code from https://en.wikipedia.org/wiki/Median_absolute_deviation
    median = np.median(d_array)
    #med_array = np.full_like(d_array,median)
    #dev_array = np.subtract(d_array,med_array)#changed to make abs dev
    dev_array = np.empty_like(d_array)
    for i in range(d_array.size):
        dev_array[i] = abs(d_array[i]-median)
    mad = np.median(dev_array)
    sum = np.sum(d_array)
    distance = sum*mad
    #This does not work with lots of categorical data because with many
    #attribute values 1 or 0 mad is easily 0
    return distance
    """
    # if not using MAD
    # for use when not diving in defintion of aff_thresh
    # return (np.sum(d_array)/len(d_array))
    # get mad of continuous distances
    # mad = stats.median_absolute_deviation(d_cont_arr,axis=None)#median_abs_deviation not available contrary to web documentation
    d_cont = 0
    d_cat = 0
    if use_mads == True:
        mad = stats.median_abs_deviation(d_cont_arr, axis=None)
        sum_of_cat = np.sum(d_cat_arr)
        # distance for cat variables = 1/number of cat (sum of (abs distances/mad)
        for i in range(len(d_cont_arr)):
            if mad != 0:
                d_cont = d_cont + (abs(d_cont_arr[i]) / mad)
            else:
                d_cont = d_cont + abs(d_cont_arr[i])
        if len(d_cont_arr > 0):  # avoid /0 error
            d_cont = d_cont / len(d_cont_arr)
        for i in range(len(d_cat_arr)):
            d_cat = d_cat + d_cat_arr[i]
        if len(d_cat_arr) > 0:  # avoid /0 error
            d_cat = d_cat / len(d_cat_arr)
        # adjust for lengths of arrays for continuous and categorical features
        d_cont = d_cont * (len(d_cont_arr) / (len(d_cont_arr) + len(d_cat_arr)))
        d_cat = d_cat * (len(d_cat_arr) / (len(d_cont_arr) + len(d_cat_arr)))

        return d_cont + d_cat
        # return np.sum(d_array)
    else:  # no mads
        for i in range(len(d_cont_arr)):
            d_cont = d_cont + abs(d_cont_arr[i])
        if len(d_cont_arr > 0):  # avoid /0 error
            d_cont = d_cont / len(d_cont_arr)
        for i in range(len(d_cat_arr)):
            d_cat = d_cat + d_cat_arr[i]
        if len(d_cat_arr) > 0:  # avoid /0 error
            d_cat = d_cat / len(d_cat_arr)
        # adjust for lengths of arrays for continuous and categorical features
        d_cont = d_cont * (len(d_cont_arr) / (len(d_cont_arr) + len(d_cat_arr)))
        d_cat = d_cat * (len(d_cat_arr) / (len(d_cont_arr) + len(d_cat_arr)))
        return d_cont + d_cat


def get_neighborhood(reference_cell, pop, aff_thresh, dataset):
    neighbors = list()
    neighbors.append(reference_cell)  # add refencecell to neighbors
    # remove reference cell from pop
    del pop[0]
    out_pop = list()  # cells not in neighbourhood 2013.08.21
    # print('get n size of pop: ', len(pop) )
    for cell in pop:
        # if distance of population cell less than aff_thresh from reference cell add to neighbourhood
        # should this be add if more than aff_thresh?
        if combined_distance(reference_cell['value'][0], cell['value'][0], dataset) < aff_thresh:
            # if combined_distance(reference_cell['value'][0],cell['value'][0],dataset) > aff_thresh:
            neighbors.append(cell)
            # print('neighbourhood addition')
        else:
            out_pop.append(cell)
        # pop.pop(0)#dislike pop + pop
    # print('get_n...',len(neighbors),len(out_pop))
    return neighbors, out_pop


def affinity_suppress(population, aff_thresh, dataset):
    # print('start of affinity_suppress()')
    out_pop = []  # holds population of cells that are first in their neighbourhoods
    flag = True
    while flag:
        new_list = list()
        first = population[0]
        neighbors, population = get_neighborhood(first, population, aff_thresh,
                                                 dataset)  # cell becomes population [i] 2023.08.21
        # print('neighbourhood: ')
        sorted_neighbors = sort_by_dist(neighbors)  # changed suppression metric from cost to distance
        out_pop.append(sorted_neighbors[0])
        # print(len(neighbors),len(out_pop),len(population))
        # new line
        # print('sorted neighbourhood')
        '''BIG CHANGE 2023.08..21'''
        '''removing these constraints  to get bigger population'''
        """
        if len(sorted_neighbors) > 0:
            base_d = sorted_neighbors[0]['distance']
            top_d = sorted_neighbors[len(sorted_neighbors)-1]['distance']
            base_c = sorted_neighbors[0]['cost']
            top_c = sorted_neighbors[len(sorted_neighbors)-1]['cost']
            pop.append(sorted_neighbors[0])
            for p in population:
                #print('for p in population check distance and cost')
                # for all categorical data try new line here
                #what does this do ? add to list if:
                #the cells distance is same or greater than the highest distance and cells cost is greater than the top cost is add to list for output
                #the cells distance is same or less than the lowest distance and cells cost is less than the lowest cost is add to list for output                
                if  p['distance'] >= top_d and p['cost'] > top_c:
                    new_list.append(p)
                if p['distance'] <= base_d and p['cost'] < base_c:
                    new_list.append(p)


                #if  p['distance'] > top_d or p['distance'] < base_d:
                        #new_list.append(p)
                        ##add cells that are outside neighborhood to new_list

        """

        # for i in range(len(new_list)):            pop.append(new_list[i])

        # population = list()
        # population = new_list
        if len(population) == 0:
            flag = False
        # print('end suppresssion')

    return out_pop


def affinity_suppress_by_cost(population, aff_thresh, dataset):
    pop = []  # holds population of cells that are first in their neighbourhoods
    flag = True
    # population enters method sorted by distance sort by cost instead
    population = sort_by_cost(population)
    while flag:
        new_list = list()
        first = population[0]
        # neighbors = get_neighborhood(first, population, aff_thresh, dataset)# cell becomes population [i]
        neighbors = get_neighborhood(first, population, aff_thresh, dataset)
        sorted_neighbors = sort_by_cost(neighbors)  # changed suppression metric from cost to distance

        if len(sorted_neighbors) > 0:
            base_d = sorted_neighbors[0]['cost']
            top_d = sorted_neighbors[len(sorted_neighbors) - 1]['cost']
            pop.append(sorted_neighbors[0])

            for p in population:
                if p['cost'] > top_d or p['cost'] < base_d:
                    new_list.append(p)
        # for i in range(len(new_list)):            pop.append(new_list[i])
        population = list()
        population = new_list

        if len(population) == 0:
            flag = False
    return pop


def descale_decode(line, columns, dataset):
    df_new_line = pd.DataFrame(columns=columns, index=[0])
    df_line = pd.DataFrame(data=line, columns=columns)
    for col in columns:
        if col in dataset['continuous']:
            if dataset['scaler'] != '':
                sc = dataset['scaler'][col]
                df_new_line[col] = sc.inverse_transform(df_line[col].values.reshape(1, -1))
        if col in dataset['non_continuous']:
            if col in dataset['label_encoder']:
                le = dataset['label_encoder'][col]
                df_new_line[col] = le.inverse_transform(df_line[col].values.astype(int))
            # elif (col in dataset['binner']):potential error because non-continuous variables can be in binnner and label_encoder change to if
            """
            #because this never executed leave commented out for now
            if col in dataset['binner']:
                bin = dataset['binner'][col]
                temp_value = bin.inverse_transform(df_line[col].values.reshape(1,-1))
                for i in range(len(bin.bin_edges_[0])) :
                    if bin.bin_edges_[0][i] <= temp_value[0][0]:
                        count = i
                    else:
                        break
                df_new_line[col] = bin.bin_edges_[0][count]
            """
    return df_new_line


def search(search_space, max_gens, pop_size, num_clones, beta, num_rand, aff_thresh, problem_size, stop_condition,
           dataset, model, prediction, target_vals):
    print('begin search')
    num_of_att = len(dataset['columns']) - len(dataset['invariants']) - 1
    pop = []
    # col = []

    for i in range(pop_size):
        # print('get new cell')
        cell = new_cell(dataset, model, prediction, target_vals)
        if cell != None:
            pop.append(cell)
    if pop == []:
        return None, pd.DataFrame(columns=dataset['X_columns_with_dummies'])
    best = None
    print('initial pop size: :', len(pop))
    for g in range(max_gens):
        print('Generation: ', g, ' of ', max_gens)
        # print('gc_count before: ',gc.get_count())
        gc.collect()
        # print('gc_count after: ', gc.get_count())
        # need to remove cells that are None type
        for p in pop:
            if p != None:
                # print('p: ',p)
                # print(' p[value]: ' ,p['value'])
                p['cost'] = objective_function(p['value'], model, prediction)
            else:
                pop.remove(p)
        calculate_normalized_distance(pop)
        calculate_normalized_cost(pop)
        pop = sort_by_cost(pop)
        # pop = ort_by_distance(pop)
        if best == None:
            best = pop[0]
        else:
            if pop[0]['distance'] < best['distance']:
                best = pop[0]
        if sort_by == 'distance':
            avgCost, progeny = average_distance(pop), None
        elif sort_by == 'cost':
            avgCost, progeny = average_cost(pop), None
        else:
            print('failure of global sort_by variable')
            breakpoint()
        # print('pop number: ',len(pop))
        flag = True
        avg_flag_count = 0
        while flag:
            progeny = list()
            # print('appending cells')
            for p in pop:
                cell = clone_cell(beta, num_clones, p, search_space, dataset, model, prediction, target_vals)
                progeny.append(cell)
                # print('cloning cell')
            if sort_by == 'distance':
                prog_avg_cost = average_distance(progeny)
            elif sort_by == 'cost':
                prog_avg_cost = average_cost(progeny)
            else:
                print('failure of global sort_by variable in cloning')
                breakpoint()
            # prog_avg_cost = average_distance(progeny)
            # prog_avg_cost = average_cost(progeny)
            print('testing avg cost i lower')
            if prog_avg_cost < avgCost:
                flag = False
            if avg_flag_count > 9:  # escape from this loop
                flag = False
                print('escaping testing avg cost i lower')
            avg_flag_count += 1

        # print('progeny number: ',len(progeny))
        # print('g: ',g, ' pop_size: ',len(pop), ' prog_avg_cost ',prog_avg_cost, ' avgCost ',avgCost)

        if num_of_att > 1:
            # change to suppress by cost see it changes results
            # print(affinity_suppress)
            # sorted_progeny = affinity_suppress(progeny,aff_thresh,dataset)
            # sorted_progeny = affinity_suppress_by_cost(progeny,aff_thresh,dataset)
            # sorted_progeny = progeny

            # suppression only by dist (unless special circumstances )
            sorted_progeny = affinity_suppress(progeny, aff_thresh, dataset)
            if sort_by == 'distance':
                # print('sorting progeny')
                sorted_progeny = sort_by_dist(sorted_progeny)
            elif sort_by == 'cost':
                sorted_progeny = sort_by_cost(sorted_progeny)
            else:
                print('failure of global sort_by variable in suppression')

            # print('end of suppression')
        else:
            # IS THIS WRONG OR JUST NEVER CALLED
            print('this branch should never be called subject to further investigation')
            breakpoint()
            # print('suppression by one feature')
            # if only one attribute and that attribute is non_continuous pointless sorting by distance as all are equally far apart
            columns = dataset['X_columns']
            for col in columns:
                if columns in dataset['invariants']:
                    columns.remove(col)

            if columns[0] in dataset['non_continuous']:
                # print('non_continuous')
                sorted_progeny = affinity_suppress_by_cost(progeny, aff_thresh, dataset)
                sorted_progeny = sort_by_cost(sorted_progeny)
            else:
                # print('continuous')
                sorted_progeny = affinity_suppress(progeny, aff_thresh, dataset)
                sorted_progeny = sort_by_dist(sorted_progeny)

        best = sorted_progeny[0]

        # print('gen: ',g+1,' pop_size: ',len(pop), 'soln 1: ', 'fitness: ', pop[0]['cost'],' value: ',pop[0]['value'])
        # print('gen: ',g+1,' pop_size: ',len(pop), 'soln 2: ', 'fitness: ', pop[1]['cost'],' value: ',pop[1]['value'])

        stop_cost = 0
        # for i in range(problem_size):#this is wrong should be pop_size
        for i in range(len(sorted_progeny)):
            # print('check stop cost')
            stop_cost = stop_cost + sorted_progeny[i]['distance']
        stop_cost = stop_cost / (problem_size * len(sorted_progeny))
        print('stop_cost: ', stop_cost, 'stop_cost/problem_size: ', stop_cost / (problem_size * len(sorted_progeny)))
        print('pop size after suppression: ', len(sorted_progeny))
        # print('gen: ',g+1,' pop_size: ',len(sorted_progeny), 'best ', problem_size,' fitness: ', stop_cost)
        if stop_cost < stop_condition:
            print('break because |cost| < ', stop_condition)
            break
        if g < max_gens - 2:  # (max_gens/2 -1)this inserts random cells in evry generation, other term stops adding cells halfway through and just optimises no cells added in last generation
            # print('g + : ',g)
            # print(g,' inject random cells')
            # old cose for i in range(int(len(sorted_progeny) *0.4)):#update here to
            # add upto 40% of initial cells not current cell
            # have a ceiling of initial cell number
            new_cell_num = int(pop_size * new_cell_rate)  # get initial population from global variable

            for i in range(new_cell_num):  # get from global
                # print('adding new cells')
                if len(sorted_progeny) < pop_size:  # get initial pop from global
                    # LOOK HERE for potential error may be adding NONE value cells
                    """
                    adding_cell = new_cell(dataset,model,prediction,target_vals)
                    if adding_cell['distance'] != None:
                        sorted_progeny.append(adding_cell)
                    """
                    sorted_progeny.append(new_cell(dataset, model, prediction, target_vals))

            # else:#print(g,' do not inject random cells')

        """#removing this new section of code a it is adding an error
        if sorted_progeny == None:
            print('sorting after adding new cells, sorted progeny is NoneType')
            breakpoint()
        if sort_by == 'distance':
            #error here
            sorted_progeny = sort_by_dist(sorted_progeny)
        elif sort_by == 'cost':
            prog_avg_cost = sort_by_cost(sorted_progeny)
        else:
            print('failure of global sort_by variable in adding new cells to pop')
            breakpoint()
        """
        pop = sorted_progeny

    # decode, debin and descale
    # best = descale_decode(best['value'][0].reshape(1,-1),dataset['columns'],dataset)
    out_pop = list()
    # create csv for writing runtime information to
    with open('opt-AINet_results.csv', 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        kp = keras_pred(model, pop[0]['value'][0].reshape(1, -1))
        correct = False
        if prediction == 0:
            if kp[1] > kp[0]:
                correct = True
        else:
            if kp[0] > kp[1]:
                correct = True
        row = (prediction, kp[0], kp[1], correct)
        writer.writerow(row)
    for cell in sorted_progeny:
        temp_values = cell['value'][0].reshape(1, -1)
        # comment out descale_decode because happens in writing to db as well
        # temp_values = descale_decode(temp_values,dataset['X_columns'],dataset)
        # temp_values = pd.DataFrame(data=temp_values, columns = dataset['columns'])
        line_columns = get_line_columns(dataset)
        """
        line_columns =dataset['X_columns']
        if dataset['use_dummies'] == True:
            line_columns = dataset['X_columns_with_dummies']
        """
        temp_values = pd.DataFrame(data=temp_values[0].reshape(1, -1),
                                   columns=line_columns)  # columns = dataset['X_columns'])
        values = list()

        for col in line_columns:  # dataset['X_columns']:
            k = col
            v = temp_values[col].values[0]
            value = {k: v}
            values.append(value)

        cost = cell['cost']
        distance = cell['distance']
        entry = {'values': values, 'cost': cost, 'distance': distance}
        out_pop.append(entry)
    best = out_pop[0]
    ais_columns = list()
    line_columns = get_line_columns(dataset)
    """
    line_columns = dataset['X_columns']
    if dataset['use_dummies'] == True:
        line_columns = dataset['X_columns_with_dummies']
    """
    for col in line_columns:  # dataset['X_columns']:
        ais_columns.append(col)
    # ais_columns = (copy.deepcopy(dataset['X_columns']))
    ais_columns.append('distance')
    ais_columns.append('cost')
    df_pop = pd.DataFrame(columns=ais_columns)
    for cell in sorted_progeny:
        temp_values = cell['value'][0].reshape(1, -1)
        # descale_decode disabled as work done in write_to_db
        # temp_values = descale_decode(temp_values,dataset['X_columns'],dataset)
        temp_values = np.append(temp_values, cell['distance'])
        temp_values = np.append(temp_values, cell['cost'])
        df_temp = pd.DataFrame(data=temp_values.reshape(1, -1), columns=ais_columns)
        # df_pop = df_pop.append(df_temp)
        df_pop = pd.concat([df_pop, df_temp])
        # print('df_temp: ',df_temp)
    print(df_pop)
    return best, df_pop


def create_optAINet_explanation(target_vals, y, pop, dataset, prob_dict, record_number, db_file, imp_arr):
    # descale/decode line back to human readable values
    # decoding causing error generating nans and NULLs

    # decoding now happend=s in write_db class
    # target_vals = descale_decode(target_vals, dataset['X_columns'], dataset)
    description = 'optAINet'
    # db_file = '../explanation_store/ex_store.db'
    conn = create_connection(db_file)

    with(conn):
        explanations_id = db_add_explanations(conn, description, record_number)
        records_id = db_add_records(conn, explanations_id)
        # add record all operands will be '='
        # add class
        class_name = dataset['class_name']
        class_value = str(y)
        operand = '='
        attributes_id = db_add_attributes_bundle(conn, class_name, operand, y)
        records_attributes_id = db_add_records_attributes(conn, attributes_id, records_id)

        """
        #add fidelity stat to records

        k= 'fidelity'
        v = fidelity

        attributes_id  = db_add_attributes_bundle(conn, k, operand , v)
        records_attributes_id = db_add_records_attributes(conn,attributes_id,records_id)
        """

        # in this section create section for if using dummies
        # add all non class columns
        # ***************ADD RECORD VALUES*************
        if dataset['use_dummies'] == True:
            # print('Records')
            for col in dataset['X_columns']:
                # if col != dataset['class_name']:redundant if using X_columns
                att_type = col
                operand = '='
                # att_value = str(target_vals[col][0])
                # att_value = int(target_vals[col][0])
                # att_value now loop through dataset['dummies'] dict
                att_value = ''
                if col in dataset['continuous']:
                    att_value = str(target_vals[col][0])

                else:
                    for sub_col in dataset['dummy'][col]:
                        if target_vals[sub_col][0] == 1:
                            att_value = sub_col

                # print('Att type: ',att_type,' Att val: ',att_value)

                att_value = decode_debin_db(att_type, att_value, dataset)
                if dataset['data_human_dict'] != {}:

                    if att_type in dataset['continuous']:
                        att_type = dataset['data_human_dict'][att_type]
                    else:
                        att_type = dataset['data_human_dict'][att_value]['name']
                        att_value = dataset['data_human_dict'][att_value]['value']

                attributes_id = db_add_attributes_bundle(conn, att_type, operand, att_value)
                records_attributes_id = db_add_records_attributes(conn, attributes_id, records_id)

            # add importances
            importance_id = db_add_importances(conn, explanations_id)
            for i in range(len(imp_arr)):
                line_columns = get_line_columns(dataset)
                # att_type = dataset['X_columns'][i]
                att_type = line_columns[i]
                operand = '='
                att_value = imp_arr[i]
                attribute_id = db_add_attributes_bundle(conn, att_type, operand, att_value)
                importances_attributes_id = db_add_importances_attributes(conn, importance_id, attribute_id)
        else:
            for col in dataset['X_columns']:
                # if col != dataset['class_name']:redundant if using X_columns
                att_type = col
                operand = '='
                # att_value = str(target_vals[col][0])
                att_value = int(target_vals[col][0])
                att_value = decode_debin_db(att_type, att_value, dataset)
                attributes_id = db_add_attributes_bundle(conn, att_type, operand, att_value)
                records_attributes_id = db_add_records_attributes(conn, attributes_id, records_id)
            # add importances
            importance_id = db_add_importances(conn, explanations_id)
            for i in range(len(imp_arr)):
                att_type = dataset['X_columns'][i]
                operand = '='
                att_value = imp_arr[i]
                attribute_id = db_add_attributes_bundle(conn, att_type, operand, att_value)
                importances_attributes_id = db_add_importances_attributes(conn, importance_id, attribute_id)

        # add invariant columns to db
        for col in dataset['invariants']:
            att_type = col
            operand = '='
            att_value = ''
            attributes_id = db_add_attributes_bundle(conn, att_type, operand, att_value)
            invariants_attributes_id = db_add_invariants_attributes(conn, attributes_id, records_id)
            """
            att_type = col
            operand = '='
            #att_value = str(target_vals[col][0])
            att_value = int(target_vals[col][0])
            att_value = decode_debin_db(att_type,att_value,dataset)
            attributes_id = db_add_attributes_bundle(conn, att_type, operand , att_value)
            invariants_attributes_id = db_add_invariants_attributes(conn,attributes_id,records_id)
            """

        # ***************Add CFs***************
        # print('CFs')
        for i in range(len(pop)):  # change to lesser of len(pop) or X, suggested value of X = 5
            # if i < 5:#is this thing that is making my expalnations wrong? is a sixth (or more) attribute being forgotten in the model?
            # add deltas to db
            deltas_id = db_add_deltas(conn, explanations_id)
            operand = '='
            att_name = ''
            att_value = ''
            # for col in pop.columns:
            # change to dataset['X_columns']
            # print('pop i: ',i)
            for col in pop.columns:
                # if col != dataset['class_name']:redundant for X_columns
                att_name = col  # key for attributes_deltas
                # att_value = int(pop.iloc[i][col])#value key for attributes_deltas
                # cost and distance cannot be an int
                line_columns = get_line_columns(dataset)
                if col in line_columns:
                    if dataset['use_dummies'] == True:
                        if col in dataset['continuous']:
                            # att_value = str(target_vals[col][0])
                            att_value = (pop[col].values[i])
                            # print('att name: ',att_name,' att value: ',att_value)
                            att_value = decode_debin_db(col, att_value, dataset, decode=True)
                            att_name = dataset['data_human_dict'][col]
                            attributes_id = db_add_attributes_bundle(conn, att_name, operand, att_value)
                            attributes_deltas_id = db_add_deltas_attributes(conn, deltas_id, attributes_id)

                        else:
                            for dummied_col in dataset['dummy']:
                                for dummy_value in dataset['dummy'][dummied_col]:
                                    if (int(pop[dummy_value].values[i]) == 1) and (col == dummy_value):
                                        att_name = dummied_col
                                        att_value = dataset['data_human_dict'][col]['value']
                                        att_name = dataset['data_human_dict'][col]['name']
                                        # print('att name: ',att_name,' att value: ',att_value)
                                        att_value = decode_debin_db(col, att_value, dataset, decode=True)
                                        attributes_id = db_add_attributes_bundle(conn, att_name, operand, att_value)
                                        attributes_deltas_id = db_add_deltas_attributes(conn, deltas_id, attributes_id)

                    else:
                        # if col in dataset['X_columns']:#ie not distance or cost
                        if col in line_columns:  # ie not distance or cost
                            att_value = int(pop.iloc[i][col])
                            # with all decoding for opt-AINet being done values should need decoding now changing decode to True
                            att_value = decode_debin_db(col, att_value, dataset, decode=True)
                            """
                            if col in dataset['continuous']:#only continuous attributes have a difference
                                diff_name = col # key for differences_attributes
                                diff_value = att_value - target_vals[col][0]
                                #att_value = target_vals[col][0] att value not used
                                #att_value = decode_debin_db(col,att_value,dataset)#why is decode debin here when done in previously
                                attributes_id = db_add_attributes_bundle(conn, diff_name, operand , diff_value)
                                differences_attributes_id = db_add_differences_attributes(conn,deltas_id,attributes_id)
                            """
                else:
                    # distance and cost
                    # distance and cost must not be transformed to ints
                    att_value = str(pop.iloc[i][col])
                    # print('decoded att_name: ', att_name,' att_value: ',att_value)
                    attributes_id = db_add_attributes_bundle(conn, att_name, operand, att_value)
                    attributes_deltas_id = db_add_deltas_attributes(conn, deltas_id, attributes_id)

        for key in prob_dict:
            class_probabilities_id = db_add_class_probabilities(conn, explanations_id)
            operand = '='
            attributes_id = db_add_attributes_bundle(conn, key, operand, prob_dict[key])
            class_probabilities_attributes_id = db_add_class_probabilities_attributes(conn, class_probabilities_id,
                                                                                      attributes_id)


def get_invariants(dataset, num_variants, X, y):
    # num_variants = 10
    lr = LinearRegression()
    lr = lr.fit(X, y)
    coeffs = lr.coef_
    coeff_dict = {}
    i = 0
    for col in dataset['X_columns']:
        coeff_dict[col] = abs(coeffs[i])
        i = i + 1
    # sort coeff_dict by abs magnitude
    sorted_coeff_dict = sorted(coeff_dict.items(), reverse=True, key=lambda x: x[1])
    # loop all invariants all outside top 5
    invariant_list = list()

    for i in range(num_variants, X.shape[1]):
        invariant_list.append(sorted_coeff_dict[i][0])
    return invariant_list  # dataset_cat['invariants']=invariant_list

    line_columns = dataset['X_columns']
    if dataset['use_dummies'] == True:
        line_columns = dataset['X_columns_with_dummies']
    return line_columns


def init_var_optAINet(blackbox, X, line_number, dataset, prob_dict, db_file, imp_arr, df_out):
    # prediction = util.predict_label(prediction)
    def predict_label(y):
        if y < 0.5:
            return 0
        else:
            return 1

    # prediction = predict_label(blackbox.predict_proba(X[line_number].reshape(1,-1)))
    #prediction = predict_label(blackbox.predict(X[line_number].reshape(1, -1))) #old line
    #new line for keras 3
    prediction = predict_label(predict_single(blackbox,X[line_number]))
    # problem configuration

    line = X[line_number]  # np rows
    aff_thresh = ((search_space[1] - search_space[0]) / affinity_constant) / (
            1 + len(dataset['X_columns']) - len(dataset['invariants']))  # 0.01 is a fiddle factor
    best, df_pop = search(search_space, max_gens, pop_size, num_clones, beta, num_rand, aff_thresh, problem_size,
                          stop_condition, dataset, blackbox, prediction, line)

    # print(df_pop)
    if best == None:
        print('no cell created for this run')
        return False, df_pop  # False for no cell with these invariant attributes
    # if df_pop == []:return False,df_out

    if df_out.shape[0] == 0:
        df_out = df_pop
    else:
        # df_pop = df_out.append(df_pop)
        df_pop = pd.concat[(df_out, df_pop)]
    line_cost = keras_pred(blackbox, line.reshape(1, -1))
    """
    df_temp = df_pop.copy(deep=True)
    df_temp = df_temp.reset_index(drop=True)

    print('line: ',line_number, 'line costs: ', line_cost,' best costs: ', keras_pred(blackbox,(df_temp.values[0][0:23]).reshape(1,-1)))
    #[1:24] removes index','cost' and 'distance'
    #print('df_pop.values: ', df_pop.values[0][0:23])

    for col in df_temp.columns:
        val = df_temp[col][0]
        print('col: ',col, ' val: ', val)
        if col in dataset['label_encoder']:#excludes 'index','distance' and 'cost'
            val = decode_debin_db(col,np.array([int(val)]).ravel(),dataset,decode=True)
            print('decoded_val: ',val)
    """
    # add 0, 0 for distance and cost to line
    ais_line = copy.deepcopy(line)
    ais_line = np.append(ais_line, 0)  # 0 distance to itself
    ais_line = np.append(ais_line, line_cost[1])
    ais_columns = list()
    line_columns = get_line_columns(dataset)
    # for col in dataset['X_columns']:
    for col in line_columns:
        ais_columns.append(col)
    # ais_columns = (copy.deepcopy(dataset['X_columns']))
    ais_columns.append('distance')
    ais_columns.append('cost')

    df_line = pd.DataFrame(data=ais_line.reshape(1, -1), columns=ais_columns)
    print('length of invariants: ', len(dataset['invariants']))
    # if len(dataset['invariants'])>0:#if these condition are not met back to calling method and remove another invariant
    # if df_pop.values.shape[0]<5
    # print('not writing for df_out rows',df_out.values.shape[0])
    # return False, df_out
    # db_file = 'explanation_store/ex_store.db'
    # conn = create_connection(db_file)
    # create_optAINet_explanation(conn,df_line,dataset['possible_outcomes'][prediction],df_pop,dataset,db_file,prob_dict)
    create_optAINet_explanation(df_line, dataset['possible_outcomes'][prediction], df_out, dataset, prob_dict,
                                line_number, db_file, imp_arr)
    if len(dataset['invariants']) > 0:
        return False, df_out
    else:
        return True, df_out  # True for cell created with these invariant attributes
