import json
import sklearn
import pandas as pd
import numpy as np
import sqlite3
import requests


def exp_to_JSON(x,explanation,dataset,covered,precision, file_name):
    #function outputs LORE output to JSON
    #add decode to print out to make intelligible output
    #decode x
    r=list()
    label_encoder = dataset['label_encoder']
    binner = dataset['binner']
    number_of_bins = dataset['number_of_bins']
    scaler = dataset['scaler']
    for k,v in x.items():
        field={}
        value={}
        if k == dataset['class_name']:
            field['field'] = k
            value['value'] = str(v)
        else:
            if k in dataset['non_continuous']:
                if k in label_encoder:
                    x_arr = label_encoder[k].inverse_transform(x[k].astype(int).reshape(-1,1))
                    field['field'] = k
                    value['value'] = str(x_arr[0])
                if k in binner:
                    x_arr = binner[k].inverse_transform(x[k].astype(int).reshape(-1,1))
                    field['field'] = k
                    value['value'] = str(x_arr[0])
            else:#continuous
                new_val = scaler[k].inverse_transform(x[k].reshape(-1,1))
                field['field'] = k
                value['value'] = str(new_val[0][0])
        d={}
        d.update(field)
        d.update(value)
        r.append(d)
    record = {'record': r}
    exp_dict = explanation[0][1].items()
    r = list()
    for k,v in exp_dict:
        field={}
        value={}
        if k in dataset['non_continuous']:
            if k in label_encoder:
                v = label_encoder[k].inverse_transform([round(float(v))])
                v = str(v[0])
            if k in binner:
                v = np.array(float(v))
                v = binner[k].inverse_transform(v.reshape(1,-1))
                v = (v[0][0])
                for i in range(len(binner[k].bin_edges_[0])) :
                    if binner[k].bin_edges_[0][i] <= v:
                        count = i
                    else:
                        break
                v= str(binner[k].bin_edges_[0][count])


        else:#eg continuous
            """
            problem here how to regex the <,=and > away from the numbers
            rescale the numbers then recreate the ruleself.
            use substring
            """
            if '=>' in v:
                end = len(v)
                ind = v.index('=')
                #ind_2 = v.index('>') #always 0
                sub_1 = v[1:ind]
                sub_2 = v[ind+1:end]
                #use debugger to catch this condition
                import pdb; pdb.set_trace()
                #elif '>' in v: elif '<' in v: elif '=' in v: all the same
            else:
                if '=' in v:
                    end = len(v)
                    ind = v.index('=')
                    sub = v[ind+1:end]
                    char_0 = v[0:ind+1]
                    new_val = scaler[k].inverse_transform([float(sub)])
                    new_val = round(new_val[0])
                    field['field'] = k
                    v = char_0 + str(new_val)
                    value['value'] = v

                else:
                    end = len(v)
                    ind = 0#v.index('=')
                    sub = v[ind+1:end]
                    char_0 = v[0:ind+1]
                    new_val = scaler[k].inverse_transform([float(sub)])
                    new_val = round(new_val[0])
                    field['field'] = k
                    v = char_0 + str(new_val)
                    value['value'] = v


        field['field']=k
        value['value']=v
        d={}
        d.update(field)
        d.update(value)
        r.append(d)

    #decoded_rule = {'rule':r}

    exp_class = explanation[0][0]

    digit = int(exp_class[dataset['class_name']])
    c = dataset['possible_outcomes'][digit]
    #rule =  {'rule': decoded_rule,'result': c}
    rule =  {'rule': r ,'result': c}
    d = list()
    for delta in explanation[1]:
        for k,v in delta.items():
            field={}
            value={}
            if k in dataset['non_continuous']:
                if k in label_encoder:
                    v = label_encoder[k].inverse_transform([[round(float(v))]])
                    v = str(v[0])
                if k in binner:
                    v = binner[k].inverse_transform([[round(float(v))]])
                    v = str(v[0])
            else:
                if '=>' in v:
                    end = len(v)
                    ind = v.index('=')
                    #ind_2 = v.index('>') #always 0
                    sub_1 = v[1:ind]
                    sub_2 = v[ind+1:end]
                    #use debugger to catch this condition
                    import pdb; pdb.set_trace()
                    #elif '>' in v: elif '<' in v: elif '=' in v: all the same
                elif '<=' in v:
                    end = len(v)
                    ind = v.index('=')
                    sub = v[ind+1:end]
                    char_0 = v[0:ind+1]
                    new_val = scaler[k].inverse_transform([float(sub)])
                    new_val = round(new_val[0])
                    field['field'] = k
                    value['value'] = str(new_val)
                    v = v[0] + str(new_val)

                else:
                    end = len(v)
                    sub = v[1:end]
                    char_0 = v[0]
                    new_val = scaler[k].inverse_transform([float(sub)])
                    new_val = round(new_val[0])
                    field['field'] = k
                    value['value'] = str(new_val)
                    v = v[0] + str(new_val)
            field['field']=k
            value['value']=v
            part={}
            part.update(field)
            part.update(value)
            d.append(part)
    D = {'delta':d}

    j={}
    j.update(record)
    j.update(rule)
    j.update(D)
    cov = {'covered':str(covered)}
    j.update(cov)
    pre = {'precision':str(precision)}
    j.update(pre)
    print('j: ',j)
    file_name = 'LORE_output/'+file_name
    with open(file_name,'w') as outfile:
        json.dump(j,outfile)

    endpoint = 'https://app.studio.arria.com:443/alite_content_generation_webapp/text/eVEXEDwmz5J'
    api_key = 'eyJhbGciOiJIUzUxMiJ9.eyJqdGkiOiJ3QkdpWHRpMjlhZW9pa3JHWXpQejBCcEYiLCJpYXQiOjE1NjEyMTA0MjYsImV4cCI6MTcxODg5MDQyNiwiaXNzIjoiQUxpdGUiLCJzdWIiOiJkRnZlS1hEVVFoUnkiLCJBTGl0ZS5wZXJtIjpbInByczp4OmVWRVhFRHdtejVKIl0sIkFMaXRlLnR0IjoidV9hIn0.yVs1tx6mfzbaZpe8M4LiltA8WURuKRY96zj1DXvRA9SY0bCzgjRBrMm1jLGrHhaS7CKv6yLqR2onMsuWlZHiBQ'

    data_dict = {}
    data_dict.update({'id':'Primary'})
    data_dict.update({'type':'json'})
    data_dict.update({'jsonData':j})
    l = list()
    l.append(data_dict)
    data = {}
    data.update({'data':l})
    data.update({'projectArguments':None})
    data.update({'options':None})
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
