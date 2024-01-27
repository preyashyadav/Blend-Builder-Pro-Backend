from flask import Flask, jsonify, Response, request
from flask_cors import CORS
import pandas as pd
import json
import csv
import os
import itertools
from utils import *
from optimization import *

from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret-key-123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
cors = CORS(app)

@app.route('/test', methods=['GET', 'POST'])
def test():
    sample = {"Test": "Hello world"}
    return Response(sample, status=200)

@app.route('/upload', methods=['GET', 'POST'])
def fileUpload():
    file = request.files['file']
    filename = secure_filename(file.filename)
    target="data"
    destination_filename = "dataset.csv"
    destination="/".join([target, destination_filename])
    file.save(destination)

    return Response(destination, status=200)

@app.route('/save_mapping', methods=['GET', 'POST'])
def saveMapping():
    data = json.loads(request.data)
    destination_filename = "mapping.json"
    with open('data/'+destination_filename, 'w') as fp:
        json.dump(data, fp)

    return Response(destination_filename, status=200)

@app.route('/get_mapping', methods=['GET', 'POST'])
def getMapping():
    data = json.loads(request.data)
    feature_type = data['type']
    destination_filename = "mapping.json"
    with open('data/'+destination_filename) as json_file:
        mapping = json.load(json_file)
    
    request_mapping = {}
    for feature in mapping:
        if mapping[feature][0] == feature_type:
            request_mapping[feature] = mapping[feature][1]

    return request_mapping

@app.route('/data_cleaning', methods=['GET', 'POST'])
def data_cleaning():
    data = json.loads(request.data)
    path = data['path']
    input_features = data['input_features']
    output_features = data['output_features']

    destination_filename = "mapping.json"
    with open('data/'+destination_filename) as json_file:
        mapping = json.load(json_file)

    cleaned_input_features = []
    for feature in input_features:
        if mapping[feature][1] != '-':
            cleaned_input_features.append(feature)


    df = pd.read_csv(path)
    df = df[cleaned_input_features+output_features]

    for column in df.columns:
        df[column] = df[column].apply(lambda x: preprocess_text(x))

    for feature in cleaned_input_features:
        df[feature] = df[feature].fillna(df[feature].min())

    for feature in output_features:
        df[feature] = df[feature].fillna(0)
    df.to_csv("./data/preprocessed.csv", index=False)

    return Response("./data/preprocessed.csv", status=200)

@app.route('/get_columns_df', methods=['GET', 'POST'])
def get_columns_df():
    df = pd.read_csv("./data/dataset.csv")
    return jsonify(list(df.columns.values))
 
@app.route('/get_columns', methods=['GET', 'POST'])
def get_columns():
    # json(request.data.decode())
    destination_filename = "mapping.json"
    with open('data/'+destination_filename) as json_file:
        mapping = json.load(json_file)
    if(request.data.decode() != ""):
        data = json.loads(request.data)
        path = data['path']
    else:
        path = "data/" + request.form['fileName']
    list_of_column_names = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for row in csv_reader:
            list_of_column_names.append(row)
            break
    columns = []
    for column in list_of_column_names[0]:
        if column in mapping and mapping[column][1] != "-":
            columns.append(column)
    return jsonify(columns)


# @app.route('/optimize_problem', methods=['GET', 'POST'])
# def optimize_problem():
#     data = json.loads(request.data)
#     print(data)
#     path = data['path']
#     input_features = list(data['input_features'].keys())
#     output_features = list(data['output_features'].keys())
#     lower_bounds = list(data['lower_bounds'].values())
#     upper_bounds = list(data['upper_bounds'].values())
#     sample_input = list(data['sample_input'].values())

#     df = pd.read_csv(path)

#     destination_filename = "mapping.json"
#     with open('data/'+destination_filename) as json_file:
#         mapper = json.load(json_file)

#     cleaned_input_features = []
#     for feature in input_features:
#         if mapper[feature][1] != '-':
#             cleaned_input_features.append(feature)

#     mapper_data = json.load(open('./data/mapping.json'))
#     mapping = {}
#     for feature in mapper_data:
#         if mapper_data[feature][0] == 'output':
#             mapping[feature] = mapper_data[feature][1]

#     problem, r2_scores = define_problem(df=df, 
#                    input_features=cleaned_input_features, 
#                    output_features=output_features, 
#                    lower_bounds=lower_bounds, 
#                    upper_bounds=upper_bounds, 
#                    sample_input=sample_input,
#                    mapping=mapping)
    
#     X, F = minimize_problem(problem=problem, n_gen=25)
#     return {'r2_scores': r2_scores, "x": X, "F": F}


@app.route('/optimize_problem', methods=['GET', 'POST'])
def optimize_problem():
    data = json.loads(request.data)
    path = data['path']
    input_features = list(data['input_features'].keys())
    output_features = list(data['output_features'].keys())
    lower_bounds = list(data['lower_bounds'].values())
    upper_bounds = list(data['upper_bounds'].values())
    sample_input = list(data['sample_input'].values())  
    # Initially it was [62, 22] -> New app [[60, 70, 2], [20, 25, 1]]
    # 60 - start, 70 - end, 2 - interval
    # [[60, 62, 64, 66, 68], [20, 22, 24]] --> [60, 20], [60, 22], [60, 24], [62, 20]
    sample_space = []
    for inp in sample_input:
        sample_space.append(list(range(inp[0], inp[1], inp[2])))
    complete_sample_space = itertools.product(*sample_space)
    df = pd.read_csv(path)
    mapper_data = json.load(open('./data/mapping.json'))
    mapping = {}
    for feature in mapper_data:
        if mapper_data[feature][0] == 'output':
            mapping[feature] = mapper_data[feature][1]

    best_X, best_F, best_sample_inp = None, None, None
    for sample_inp in complete_sample_space:
        problem, r2_scores = define_problem(df=df, 
                    input_features=input_features, 
                    output_features=output_features, 
                    lower_bounds=lower_bounds, 
                    upper_bounds=upper_bounds, 
                    sample_input=sample_inp,
                    mapping=mapping)
    
        X, F = minimize_problem(problem=problem, n_gen=10)
        if best_X == None:
            best_X = X
            best_F = F
            best_sample_inp = sample_inp
        else:
            if min(best_F) < min(F):
                best_F = F
                best_X = X
                best_sample_inp = sample_inp

    return {'r2_scores': r2_scores, "x": X, "F": F, "sample_input": best_sample_inp}



