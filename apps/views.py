# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

# Flask modules
from msvcrt import kbhit
from flask   import Flask, session, render_template, request, abort, redirect, url_for, flash, send_from_directory,send_file, jsonify
from jinja2  import TemplateNotFound
import os
import tempfile
# import framework_webapp.scripts.improve_quality as iq
# import framework_webapp.scripts.associationRules as assr
import pandas as pd

from ydata_profiling import ProfileReport
import numpy as np


# import framework_webapp.scripts.knowledge as kb
from datetime import datetime
from werkzeug.utils import secure_filename
import apps.scripts.kb_test as kb
import apps.scripts.allThePlots as myPlots
import apps.scripts.quality_dimensions as quality_dims
import apps.scripts.improve_quality_laura as improve
import apps.scripts.data_imputation_tecniques as imputes
from apps.scripts.preprocessing import preprocess_input_df
from flask_session import Session

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# App modules
from apps import app

@app.template_filter('json_round')
def json_round_filter(json_obj):
    def _json_round(obj):
        if isinstance(obj, float):
            return round(obj, 2)
        elif isinstance(obj, dict):
            return {_json_round(key): _json_round(val) for key, val in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_json_round(elem) for elem in obj]
        else:
            return obj

    return _json_round(json_obj)



HERE = os.path.dirname(os.path.abspath(__file__))

SAMPLE_DATA = {
    'iris': os.path.join(HERE, 'datasets/iris.csv'),
    'beers': os.path.join(HERE, 'datasets/beers.csv'),
}


DF_NAME = 'datasets/data.csv'
DF_NAME_TEMP = 'datasets/data_temp.csv'

app.secret_key = 'my_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

dataframes = [] #global variable
column_list = [0]

def get_columns(columns):
    if isinstance(columns, str):        
        columns = columns.replace("[", "")
        columns = columns.replace("]", "")
        columns = columns.replace("'", "")
        columns = columns.replace(" ", "")
        columns = columns.split(",")
    return columns

def save_data(df):
    # save the cleaned dataset as csv
    df.to_csv(os.path.join(HERE, DF_NAME))

def save_data_temp(df):
    # save the cleaned dataset as csv
    df.to_csv(os.path.join(HERE, DF_NAME_TEMP))


def get_data(name, dirname):
    upload_path = os.path.join(tempfile.gettempdir(), dirname)
    data_path = os.path.join(upload_path, name + '.csv')
    if not os.path.exists(data_path):
        abort(404)

    try:
        df = pd.read_csv(data_path)
    except pd.errors.ParserError:
        flash('Bad CSV file – could not parse', 'warning')
        return redirect(url_for('home'))

    (df, columns) = preprocess_input_df(df)

    return df, columns

def last_dataframe():
    global dataframes
    df = dataframes[-1]
    
    # Get the last DataFrame in the session list
    # df_list = session.get('df_list', [])
    # # last_df = df_list[-1] if len(df_list) > 0 else None

    # last_element = df_list[-1]

    # # convert the last element to a DataFrame object
    # df = pd.DataFrame(last_element)
    
    # Render a template that displays the DataFrame
    return df

def rank_kb(df, algorithm):
    kb_read_example = pd.read_csv("apps/scripts/kb-toy-example.csv", sep=",")
    ranking_kb = kb.predict_ranking(kb_read_example, df, algorithm)
    ranking_kb = str(ranking_kb)
    ranking_kb = ranking_kb.replace("[", "")
    ranking_kb = ranking_kb.replace("]", "")
    ranking_kb = ranking_kb.replace("'", "")
    ranking_kb = ranking_kb.split()
    return ranking_kb

def rank_dim(accuracy, uniqueness, completeness):
    ordered_values = sorted([accuracy, uniqueness, completeness])
    ranking_dim = []
    for i in range(2):
        if ordered_values[i] == accuracy:
            ranking_dim.append('ACCURACY')
        if ordered_values[i] == completeness:
            ranking_dim.append('COMPLETENESS')
        if ordered_values[i] == uniqueness:
            ranking_dim.append('UNIQUENESS')
    return str(ranking_dim)

def average_ranking(ranking_kb, ranking_dim):
    # Get the unique values in both lists using set() function
    unique_values = set(ranking_kb) | set(ranking_dim)

    # Create a new list with the average order
    new_list = [value for value in ranking_kb if value in unique_values]
    new_list += [value for value in ranking_dim if value not in new_list]

    # Print the new list with the average order
    return new_list




#delete json report once the upload or new upload is pressed
@app.route('/delete_file', methods=['POST'])
def delete_file():
       
    # Create a relative file path using os.path.join()
    file_path = os.path.join(HERE, 'report/profile.json')

    # Delete the file
    os.remove(file_path)
    
    # Return a JSON response with a success message
    response = jsonify({'message': 'File deleted successfully'})
    return response
    

# App main route + generic routing
@app.route('/', methods=['GET'])
def home():
    return render_template('home/index.html')

@app.route('/upload.html', methods=['GET'])
def upload():
    return render_template('home/upload.html')

@app.route('/audit', methods=['POST'])
def upload_file():
    referer = request.headers.get('referer')
    redirect_url = referer or url_for('home/index.html')

    file_ = request.files.get('file')

    if not file_ or not file_.filename:
        flash('Please select a file', 'warning')
        return redirect(redirect_url)

    (name, ext) = os.path.splitext(file_.filename)
    if not ext.lower() == '.csv':
        flash('Bad file type – CSV required', 'warning')
        return redirect(redirect_url)

    dirpath = tempfile.mkdtemp(prefix='')
    filename = secure_filename(file_.filename)
    file_.save(os.path.join(dirpath, filename))
    if request.method == 'POST':
        return redirect(url_for('audit_file',
                            dirname=os.path.basename(dirpath),
                            name=name))



#the user can also select one of the existing datasets
@app.route('/audit/<name>/', methods=['GET'])
def upload_sample(name):
    if name not in SAMPLE_DATA:
        abort(404)

    source_path = SAMPLE_DATA[name]
    filename = os.path.basename(source_path)
    (name, _ext) = os.path.splitext(filename)
    dirpath = tempfile.mkdtemp(prefix='')
    dest_path = os.path.join(dirpath, filename)
    os.symlink(source_path, dest_path) 
    return redirect(url_for('audit_file',
                            dirname=os.path.basename(dirpath),
                            name=name))

# Function to append dataframe to the global list
def append_dataframe(df):
    global dataframes
    dataframes.append(df)


@app.route('/audit/<name>/<dirname>/', methods=['GET', 'POST'])
def audit_file(name, dirname):
    (df, columns) = get_data(name, dirname)
    session['name'] = name
    session['dirname'] = dirname

    selected_attributes = []
       
    algorithm = request.form.get("algorithm")
    session['algorithm'] = algorithm

    for col in columns:
        if col == request.form.get(col):
            selected_attributes.append(col)

     # store selected_attributes in session
    session['selected_attributes'] = selected_attributes
    

    if request.form.get('Support'):
        support = request.form.get('Support')
    else: support = 80
    session['support'] = support
    
    if request.form.get('Confidence'):
        confidence = request.form.get('Confidence')
    else: confidence = 90
    session['confidence'] = confidence

    
    
    df = df[:][selected_attributes]
    dataFrame= save_data(df)
    append_dataframe(df)
    
    # Append the modified DataFrame to the session list
    # df_copy = df.copy()
    # session['df_list'] = []
    # session['df_list'].append(df_copy.to_dict(orient='records'))
    
    if str(request.form.get("submit")) == "Profiling":
        profile = ProfileReport(df)
        # i=len(session['df_list'])
        i=len(dataframes)
        profile.to_file(f'apps/report/profile_{i}.json')
        # profile.to_file('apps/report/profile.json')

        return redirect(url_for("data_profiling",
                        name=name,
                        dirname=dirname,
                        algorithm=algorithm,
                    
                        support=support,
                        confidence=confidence
                            )
                        )

    else:

        return render_template("home/audit.html",
                                name=name,
                                dirname=dirname,
                                columns=columns
                                )
    

# @app.route("/update_order", methods=["POST"])
# def update_order():
#     order1 = request.form.getlist("order1[]")
#     order2 = request.form.getlist("order2[]")
    
#     # Update order in the backend as desired
    
#     return jsonify({"success": True})

# @app.route('/update_data', methods=['POST'])
# def update_data():
#     if request.form.get('Submit Outliers'):
#         min_max_values = request.form.getlist('values')  # get the list of 'min' values from the form data
    
#     min_values = [min_max_values[0] for min_max_values in min_max_values]
#     max_values = [min_max_values[1] for min_max_values in min_max_values]
#     # update the data in your application's database or data structure using the min_values and max_values
#     return min_values, max_values

 # if str(request.form.get("submit")) == "Submit Outliers":
    #         print("success")
    #         j=0
    #         for col in typeNUMlist:
    #             if request.form.get('min_' + col):
    #                 min_values[j] = request.form.get('min_' + col)
    #             else: min_values[j] = minValueList[j]
    #             if request.form.get('max_' + col):
    #                 max_values[j] = request.form.get('max_' + col)
    #             else: max_values[j] = maxValueList[j]
    #             j=+1
    
# @app.route('/submit-outliers', methods=['POST'])
# def submit_outliers():
#   min_values = {}
#   max_values = {}
#   for key, value in request.form.items():
#     if key.startswith('min_'):
#       min_values[key[4:]] = value
#     elif key.startswith('max_'):
#       max_values[key[4:]] = value
#   # Process the form data here
#   # Return a response
#   session['min_values'] = min_values
#   session['max_values'] = max_values
#   return jsonify({'status': 'success'})

@app.route('/submit_outliers', methods=['POST'])
def submit_outliers():
    form_data = request.get_json() # read the form data from the AJAX request
    # do some processing with the form data (e.g., detect outliers)
    response_data = {'message': 'Outliers detected.'} # create a response message
    print(form_data)
    min_values = []
    max_values = []
    min_values = [float(v) for k, v in form_data.items() if k.startswith('min_')]
    max_values = [float(v) for k, v in form_data.items() if k.startswith('max_')]
   
  # Process the form data here
  # Return a response
    session['min_values'] = min_values
    session['max_values'] = max_values
    
    print(session['min_values'])
    print(session['max_values'])
    return jsonify(response_data) # send the response back to the JavaScript frontend


def get_techniques():
    return [
        {"id": "remove_duplicates", "name": "",  "text": "Remove duplicates", "dimension":"UNIQUENESS" },
        {"id": "impute_standard", "name": "Standard",  "text": "Impute missing values (0/Missing)", "dimension":"COMPLETENESS"},
        {"id": "drop_cols", "name": "", "text": "Drop columns with missing values", "dimension":"COMPLETENESS"},
        {"id": "drop_rows", "name": "", "text": "Drop rows with missing values", "dimension":"COMPLETENESS"},
        {"id": "impute_mean", "name": "Mean", "text": "Impute missing values (mean/mode)", "dimension":"COMPLETENESS"},
        {"id": "impute_std", "name": "Std", "text": "Impute missing values (standard deviation/mode)", "dimension":"COMPLETENESS"},
        {"id": "impute_mode", "name": "Mode", "text": "Impute missing values (mode)", "dimension":"COMPLETENESS"},
        {"id": "impute_median", "name": "Median", "text": "Impute missing values (median/mode)", "dimension":"COMPLETENESS"},
        {"id": "impute_knn", "name": "KNN", "text": "Impute missing values of numerical variables using KNN", "dimension":"COMPLETENESS"},
        {"id": "impute_knn_cat", "name": "KNN", "text": "Impute missing values of categorical variables using KNN", "dimension":"COMPLETENESS"},
        {"id": "impute_mice", "name": "Mice", "text": "Impute missing values of numerical variables using Mice", "dimension":"COMPLETENESS"},
        {"id": "impute_mice_cat", "name": "Mice", "text": "Impute missing values of categorical variables using Mice", "dimension":"COMPLETENESS"},
        
        {"id": "outlier_correction", "name": "", "text": "Outlier correction", "dimension":"ACCURACY"},
        {"id": "oc_impute_standard", "name": "Standard", "text": "Outlier correction with imputation ((0/Missing)", "dimension":"ACCURACY"},
        {"id": "oc_drop_cols", "name": "", "text": "Outlier correction with drop columns", "dimension":"ACCURACY"},
        {"id": "oc_drop_rows", "name": "", "text": "Outlier correction with drop rows", "dimension":"ACCURACY"},
        {"id": "oc_impute_mean", "name": "Mean", "text": "Outlier correction with imputation (mean/mode)", "dimension":"ACCURACY"},
        {"id": "oc_impute_std", "name": "Std", "text": "Outlier correction with imputation (standard deviation/mode)", "dimension":"ACCURACY"},
        {"id": "oc_impute_mode", "name": "Mode", "text": "Outlier correction with imputation (mode)", "dimension":"ACCURACY"},
        {"id": "oc_impute_median", "name": "Median", "text": "Outlier correction with imputation (median/mode)", "dimension":"ACCURACY"},
        {"id": "oc_impute_knn", "name": "KNN", "text": "Outlier correction with imputation (KNN)", "dimension":"ACCURACY"},
        {"id": "oc_impute_mice", "name": "Mice", "text": "Outlier correction with imputation (Mice)", "dimension":"ACCURACY"}
    ]





@app.route("/apply", methods=["POST"])
def save_and_apply():
    sorted_list = request.get_json()
    # do something with the sorted list
    print(sorted_list)

    cols = session.get('selected_attributes', [])
    df = last_dataframe()
    
    min_values = session.get('min_values', [])
    max_values = session.get('max_values', [])
    outlier_range = [list(x) for x in zip(min_values, max_values)]
    
    # algorithm = session.get('algorithm', [])
    # support = session.get('support', [])
    # confidence = session.get('confidence', [])
    # name = session.get('name', [])
    # dirname = session.get('dirname', [])

    techniques=get_techniques()

    for tech in sorted_list:
        if tech == "remove_duplicates":
            df = improve.remove_duplicates(df)

        elif tech == "impute_standard":
            impute = imputes.impute_standard()
            df = impute.fit(df)

        elif tech == "drop_cols":
            impute = imputes.drop()
            df = impute.fit_cols(df)
            # store selected_attributes in session
            session['selected_attributes'] = list(df.columns)

        elif tech == "drop_rows":
            impute = imputes.drop()
            df = impute.fit_rows(df)

        elif tech == "impute_mean":
            impute = imputes.impute_mean()
            df = impute.fit_mode(df)

        elif tech == "impute_std":
            impute = imputes.impute_std()
            df = impute.fit_mode(df)

        elif tech == "impute_mode":
            impute = imputes.impute_mode()
            df = impute.fit(df)

        elif tech == "impute_median":
            impute = imputes.impute_median()
            df = impute.fit_mode(df)

        elif tech == "impute_knn":
            impute = imputes.impute_knn()
            df = impute.fit(df)

        elif tech == "impute_knn_cat":
            impute = imputes.impute_knn()
            df = impute.fit_cat(df)

        elif tech == "impute_mice":
            impute = imputes.impute_mice()
            df = impute.fit(df)

        elif tech == "impute_mice_cat":
            impute = imputes.impute_mice()
            df = impute.fit_cat(df)

        elif tech == "outlier_correction":
            df = improve.outlier_correction(df, outlier_range)
            
        elif tech == "oc_impute_standard":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.impute_standard()
            df = impute.fit(df)

        elif tech == "oc_drop_cols":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.drop()
            df = impute.fit_cols(df)
            global column_list
            column_list = list(df.columns)

        elif tech == "oc_drop_rows":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.drop()
            df = impute.fit_rows(df)

        elif tech == "oc_impute_mean":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.impute_mean()
            df = impute.fit_mode(df)

        elif tech == "oc_impute_std":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.impute_std()
            df = impute.fit_mode(df)

        elif tech == "oc_impute_mode":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.impute_mode()
            df = impute.fit(df)

        elif tech == "oc_impute_median":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.impute_median()
            df = impute.fit_mode(df)

        elif tech == "oc_impute_knn":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.impute_knn()
            df = impute.fit(df)

        elif tech == "oc_impute_mice":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.impute_mice()
            df = impute.fit(df)

            
    global dataframes
    dataframes.append(df)
    # Append the modified DataFrame to the session list
    # df_copy = df.copy()
    # print(len(df_copy))
    # session['df_list'] = []
    # session['df_list'].append(df_copy.to_dict(orient='records'))

    save_data(df) #saves the dataframe in file data.csv
  




@app.route('/dataprofiling/<name>/<dirname>/<algorithm>/<support>/<confidence>/', methods=['GET', 'POST'])
def data_profiling(name, dirname, algorithm, support, confidence):
    # access selected_attributes from session
    col_list = session.get('selected_attributes', '0')
    
    # access dataframe from global
    df = last_dataframe()
    
    global column_list
    if column_list[0]==0:
        columns = get_columns(col_list)
    else:
        columns = column_list
        
    # df = pd.read_csv(os.path.join(HERE, DF_NAME))
    
    df = df[:][columns]
    columns_names= list(df.columns)

    global dataframes
    # Load the JSON file into a string variable
    i = len(dataframes)
    with open(f'apps/report/profile_{i}.json', "r") as f:
        json_str = f.read()

    # Parse the JSON string into a dictionary object
    profile = json.loads(json_str)
    typeList =[]
    typeNUMlist =[]
    typeCATlist =[]
    
    for var in profile['variables'].values():
        typeList.append(var['type'])

    for i in range(len(typeList)):
        if typeList[i]=="Numeric":
            typeNUMlist.append(columns_names[i])
        else: typeCATlist.append(columns_names[i])
    

    # typeNUMlist = df.select_dtypes(include=['int64','float64']).columns
    minValueList = []
    maxValueList = []
    for var in columns:
        if var in typeNUMlist:
            minValueList.append(df[var].min())
            maxValueList.append(df[var].max()) 

    #plot generation
    outliers_html_list = myPlots.boxPlot(df, typeNUMlist)  #outlier plots
    myPlots.heatmap(df) #heatmap of the correlation
    distr_html_list = myPlots.distributionPlot(df,typeNUMlist)  #distribution plots(distr_html_list takes the list of html addresses where the plots are saved)
    distrCAT_html_list = myPlots.distributionCategorical(df,typeCATlist)
    #treeMap_html_list= myPlots.treePlot(df, typeCATlist)
    myPlots.missing_data(df) #missigno plots

    min_values = []
    max_values = []

    if 'min_values' not in session:
        for i in range(len(typeNUMlist)):
            min_values.append(minValueList[i])
        session['min_values'] = min_values
        
    if 'max_values' not in session:
        for i in range(len(typeNUMlist)):
            max_values.append(minValueList[i])
        session['max_values'] = max_values
        
    min_values = session.get('min_values', [])
    max_values = session.get('max_values', [])
    print(min_values)
    print(max_values)
    
    #calculate dimensions
    accuracy=quality_dims.accuracy_value(df, profile, columns, typeNUMlist, min_values, max_values)
    uniqueness=quality_dims.uniqueness_value(profile)
    completeness=quality_dims.completeness_value(profile, columns)
    
    #ranking of the dimensions
    ranking_dim = rank_dim(accuracy, uniqueness, completeness)
    #ranking based on the characteristics of the knowledge base
    ranking_kb = rank_kb(df, algorithm)

    average_rank = average_ranking(ranking_kb, ranking_dim)

    df_updates = []
    df_updates.append(df)

    techniques=get_techniques()    


    if str(request.form.get("submit")) == "Upload new dataset":
        return redirect(url_for('upload'))
    
    if str(request.form.get("submit")) == "Modify choices":
        
        return redirect(url_for('audit_file', dirname=os.path.basename(dirname), name=name))
    
    if str(request.form.get("submit")) == "Download your csv file":
        return send_from_directory(directory=HERE + "/datasets", path="data.csv")
    
    if str(request.form.get("submit")) == "Apply modifications":
        return redirect(url_for('apply_modifications', 
                        name=name,
                        dirname=dirname,
                        algorithm=algorithm,
                    
                        support=support,
                        confidence=confidence))
    
    else:
        
        
        
        return render_template("home/profiling.html",
                                name=name,
                                dirname=dirname,
                                columns=columns,
                                algorithm=algorithm,
                                average_rank=average_rank,
                                sample=df.head(20),
                                techniques=techniques,

                                support=support,
                                confidence=confidence,

                                typeList=typeList,
                                typeNUMlist=typeNUMlist,

                                min_values=min_values,
                                max_values=max_values,
                                uniqueness=uniqueness,
                                accuracy=accuracy,
                                completeness=completeness,

                                profile = profile,
                                distr_html_list=distr_html_list,
                                distrCAT_html_list=distrCAT_html_list,
                                outliers_html_list=outliers_html_list,
                                )


@app.route('/apply_modifications/<name>/<dirname>/<algorithm>/<support>/<confidence>/', methods=['GET', 'POST'])
def apply_modifications(name,dirname,algorithm,support,confidence):
        global dataframes
        # start_report_generation()
        df = last_dataframe()
        profile = ProfileReport(df)
        i=len(dataframes)
        profile.to_file(f'apps/report/profile_{i}.json')

        return redirect(url_for("data_profiling",
                        name=name,
                        dirname=dirname,
                        algorithm=algorithm,
                    
                        support=support,
                        confidence=confidence
                            )
                        )



#nav(df,variables=None,Interactions=True, Correlations=None,  Missing_values=None, Sample=None)
#missing values, correlazione tra colonne, overview iniziale, no interazioni
#fase di mezzanea calcola outliers, ispezione functional dependencies e conoscenza aggiuntiva per ouliers (range)








