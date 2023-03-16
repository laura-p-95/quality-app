import numpy as np
import pandas as pd
import apps.scripts.kb_test as kb

def accuracy_value(df, profile, columns, typeNUMlist, min_values, max_values):
    # Syntactic Accuracy: Number of correct values/total number of values
    i=0
    correct_values_tot=0
    tot_n_values = profile['table']['n']*len(columns)
    for var in columns:
        if var in typeNUMlist:
            range_correct_values = np.arange(float(min_values[i]),float(max_values[i])) #start value, stop value and step value. I choose a small step as we do not know the real content of the columns
            correct_values_i=sum(1 for item in df[var] if item in range_correct_values)
            correct_values_tot = correct_values_tot + correct_values_i
            i+=1
    accuracy = correct_values_tot / tot_n_values * 100
    return accuracy

def uniqueness_value(profile):
    # Uniqueness = percentage calculated as Cardinality (count of the number 
    # of distinct actual values) divided by the total number of records.
    n_tot_distinct=0
    tot_n_values=0

    for var in profile['variables'].values():
        n_tot_distinct+=var['n_distinct']
        tot_n_values+=var['n']

    uniqueness=n_tot_distinct/tot_n_values*100
    return uniqueness

def completeness_value(profile, columns):
    # Completeness: Number of not null values/total number of values
    tot_n_values = profile['table']['n']*len(columns)
    tot_not_null = tot_n_values - profile['table']['n_cells_missing']
    completeness = tot_not_null / tot_n_values *100
    return completeness

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