import numpy as np

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