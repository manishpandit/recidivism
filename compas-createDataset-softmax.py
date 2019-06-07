
# compute compas-only baseline
import pandas as pd
import numpy as np
import copy

def convert_columns(df, conversion_dict): # modify df in place
    for col_key in conversion_dict.keys():
        key_dict = conversion_dict[col_key]
        df[col_key] = df[col_key].apply(lambda x: key_dict[x])

all_file = 'compas-scores-two-years.csv'

all_df = pd.read_csv(all_file)

y = np.ndarray.astype(all_df.values[:,-1],int)

column_list = [
    'id', 'sex','age','age_cat', 'race',
    'juv_fel_count','decile_score','juv_misd_count','juv_other_count',
    'priors_count',
    'decile_score.1', 'score_text', 'v_decile_score', 'v_score_text',
    'priors_count.1'
]
# todo: add categorical variables (race). consider specific analysis for crime category / crime text
# also should think about meaning of columns with NaN
conversion_dict = {
    'sex': { 'Female': 1, 'Male': 0 },
    'age_cat': { 'Less than 25': -1, '25 - 45': 0, 'Greater than 45': 1},
    'race': {'Other': 5, 'African-American': 0, 'Hispanic': 1, 'Native American': 3, 'Asian': 4, 'Caucasian': 2},
    'score_text': { 'High': 1, 'Low': -1, 'Medium': 0 },
    'v_score_text': { 'High': 1, 'Low': -1, 'Medium': 0 }
}

Xy_df = all_df[column_list]
Xy_dfcopy = copy.deepcopy(Xy_df)
convert_columns(Xy_dfcopy, conversion_dict)

Xy_dfcopy['BlackLatinoMale'] = np.where(np.logical_and(Xy_dfcopy['race'] <= 1, Xy_dfcopy['sex'] == 0), 1, 0)

Xy_dfcopy['African-American'] = np.where(Xy_dfcopy['race'] == 0, 1, 0)
Xy_dfcopy['Hispanic'] = np.where(Xy_dfcopy['race'] == 1, 1, 0)
Xy_dfcopy['Caucasian'] = np.where(Xy_dfcopy['race'] == 2, 1, 0)
Xy_dfcopy['Native American'] = np.where(Xy_dfcopy['race'] == 3, 1, 0)
Xy_dfcopy['Asian'] = np.where(Xy_dfcopy['race'] == 4, 1, 0)
Xy_dfcopy['Other'] = np.where(Xy_dfcopy['race'] == 5, 1, 0)

Xy_dfcopy.insert(Xy_df.shape[1], 'y', y)
Xy_dfcopy.to_csv('all-xy-with-softmax.csv', index=False)