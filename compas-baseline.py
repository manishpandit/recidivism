# compute compas-only baseline
import pandas as pd
import numpy as np

def convert_columns(df, conversion_dict): # modify df in place
    for col_key in conversion_dict.keys():
        key_dict = conversion_dict[col_key]
        df[col_key] = df[col_key].apply(lambda x: key_dict[x])

violent_file = 'compas-scores-two-years-violent.csv'
all_file = 'compas-scores-two-years.csv'

all_df = pd.read_csv(all_file)

y = np.ndarray.astype(all_df.values[:,-1],int)

column_list = [
    'id', 'sex','age','age_cat',
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
    'score_text': { 'High': 1, 'Low': -1, 'Medium': 0 },
    'v_score_text': { 'High': 1, 'Low': -1, 'Medium': 0 }
}

Xy_df = all_df[column_list]
convert_columns(Xy_df, conversion_dict)
Xy_df.insert(Xy_df.shape[1], 'y', y)
Xy_df.to_csv('all-xy.csv', index=False)

