
import pandas as pd
import numpy as np
import copy

def convert_columns(df, conversion_dict): # modify df in place
    for col_key in conversion_dict.keys():
        key_dict = conversion_dict[col_key]
        df[col_key] = df[col_key].apply(lambda x: key_dict[x])

all_file = 'all-xy-charge-aug2.csv'

all_df = pd.read_csv(all_file)


column_list = [
    'id', 'sex', 'race', 'age', 'age_cat', 'juv_fel_count',
    'decile_score', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_days_from_compas',
    'decile_score.1', 'score_text', 'v_decile_score', 'v_score_text', 'priors_count.1',
    'c_charge_degree', 'days_b_screening_arrest', 'length_of_stay', 'y', 'sensitive', 'y_charge_mean',
    'charge_id_0', 'charge_id_1', 'charge_id_2', 'charge_id_3', 'charge_id_4'
]

Xy_df = all_df[column_list]
Xy_dfcopy = copy.deepcopy(Xy_df)

Xy_dfcopy['BlackLatinoMale'] = np.where(np.logical_and(np.logical_or(Xy_dfcopy['race'] == 0, Xy_dfcopy['race'] == 3), Xy_dfcopy['sex'] == 0), 1, 0)

Xy_dfcopy['African-American'] = np.where(Xy_dfcopy['race'] == 0, 1, 0)
Xy_dfcopy['Hispanic'] = np.where(Xy_dfcopy['race'] == 3, 1, 0)
Xy_dfcopy['Caucasian'] = np.where(Xy_dfcopy['race'] == 2, 1, 0)
Xy_dfcopy['Native American'] = np.where(Xy_dfcopy['race'] == 4, 1, 0)
Xy_dfcopy['Asian'] = np.where(Xy_dfcopy['race'] == 1, 1, 0)
Xy_dfcopy['Other'] = np.where(Xy_dfcopy['race'] == 5, 1, 0)

Xy_dfcopy.to_csv('all-xy-with-aug-softmax.csv', index=False)