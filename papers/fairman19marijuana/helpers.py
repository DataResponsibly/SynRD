import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import glob
import json
import os
import statsmodels.api as st
from statsmodels.discrete.discrete_model import MNLogit
import warnings


def plot_figure(df_):
    cols = ['MARIJUANA', 'CIGARETTES', 'ALCOHOL', 'OTHER_TABACCO', 'OTHER_DRUGS', 'NO_DRUG_USE']
    prop_sex = pd.crosstab(index=df_['SEX'], columns=df_['CLASS'], normalize="index") * 100
    prop_race = pd.crosstab(index=df_['RACE'], columns=df_['CLASS'], normalize="index") * 100
    prop_age = pd.crosstab(index=df_['AGE_GROUP'], columns=df_['CLASS'], normalize="index") * 100
    prop_year = pd.crosstab(index=df_['YEAR'], columns=df_['CLASS'], normalize="index") * 100
    figure, axis = plt.subplots(1, 4, gridspec_kw={'width_ratios': [1, 2, 3, 4]})
    figure.set_size_inches(15, 6, forward=True)
    figure.tight_layout() 
    prop_sex[cols].plot(kind='bar', stacked=True, colormap='tab20', ax=axis[0], legend=None, xlabel='Sex')
    prop_age[cols].plot(kind='bar', stacked=True, colormap='tab20', ax=axis[1], legend=None, xlabel='Age Group')
    prop_race[cols].plot(kind='bar', stacked=True, colormap='tab20', ax=axis[2], legend=None, xlabel='Race/Ethnicity')
    prop_year[cols].plot(kind='bar', stacked=True, colormap='tab20', ax=axis[3], legend=None, xlabel='Survey Year')
    axis[0].yaxis.set_major_formatter('{x:1.0f}%')
    axis[1].set_yticks([])
    axis[2].set_yticks([])
    axis[3].set_yticks([])
    axis[0].xaxis.set_label_coords(.5, -.2)
    axis[1].xaxis.set_label_coords(.5, -.2)
    axis[2].xaxis.set_label_coords(.5, -.2)
    axis[3].xaxis.set_label_coords(.5, -.2)
    plt.subplots_adjust(wspace=0.05)
    plt.legend(loc='upper center', bbox_to_anchor=(-0.25, 1.1),  ncol = 6)
    # plt.savefig('figure_1.png', facecolor='white', transparent=False, bbox_inches='tight', pad_inches=.5)
    plt.show()
    
def get_encoded_df(df):
    df['SEX'] = df['SEX'].astype('category')
    df['RACE'] = df['RACE'].astype('category')
    df['AGE_GROUP'] = df['AGE_GROUP'].astype('category')
    df['AGE'] = df['AGE'].astype('category')
    df['CLASS'] = df['CLASS'].astype('category')
    df['YEAR'] = df['YEAR'].astype('category')
    categorical = df.select_dtypes(['category']).columns
    df_cat = df[categorical].apply(lambda x: x.cat.codes)
    return df_cat

def get_unencoded_df(df, df_cat):
    df_noncat = pd.DataFrame(columns=df.columns)
    categorical = df.select_dtypes(['category']).columns
    for cat in categorical:
        mapping = dict(enumerate(df[cat].cat.categories))
        df_noncat[cat] = df_cat[cat].map(mapping)
    return df_noncat

def load_data(base_dir='../..', file_name='data/nsduh_processed_data.csv'):
    return pd.read_csv(os.path.join(base_dir, file_name), index_col=0)

def get_log_reg(df):
    class_mapping = {
        'MARIJUANA': 0,
        'ALCOHOL': 1,
        'CIGARETTES': 2,
        'NO_DRUG_USE': 3,
        'OTHER_TABACCO': 4,
        'OTHER_DRUGS': 5
    }
    data = df.copy()
    data['CLASS'] = data.CLASS.map(class_mapping)
    data['CLASS'] = data.CLASS.astype(np.int32)
    data['AGE'] = data.AGE.astype(np.int32)
    data['YEAR'] = data.YEAR.astype(np.int32)
    model = st.MNLogit.from_formula(
        'CLASS ~ C(SEX, Treatment(reference="Male")) + AGE + C(RACE, Treatment(reference="White")) + YEAR',
        data=data)
    log_reg = model.fit(method='newton', maxiter=100, full_output=True)
    return log_reg

def get_aOR(df):
    log_reg = get_log_reg(df)
    dfs = []
    lower_CI = log_reg.conf_int()['lower']
    upper_CI = log_reg.conf_int()['upper']
    for indx in [0, 1, 2]:
        dfs.append(
            pd.DataFrame(
                {
                    "aOR": np.exp(log_reg.params[indx]),
                    "Lower CI": np.exp(lower_CI.iloc[lower_CI.index.get_level_values('CLASS') == str(indx + 1)].values),
                    "Higher CI": np.exp(upper_CI.iloc[upper_CI.index.get_level_values('CLASS') == str(indx + 1)].values),
                    "pvalue": log_reg.pvalues[indx],
                }
            )
        )

    aOR = pd.concat(dfs, axis=1)
    aOR = aOR.drop('Intercept')
    aOR.index = [x[-1].replace(')', '').replace('T.', '').replace('"', '').replace('[', ' vs. ').replace(']', '') for x in aOR.index.str.split('reference=')]
    aOR = round(aOR, 3)
    return aOR
