import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch

UCI_datafolder = "../data/UCI/"

def dummy_encode(df):
    """
   Auto encodes any dataframe column of type category or object.
   """

    columnsToEncode = list(df.select_dtypes(include=['category', 'object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding '+feature)
    return df

def normalize_col(s):
    std = s.std()
    mean = s.mean()
    if std > 0:
        return (s - mean) / std
    else:
        return s - mean

def normalize_cols(df, columns=None):
    if columns is None:
        columns = df.columns
    for col in columns:
        df[col] = normalize_col(df[col])
    return df
    
def reg_to_class(s):
    return (s > s.mean()).astype(int)


def mul_to_bin(s, border=None):
    if border is None:
        border = s.median()
    return (s > border).astype(int)


def uci_data(data_mode):

    if data_mode == 'bank':
        df = pd.read_csv(f'{UCI_datafolder}/bank//bank-full.csv', sep=';')
        df['balance'] = normalize_col(df['balance'])
        df = dummy_encode(df)
        df.rename(columns={'y': 'target'}, inplace=True)

    elif data_mode == 'concrete':
        df = pd.read_excel(f'{UCI_datafolder}/concrete//Concrete_Data.xls')
        df = normalize_cols(df)
        df.rename(columns={'Concrete compressive strength(MPa, megapascals) ': 'target'}, inplace=True)
        df['target'] = reg_to_class(df['target'])

    elif data_mode == 'housing':
        df = pd.read_fwf(f'{UCI_datafolder}/housing//housing.data.txt', header=None)
        df = normalize_cols(df)
        df.rename(columns={13: 'target'}, inplace=True)
        df['target'] = reg_to_class(df['target'])

    elif data_mode == 'landsat':
        df = pd.read_csv(f'{UCI_datafolder}/landsat//sat.trn.txt', header=None, sep=' ')
        df = pd.concat([df, pd.read_csv(f'{UCI_datafolder}/landsat//sat.tst.txt', header=None, sep=' ')])
        df = normalize_cols(df, columns=[x for x in range(36)])
        df.rename(columns={36: 'target'}, inplace=True)
        df['target'] = mul_to_bin(df['target'])

    elif data_mode == 'mushroom':
        df = pd.read_csv(f'{UCI_datafolder}/mushroom//agaricus-lepiota.data.txt', header=None)
        df = dummy_encode(df)
        df.rename(columns={0: 'target'}, inplace=True)

    elif data_mode == 'pageblock':
        df = pd.read_fwf(f'{UCI_datafolder}/pageblock//page-blocks.data', header=None)
        df = normalize_cols(df, columns=[x for x in range(10)])
        df.rename(columns={10: 'target'}, inplace=True)
        df['target'] = mul_to_bin(df['target'], 1)

    elif data_mode == 'shuttle':
        df = pd.read_csv(f'{UCI_datafolder}/shuttle//shuttle.trn', header=None, sep=' ')
        df = pd.concat([df, pd.read_csv(f'{UCI_datafolder}/shuttle//shuttle.tst.txt', header=None, sep=' ')])
        df = normalize_cols(df, columns=[x for x in range(9)])
        df.rename(columns={9: 'target'}, inplace=True)
        df['target'] = mul_to_bin(df['target'], 1)

    elif data_mode == 'spambase':
        df = pd.read_csv(f'{UCI_datafolder}/spambase//spambase.data.txt', header=None, sep=',')
        df = normalize_cols(df, columns=[x for x in range(57)])
        df.rename(columns={57: 'target'}, inplace=True)

    elif data_mode == 'wine':
        df = pd.read_csv(f'{UCI_datafolder}/wine//winequality-red.csv', sep=';')
        df_w = pd.read_csv(f'{UCI_datafolder}/wine//winequality-white.csv', sep=';')
        df['target'] = 1
        df_w['target'] = 0
        df = pd.concat([df, df_w])
        df = normalize_cols(df, [x for x in df.columns if x != 'target'])

    df_neg = df[df['target'] == 0]
    n_data = df_neg.drop(['target'], axis=1).values
    n_shuffle = np.random.permutation(len(n_data))
    n_data = n_data[n_shuffle]

    df_pos = df[df['target'] == 1]
    p_data = df_pos.drop(['target'], axis=1).values
    p_shuffle = np.random.permutation(len(p_data))
    p_data = p_data[p_shuffle]

    return p_data, n_data 


class UCI_data(torch.utils.data.Dataset):
    def __init__(self, p_data, n_data, train=True):

        if train:
            self.p_data = p_data[ :len(p_data)*2//3].astype(np.float32)
            self.n_data = n_data[ :len(n_data)*2//3].astype(np.float32)
        else:  
            self.p_data = p_data[len(p_data)*2//3:].astype(np.float32)
            self.n_data = n_data[len(n_data)*2//3:].astype(np.float32)

        self.transform = None
        self.target_transform = None

    def __len__(self): 
        return len(self.n_data) + len(self.p_data)
 