import logging
import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.tree import DecisionTreeClassifier


def log_b(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, force=True)
    file = logging.FileHandler(os.path.join(dir, "log.txt"))
    logging.getLogger().addHandler(file)


Operations = {
    "doubles": ["add", "subtract", "multiply", "divide"],
    "single": ["rabs", 'square', 'reciprocal', 'log', 'sqrt', 'power3', 'None'],
    "special_generation": ["reserve", "replace", "delete"],
    "discrete": ["cross", 'nunique']
}


def get_action(n_features_c, n_features_d):
    c_generation = []
    d_generation = []
    sps = []
    doubles = Operations["doubles"]
    single_c = Operations["single"]
    specials = Operations["special_generation"]
    discrete = Operations["discrete"]
    if n_features_c == 0:
        c_generation = []
    elif n_features_c == 1:
        c_generation.extend(single_c)
    else:
        for i in range(4):
            op = doubles[i]
            for j in range(n_features_c):
                c_generation.append(op)
        c_generation.extend(single_c)
    if n_features_d != 0:
        for i in range(len(discrete)):
            op = discrete[i]
            for j in range(n_features_d):
                d_generation.append(op)
    else:
        d_generation = []

    max_len = max(len(c_generation), len(d_generation))
    if len(d_generation) < max_len and len(d_generation) != 0:
        d_ori_generation = d_generation
        while len(d_generation) < max_len:
            d_generation.extend(d_ori_generation)
        d_generation = d_generation[:max_len]
    if len(c_generation) < max_len and len(c_generation) != 0:
        c_ori_generation = c_generation
        while len(c_generation) < max_len:
            c_generation.extend(c_ori_generation)
        c_generation = c_generation[:max_len]

    while len(sps) < max_len:
        sps.extend(specials)

    sps = sps[:max_len]

    return c_generation, d_generation, sps


def get_binning_df(args, df, v_columns, d_columns, type):
    if df.shape[1] > 1000:

        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        selector = SelectKBest(score_func=mutual_info_regression, k=100)
        X_new = selector.fit_transform(X, y)
        new_df = pd.concat([X_new,y],axis=1)
        new_v_columns = list(X_new.columns)
        new_d_columns = []
    else:
        new_df = pd.DataFrame()
        new_v_columns = []
        new_d_columns = []
        label = df.loc[:, args.target]
        if type == 'classify':
            for col in v_columns:
                new_df[col] = df[col]
                new_v_columns.append(col)
            for col in v_columns:
                ori_fe = np.array(df[col])
                label = np.array(label)
                new_fe = binning(ori_fe, label)
                new_name = 'bin_' + col
                new_df[new_name] = new_fe
                new_d_columns.append(new_name)
            for col in d_columns:
                new_df[col] = df[col]
                new_d_columns.append(col)

        else:
            for col in v_columns:
                new_df[col] = df[col]
                new_v_columns.append(col)
            for col in d_columns:
                new_df[col] = df[col]
                new_d_columns.append(col)
        new_df[args.target] = label
    return new_df, new_v_columns, new_d_columns


def get_actions(actions_generation, actions_discrimination, c_generation, d_generation, df_c, df_d, n_c_features, n_d_features):
    operations_c = len(Operations["doubles"]) * n_c_features + len(Operations["single"])
    operations_d = len(Operations["discrete"]) * n_d_features
    add = []
    subtract = []
    multiply = []
    divide = []
    cross = []
    nunique = []
    singles = {}
    len_c = n_c_features
    for index, (generation, opt) in enumerate(zip(actions_generation, actions_discrimination)):
        real_discrimination = opt % 3
        if 0 <= index < len_c:
            if operations_c <= 7:
                break
            generation = generation % operations_c
            if 0 <= generation < n_c_features:
                add.append([index, generation, real_discrimination])
            elif n_c_features <= generation < (2 * n_c_features):
                subtract.append([index, generation - n_c_features, real_discrimination])
            elif (2 * n_c_features) <= generation < (3 * n_c_features):
                multiply.append([index, generation - n_c_features * 2, real_discrimination])
            elif (3 * n_c_features) <= generation < (4 * n_c_features):
                divide.append([index, generation - n_c_features * 3, real_discrimination])
            else:
                singles[index] = [generation - n_c_features * 4, real_discrimination]
        else:
            if operations_d <= 0:
                break
            x = generation % operations_d
            if 0 <= x < n_d_features:
                cross.append([index - len_c, x, real_discrimination])
            elif n_d_features <= x < 2 * n_d_features:
                nunique.append([index - len_c, x - n_d_features, real_discrimination])
            else:
                cross.append([index - len_c, 'None', real_discrimination])

    actions = {"add": add,"subtract": subtract, "multiply": multiply, "divide": divide, "cross": cross,
               "singles": singles, "nunique": nunique}
    return actions


def get_pos_emb(input_data, con_or_dis):
    position = np.array(con_or_dis).reshape(-1, 1)
    div_term = 10 * np.exp(np.arange(0, 128, 1) * -(np.log(10.0) / 128))
    pos_encoding = np.sin(position * div_term) / 10
    return pos_encoding


def binning(ori_fe, label):
    boundaries = []
    clf = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=6, min_samples_leaf=0.05)
    fe = ori_fe.reshape(-1, 1)
    clf.fit(fe, label.astype("int"))
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold
    for i in range(n_nodes):
        if children_left[i] != children_right[i]:
            boundaries.append(threshold[i])
    boundaries.sort()

    def assign_bin(value, boundaries):
        for i, boundary in enumerate(boundaries):
            if value <= boundary:
                return i
        return len(boundaries)

    if boundaries:
        new_fe = np.array([assign_bin(x, boundaries) for x in ori_fe])
    else:
        new_fe = ori_fe
    return new_fe


datainfos = {
    "ionosphere": {'type': 'classify',
                   'v_columns': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
                                 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23',
                                 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32'],
                   'd_columns': ['D1', 'D2'],
                   'target': 'label',
                   },
    "svmguide3": {'type': 'classify',
                  'v_columns': ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12',
                                'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19'],
                  'd_columns': ['D1'],
                  'target': 'target',
                  },
    "messidor_features": {'type': 'classify',
                          'v_columns': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
                                        'C13', 'C14', 'C15', 'C16'],
                          'd_columns': ['D1', 'D2', 'D3'],
                          'target': 'label',
                          },
    "PimaIndian": {'type': 'classify',
                   'v_columns': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
                   'd_columns': [],
                   'target': 'label',
                   },
    'SPECTF': {'type': 'classify',
               'v_columns': ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13',
                             'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26',
                             'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39',
                             'V40', 'V41', 'V42', 'V43'],
               'd_columns': [],
               'target': 'label',
               },
    'megawatt1': {'type': 'classify',
                  'v_columns': ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                                'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                                'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29',
                                'V30', 'V31', 'V32', 'V33', 'V34'],
                  'd_columns': ['D1', 'D2', 'D3'],
                  'target': 'def',
                  },
    'german_credit': {'type': 'classify',
                         'v_columns': ['C0', 'C1', 'C2'],
                         'd_columns': ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12',
                                       'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20'],
                         'target': 'label',

                         },
    'Bikeshare_DC': {'type': 'regression',
                     'v_columns': ['temp', 'atemp', 'humidity', 'windspeed', 'casual',
                                   'registered'],
                     'd_columns': ['season', 'holiday', 'weekday', 'workingday', 'weather'],
                     'target': 'count',
                     },
    'airfoil': {'type': 'regression',
                'v_columns': ['V0', 'V1', 'V2', 'V3', 'V4'],
                'd_columns': [],
                'target': 'label',
                },
    'Housing_Boston': {'type': 'regression',
                       'v_columns': ['V0', 'V1', 'V2', 'V4', 'V5', 'V6',
                                     'V7', 'V8', 'V9', 'V10', 'V11', 'V12'],
                       'd_columns': ['V3'],
                       'target': 'label',
                       },
    'Openml_586': {'type': 'regression',
                   'v_columns': ['oz1', 'oz2', 'oz3', 'oz4', 'oz5', 'oz6', 'oz7', 'oz8', 'oz9', 'oz10',
                                 'oz11', 'oz12', 'oz13', 'oz14', 'oz15', 'oz16', 'oz17', 'oz18', 'oz19',
                                 'oz20', 'oz21', 'oz22', 'oz23', 'oz24', 'oz25'],
                   'd_columns': [],
                   'target': 'oz26',
                   },
    "ilpd": {"type": "classify",
             "v_columns": ['V1', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10'],
             "d_columns": ['V2'],
             "target": "label",
             },
    'Openml_592': {'type': 'regression',
                   'v_columns': ['oz1', 'oz2', 'oz3', 'oz4', 'oz5', 'oz6', 'oz7', 'oz8', 'oz9',
                                 'oz10', 'oz11', 'oz12', 'oz13', 'oz14', 'oz15', 'oz16', 'oz17',
                                 'oz18', 'oz19', 'oz20', 'oz21', 'oz22', 'oz23', 'oz24', 'oz25',],
                   'd_columns': [],
                   'target': 'label',
                   },
    'Openml_584': {'type': 'regression',
                   'v_columns': ['oz1', 'oz2', 'oz3', 'oz4', 'oz5', 'oz6', 'oz7', 'oz8', 'oz9',
                                 'oz10', 'oz11', 'oz12', 'oz13', 'oz14', 'oz15', 'oz16', 'oz17',
                                 'oz18', 'oz19', 'oz20', 'oz21', 'oz22', 'oz23', 'oz24', 'oz25',],
                   'd_columns': [],
                   'target': 'label',
                   },
    'Openml_599': {'type': 'regression',
                   'v_columns': ['oz1', 'oz2', 'oz3', 'oz4', 'oz5'],
                   'd_columns': [],
                   'target': 'label',
                   },
}
