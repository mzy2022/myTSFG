import logging

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer,  mutual_info_score
from sklearn.model_selection import cross_val_score
from process_data.first_pro_data import ProcessData
from utils import get_actions
from process_data.update_data import Update_data


def getBatches(args, dqn_generation, dqn_discrimination, data_infomation, df_c, df_d, df_label, c_generation, d_generation, i, batchs, device,
           steps_done):
    states = []
    processData_list = []
    con_or_diss = []
    ori_x_c_ds = []
    n_c_features = df_c.shape[1] - 1
    n_d_features = df_d.shape[1] - 1
    df_encode = pd.concat(
        [df_c.drop(df_c.columns[-1], axis=1), df_d.drop(df_d.columns[-1], axis=1),
         df_label], axis=1)
    ori_x_c_d = pd.concat(
        [df_c.drop(df_c.columns[-1], axis=1), df_d.drop(df_d.columns[-1], axis=1)], axis=1)
    init_state = torch.from_numpy(df_encode.values).float().transpose(0, 1).to(device)
    init_con_or_dis = [1] * len(data_infomation['continuous_columns']) + [-1] * len(data_infomation['discrete_columns']) + [0]
    for_next = False
    steps_done += 1
    if i == 0:
        for _ in range(args.episodes):
            states.append(init_state)
            processData_list.append(ProcessData(data_infomation))
            con_or_diss.append(init_con_or_dis)
            ori_x_c_ds.append(ori_x_c_d)
    else:
        for j in range(args.episodes):
            states.append(batchs[j].c_d)
            processData_list.append(Update_data(batchs[j].features_c, batchs[j].features_d, data_infomation))
            con_or_diss.append(batchs[j].con_or_dis)
            ori_x_c_ds.append(batchs[j].x_c_d)

    for i in range(args.episodes):
        x_c = pd.DataFrame()
        x_d = pd.DataFrame()
        actions_generation, states_generation = dqn_generation.choose_action_generation(states[i], for_next, steps_done, con_or_diss[i])
        actions_discrimination, states_discrimination = dqn_discrimination.choose_action_discrimination(actions_generation, states_generation, for_next, steps_done)
        op_list = get_actions(actions_generation, actions_discrimination, c_generation, d_generation, df_c, df_d, n_c_features, n_d_features)
        df_c, df_d, specials = processData_list[i].process_data(op_list)
        if df_c.shape[0] != 0:
            df_c = pd.concat([df_c, df_label], axis=1)
            x_c = df_c.iloc[:, :-1].astype(np.float32).apply(np.nan_to_num)
        if df_d.shape[0] != 0:
            df_d = pd.concat([df_d, df_label], axis=1)
            x_d = df_d.iloc[:, :-1].astype(np.float32).apply(np.nan_to_num)

        if x_c.shape[0] == 0:
            data_x = np.hstack((x_d, df_label.values.reshape(-1, 1)))
            x_c_d = x_d
        elif x_d.shape[0] == 0:
            data_x = np.hstack((x_c, df_label.values.reshape(-1, 1)))
            x_c_d = x_c
        else:
            data_x = np.hstack((x_c, x_d, df_label.values.reshape(-1, 1)))
            x_c_d = pd.concat([x_c, x_d], axis=1)
        data_x = torch.from_numpy(data_x).float().transpose(0, 1).to(device)
        con_or_dis = get_con_or_dis(x_c_d, data_infomation)
        batchs[i].c_d = data_x
        batchs[i].states_generation = states_generation
        batchs[i].states_discrimination = states_discrimination
        batchs[i].actions_generation = actions_generation
        batchs[i].actions_discrimination = actions_discrimination
        batchs[i].features_c = x_c
        batchs[i].features_d = x_d
        batchs[i].x_c_d = x_c_d
        batchs[i].op_list.append(op_list)
        batchs[i].con_or_dis = con_or_dis
        batchs[i].ori_x_c_d = ori_x_c_ds[i]
        batchs[i].specials = specials
    return batchs


def getBatches_update(args, data_infomation, dqn_generation, dqn_discrimination, df_c, df_d, df_label, c_generation, d_generation, batchs,
                  steps_done, device):
    states_ = []
    n_c_features = df_c.shape[1] - 1
    n_d_features = df_d.shape[1] - 1
    for i in range(args.episodes):
        states_.append(batchs[i].c_d)
    for_next = True
    steps_done += 1
    for i in range(args.episodes):
        features_c = batchs[i].features_c
        features_d = batchs[i].features_d
        processData = Update_data(features_c, features_d, data_infomation)
        actions_generation, states_generation = dqn_generation.choose_action_generation(states_[i], for_next, steps_done, batchs[i].con_or_dis)
        actions_discrimination, states_discrimination = dqn_discrimination.choose_action_discrimination(actions_generation, states_generation, for_next, steps_done)
        op_list = get_actions(actions_generation, actions_discrimination, c_generation, d_generation, df_c, df_d, n_c_features, n_d_features)
        df_c, df_d, special = processData.process_data(op_list)
        df_c = pd.concat([df_c, df_label], axis=1)
        df_d = pd.concat([df_d, df_label], axis=1)
        x_c = df_c.iloc[:, :-1].astype(np.float32).apply(np.nan_to_num)
        x_d = df_d.iloc[:, :-1].astype(np.float32).apply(np.nan_to_num)
        if x_d.shape[0] == 0:
            data_x = np.hstack((x_c, df_label.values.reshape(-1, 1)))
        elif x_c.shape[0] == 0:
            data_x = np.hstack((x_d, df_label.values.reshape(-1, 1)))
        else:
            data_x = np.hstack((x_c, x_d, df_label.values.reshape(-1, 1)))
        data_x = torch.from_numpy(data_x).float().transpose(0, 1).to(device)

        batchs[i].states_ = states_
        batchs[i].states_discrimination_ = states_discrimination
        batchs[i].states_generation_ = states_generation
        batchs[i].actions_generation_ = actions_generation
        batchs[i].actions_discrimination_ = actions_discrimination
        batchs[i].features_c_ = x_c
        batchs[i].features_d_ = x_d
        batchs[i].op_list_ = op_list
        batchs[i].c_d_ = data_x
    return batchs


def cal_reward(args, data_infomation,batchs, scores_b, type, metric, y):
    for i in range(args.episodes):
        x_c = batchs[i].features_c
        x_d = batchs[i].features_d
        if x_c.shape[0] != 0:
            x_c= delete_dup(x_c)
        if x_d.shape[0] != 0:
            x_d = delete_dup(x_d)
        if x_c.shape[0] == 0:
            x = np.array(x_d)
            x_c_d = x_d
        elif x_d.shape[0] == 0:
            x = np.array(x_c)
            x_c_d = x_c
        else:
            x = np.concatenate((x_c, x_d), axis=1)
            x_c_d = pd.concat([x_c, x_d], axis=1)
        y = np.array(y)
        d = args.d
        e = args.e
        f = args.f
        g = args.g
        score = get_reward(x, y, args, type, metric)

        Rre = 0
        Rde = 0
        Rco = 0
        num_Rre = 0
        num_Rde = 0
        num_Rco = 0

        for k, v in batchs[i].specials.items():
            if k == 'replace':
                for double in v:
                    double[1] = np.where(double[1] > 1e15, 0, double[1])
                    q = mutual_info_score(double[1], y)
                    w = mutual_info_score(double[0], y)
                    x = q - w
                    x = float(x)
                    Rre += x
                    num_Rre += 1
                if Rre != 0:
                    Rre /= num_Rre
            elif k == 'delete':
                for double in v:
                    double[1] = np.where(double[1] > 1e15, 0, double[1])
                    q = mutual_info_score(double[0], y)
                    w = mutual_info_score(double[1], y)
                    x = q - w
                    x = float(x)
                    Rde += x
                    num_Rde += 1
                if num_Rde != 0:
                    Rde /= num_Rde
            else:
                for double in v:
                    double[1] = np.where(double[1] > 1e15, 0, double[1])
                    double[1] = np.where(double[1] < -1e15, 0, double[1])
                    x = mutual_info_score(double[0], double[1])
                    x = float(x)
                    Rco += x
                    num_Rco += 1
                if num_Rco != 0:
                    Rco /= num_Rco

        batchs[i].scores = score
        batchs[i].scores_b = score
        w = (np.mean(score) - np.mean(scores_b))
        batchs[i].reward_1 = w
        batchs[i].reward_2 = d * w + e * Rre + f * Rde - g * Rco
        batchs[i].x_c_d = x_c_d
    return batchs


def get_reward(x, y, args, type, metric):
    if type == "classify":
        clf = RandomForestClassifier(n_estimators=10, random_state=0)
        scores = cross_val_score(clf, x, y, scoring='f1_micro', cv=5)
    else:
        model1 = RandomForestRegressor(n_estimators=10, random_state=0)
        rae_score1 = make_scorer(sub_rae, greater_is_better=True)
        scores = cross_val_score(model1, x, y, cv=5, scoring=rae_score1)
    return scores


def delete_dup(data):
    features, idx = np.unique(data, axis=1, return_index=True)
    y = data.iloc[:, np.sort(idx)]
    return y


def get_con_or_dis(data, data_infomation):
    con_or_dis = []
    features = data.columns.values
    for feature_name in features:
        flag = 1
        for con_name in data_infomation["discrete_columns"]:
            if con_name in feature_name:
                flag = -1
                break
        con_or_dis.append(flag)
    con_or_dis.append(0)
    return con_or_dis


def sub_rae(y, y_hat):
    y = np.array(y).reshape(-1)
    y_hat = np.array(y_hat).reshape(-1)
    y_mean = np.mean(y)
    absolute_errors = np.abs(y_hat - y)
    mean_errors = np.abs(y_mean - y)
    rae = np.sum(absolute_errors) / np.sum(mean_errors)
    res = 1 - rae
    return res