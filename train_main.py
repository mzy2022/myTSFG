import argparse
import logging
import os
import random
import sys
import warnings

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from utils import get_binning_df, get_action, log_b
from feature_generation import DQN_generation, DQN_discrimination
from feature_generation.replay import Replay_generation, Replay_discrimination
from feature_generation.train import getBatches, cal_reward, getBatches_update
from feature_generation.Batch import Batch
from utils import datainfos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
warnings.filterwarnings("ignore")

class TSFG:
    def __init__(self, input_data, args):
        log_path = fr"./logs/{args.file_name}"
        log_b(log_path)
        data_info = datainfos[args.file_name]
        self.target = data_info['target']
        args.target = self.target
        self.type = data_info['type']
        if self.type == 'classify':
            self.metric = 'f1'
        elif self.type == 'regression':
            self.metric = 'rae'
        self.v_columns = data_info['v_columns']
        self.d_columns = data_info['d_columns']
        logging.info(f'File name: {args.file_name}')
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        random.seed(1)
        np.random.seed(1)
        self.best_score = 0
        self.ori_df = input_data
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def main_function(self, args):
        df = self.ori_df
        v_columns, d_columns = self.v_columns, self.d_columns
        target, type, metric = self.target, self.type, self.metric
        new_df, new_v_columns, new_d_columns = get_binning_df(args, df, v_columns, d_columns, type)
        n_features_c, n_features_d = len(new_v_columns), len(new_d_columns)
        c_generation, d_generation, sps = get_action(n_features_c, n_features_d)
        score_b, scores_b = self.data_scores(new_df, args, type, metric)
        score_ori, scores_ori = self.data_scores(df, args, type, metric)
        logging.info(f'score_ori={score_ori}')
        df_c, df_d = new_df.loc[:, new_v_columns + [target]], new_df.loc[:, new_d_columns + [target]]
        ori_continuous_data = df.loc[:, v_columns]
        df_label = new_df.loc[:, target]
        feature_nums = n_features_c + n_features_d
        data_nums = new_df.shape[0]
        operations_c = len(c_generation)
        operations_d = len(d_generation)
        d_model = args.d_model
        batch_size = args.batch_size
        memory_size = args.memory
        hidden_size = args.hidden_size
        self.memory_generation = Replay_generation(batch_size, memory_size)
        self.memory_discrimination = Replay_discrimination(batch_size, memory_size)
        self.dqn_generation = DQN_generation(args, data_nums, feature_nums, operations_c, operations_d, d_model, self.memory_generation,
                               self.device)
        self.dqn_discrimination = DQN_discrimination(args, operations_c, operations_d, hidden_size, d_model, self.memory_discrimination, self.device)
        self.steps_done = 0
        data_infomation = {'dataframe': new_df,
                           'continuous_columns': new_v_columns,
                           'discrete_columns': new_d_columns,
                           'continuous_data': df_c,
                           'discrete_data': df_d,
                           'label_name': target,
                           'type': type,
                           'ori_continuous_data': ori_continuous_data
                           }

        for epoch in tqdm(range(args.epochs)):
            batchs = []
            for _ in range(args.episodes):
                batch = Batch(args)
                batch.scores_b = scores_b
                batch.best_score = 0
                batchs.append(batch)
            for i in range(args.steps_num):
                batchs_dqn = getBatches(args, self.dqn_generation, self.dqn_discrimination, data_infomation, df_c, df_d,
                                        df_label, c_generation, d_generation, i, batchs, self.device, self.steps_done)
                batchs_reward = cal_reward(args, data_infomation, batchs_dqn, scores_b, type, metric,
                                                    df_label.values)
                batchs_dqn_ = getBatches_update(args, data_infomation, self.dqn_generation, self.dqn_discrimination, df_c,
                                                df_d,
                                                df_label, c_generation, d_generation, batchs_reward, self.steps_done, self.device)

                self.dqn_generation.store_transition(args, batchs_dqn_)
                self.dqn_discrimination.store_transition(args, batchs_dqn_)

                for i, batch in enumerate(batchs_dqn_):
                    if np.mean(batch.scores) > self.best_score:
                        self.best_score = np.mean(batch.scores)
                        logging.info(f"epoch:{epoch}_new_best_score{self.best_score}")
                self.dqn_generation.learn(args, batchs_dqn_, self.device)
                self.dqn_discrimination.learn(args, batchs_dqn_, self.device)

    def data_scores(self, df: pd.DataFrame, args, type, metric):
        target = args.target
        X = df.drop(columns=[target])
        y = df[target]
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        if type == "classify":
            clf = RandomForestClassifier(n_estimators=10, random_state=0)
            scores = cross_val_score(clf, X, y, scoring='f1_micro', cv=5)
        else:
            model = RandomForestRegressor(n_estimators=10, random_state=0)
            rae_score1 = make_scorer(rae, greater_is_better=True)
            scores = cross_val_score(model, X, y, cv=5, scoring=rae_score1)
        return np.array(scores).mean(), scores

def rae(y, y_hat):
    y = np.array(y).reshape(-1)
    y_hat = np.array(y_hat).reshape(-1)
    y_mean = np.mean(y)
    absolute_errors = np.abs(y_hat - y)
    mean_errors = np.abs(y_mean - y)
    rae = np.sum(absolute_errors) / np.sum(mean_errors)
    res = 1 - rae
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default="0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps_num", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--file_name", type=str, default='airfoil')
    parser.add_argument("--seed", type=int, default=1, help='seed')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--memory', type=int, default=24, help='memory capacity')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--hidden-size', type=int, default=8)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d", type=float, default=1.0)
    parser.add_argument("--e", type=float, default=0.1)
    parser.add_argument("--f", type=float, default=0.1)
    parser.add_argument("--g", type=float, default=0.01)
    args = parser.parse_args()
    dataset_path = f"{BASE_DIR}\\data\\{args.file_name}.csv"
    df = pd.read_csv(dataset_path)
    autofe = TSFG(df, args)
    autofe.main_function(args)