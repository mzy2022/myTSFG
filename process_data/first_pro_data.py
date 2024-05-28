from .operation import *
class ProcessData():
    def __init__(self, args):
        self.ori_dataframe = args['dataframe']
        self.Candidate_features = args['dataframe']
        self.continuous_columns = args['continuous_columns']
        self.discrete_columns = args['discrete_columns']
        self.type = args['type']
        self.label = self.ori_dataframe[args['label_name']].values
        self.continuous = self.ori_dataframe[self.continuous_columns].copy()
        self.ori_features_continuous = self.ori_dataframe[self.continuous_columns]
        self.cross = self.ori_dataframe[self.discrete_columns].copy()
        self.ori_features_cross = self.ori_dataframe[self.discrete_columns]

    def process_data(self, actions):
        self.specials = {'concat': [], 'replace': [], 'delete': []}
        for action, action_list in actions.items():
            if action in ['add', 'subtract', 'multiply', 'divide']:
                self.double_oper(action, action_list)
            elif action in ["singles"]:
                self.single_oper(action, action_list)
            else:
                self.feature_cross(action, action_list)
        return self.continuous, self.cross, self.specials

    def double_oper(self, action, action_list):
        operation = action
        for feature_index_tuple in action_list:
            special_op = feature_index_tuple[2]
            feature1_index, feature2_index = feature_index_tuple[:2]
            if self.continuous.shape[1] == 0:
                continue
            feature1 = self.continuous.iloc[:, feature1_index]
            feature2 = self.ori_features_continuous.iloc[:, feature2_index]
            x = (abs(feature1 - feature2)).sum()
            if abs(x) < 0.1 and operation in ['subtract', 'divide']:
                continue
            else:
                name = feature1.name + '_' + operation + '_' + feature2.name
                new_fe = globals()[operation](feature1.values, feature2.values)
            new_fe = np.nan_to_num(new_fe).astype(np.float32)
            new_fe = np.where(new_fe > 10000000, 0, new_fe)
            if special_op == 0:
                self.continuous[name] = new_fe
                self.specials['concat'].append([feature1.values, new_fe])
            elif special_op == 1:
                self.continuous[feature1.name] = new_fe
                self.continuous = self.continuous.rename(columns={feature1.name: name})
                self.specials['replace'].append([feature1.values, new_fe])

    def single_oper(self, action, action_list):
        operations = ["rabs", 'square', 'reciprocal', 'log', 'sqrt', 'power3', 'None']
        for index, operation in action_list.items():
            if self.continuous.shape[1] == 0:
                continue
            ori_feature = self.continuous.iloc[:, index].copy()
            if operation[0] == 6:
                new_fe = np.array(self.continuous.iloc[:, index])
                name = self.continuous.iloc[:, index].name
            else:
                name = ori_feature.name + '_' + operations[operation[0]]
                new_fe = globals()[operations[operation[0]]](ori_feature.values)

            new_fe = np.nan_to_num(new_fe).astype(np.float32)
            if operation[1] == 0:
                self.continuous[name] = new_fe.copy()
                self.specials['concat'].append([ori_feature.values, new_fe])
            elif operation[1] == 1:
                self.continuous[ori_feature.name] = new_fe
                self.continuous = self.continuous.rename(columns={ori_feature.name: name})
                self.specials['replace'].append([ori_feature.values, new_fe])

    def feature_cross(self, action, action_list):
        delete_index = []
        if action == "cross":
            for actions in action_list:
                if actions[1] == 'None':
                    continue
                index1 = index = actions[0]
                ori_fe1 = self.cross.iloc[:, index1]
                ori_fe2 = self.ori_features_cross.iloc[:, actions[1]]
                name = ori_fe1.name + '_cross_' + ori_fe2.name
                feasible_values = {}
                cnt = 0
                for x in np.unique(ori_fe1):
                    for y in np.unique(ori_fe2):
                        feasible_values[str(int(x)) + str(int(y))] = cnt
                        cnt += 1
                new_fe = generate_cross_fe(ori_fe1.values, ori_fe2, feasible_values)
                new_fe = np.nan_to_num(new_fe).astype(np.float32)
                if actions[2] == 0:
                    self.cross[name] = new_fe
                elif actions[2] == 1:
                    self.cross[ori_fe1.name] = new_fe
                    self.cross = self.cross.rename(columns={ori_fe1.name: name})
                else:
                    delete_index.append(index)

        elif action == "nunique":
            for actions in action_list:
                index = actions[0]
                ori_fe1 = self.cross.iloc[:, index]
                ori_fe2 = self.ori_features_cross.iloc[:, actions[1]]
                name = ori_fe1.name + '_nunique_' + ori_fe2.name
                new_fe = get_nunique_feature(ori_fe1, ori_fe2)
                new_fe = np.nan_to_num(new_fe).astype(np.float32)
                if actions[2] == 0:
                    self.cross[name] = new_fe
                    self.specials['concat'].append([ori_fe1.values, new_fe])
                elif actions[2] == 1:
                    self.cross[ori_fe1.name] = new_fe
                    self.cross = self.cross.rename(columns={ori_fe1.name: name})
                    self.specials['replace'].append([ori_fe1.values, new_fe])
                else:
                    delete_index.append(index)
                    self.specials['delete'].append([ori_fe1.values, new_fe])
