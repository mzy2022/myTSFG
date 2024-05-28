class Batch(object):
    def __init__(self, args):
        self.args = args
        self.states = None
        self.actions_discrimination = None
        self.actions_generation = None
        self.values = None
        self.accs = None
        self.model = None
        self.orders = None
        self.steps = None
        self.features_c = None
        self.features_d = None
        self.df = None
        self.scores = None
        self.scores_test = None
        self.state = None
        self.actions_generation = None
        self.actions_discrimination = None
        self.encodes_states = None
        self.states_ = None
        self.encodes_states_ = None
        self.actions_generation_ = None
        self.actions_discrimination_ = None
        self.features_c_ = None
        self.features_d_ = None
        self.states_generation = None
        self.states_discrimination = None
        self.states_discrimination_ = None
        self.states_generation_ = None
        self.c_d = None
        self.reward = None
        self.scores_b = None
        self.x_c_d = None
        self.con_or_dis = None
        self.reward_1 = None
        self.reward_2 = None
        self.special = None
        self.best_score = 0
        self.op_list = []



