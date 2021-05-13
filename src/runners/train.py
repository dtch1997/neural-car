class TrainingRunner:

    def __init__(self):
        pass

    def __init__(self, args, env, agent):
        self.env = env
        self.agent = agent

    @staticmethod 
    def add_argparse_args(parser):
        return parser 

    @staticmethod 
    def from_argparse_args(args, env, agent):
        return TrainingRunner(args, env, agent)

    def run(self):
        raise NotImplementedError
    

    