from simple_parsing import ArgumentParser
from src.agents import registry as agent_registry
from src.envs import registry as environment_registry
from src.runners import registry as runner_registry

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--runner', default = 'test', choices = runner_registry.keys(), 
                        help = "Specify whether to train or test an agent")
    parser.add_argument('--environment', default = "car", choices = environment_registry.keys(), 
                        help = "Environment to train/test agent on")
    parser.add_argument('--agent', default = 'scp', choices = agent_registry.keys(),
                        help = 'Agent to train/test')

    args, _ = parser.parse_known_args()
    runner_class = runner_registry[args.runner]
    agent_class = agent_registry[args.agent]
    environment_class = environment_registry[args.environment]

    parser = agent_class.add_argparse_args(parser)
    parser = environment_class.add_argparse_args(parser)
    parser = runner_class.add_argparse_args(parser)
    return parser.parse_args()

def main(args):
    runner_class = runner_registry[args.runner]
    agent_class = agent_registry[args.agent]
    environment_class = environment_registry[args.environment]
    
    agent = agent_class.from_argparse_args(args)
    env = environment_class.from_argparse_args(args)
    runner = runner_class.from_argparse_args(args, env, agent)

    runner.run()

if __name__ == "__main__":
    args = get_args()
    main(args)
