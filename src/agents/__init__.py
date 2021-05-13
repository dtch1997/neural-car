from .car import SCPAgent, NeuralNetAgent

registry = {
    'scp': SCPAgent,
    'nn': NeuralNetAgent
}