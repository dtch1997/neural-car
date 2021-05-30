from .car import SCPAgent, NeuralNetAgent, SCODNetAgent

registry = {
    'scp': SCPAgent,
    'nn': NeuralNetAgent, 
    'scod': SCODNetAgent    
}