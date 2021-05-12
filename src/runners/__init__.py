from .train import TrainingRunner
from .eval import EvaluationRunner

registry = {
    'train': TrainingRunner,
    'test': EvaluationRunner
}