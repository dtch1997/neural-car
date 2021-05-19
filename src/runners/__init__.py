from src.runners.gen import DataGenerationRunner
from .train import TrainingRunner
from .eval import EvaluationRunner
from .gen import DataGenerationRunner

registry = {
    'train': TrainingRunner,
    'test': EvaluationRunner,
    'gen': DataGenerationRunner
}