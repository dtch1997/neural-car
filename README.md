# AA203 - Neural Car Controller
Train a neural network to emulate a MPC controller for a Reed-Shepp car

## Getting Started
With Anaconda / Miniconda:
```
conda create -n neural-car python=3.8
conda activate neural-car

conda install pytorch cudatoolkit=10.2 -c pytorch
python -m pip install gym[box2d] numpy matplotlib cvxpy simple-parsing pillow opencv-python pandas pytorch-lightning
```

Generate training data, train NN agent, and evaluate NN agent: 
```
python main.py --runner gen --agent scp --num-simulation-time-steps 5000 --num-time-steps-ahead 1000 --num-rollouts 1 --world-seed 0
python main.py --runner train --agent nn
# check your lightning_logs dir to find the exact checkpoint to load
python main.py --runner test --agent nn --checkpoint-path <path_to_checkpoint.ckpt> --num-simulation-time-steps 5000 --num-rollouts 1 --world-seed 0
```
