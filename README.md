# AA203 - Neural Car Controller
Train a neural network to emulate a MPC controller for a Reed-Shepp car

## Getting Started
With Anaconda / Miniconda:
```
conda create -n neural-car python=3.8
conda activate neural-car

conda install pytorch cudatoolkit=10.2 -c pytorch
conda install -c anaconda swig
python -m pip install gym[box2d] numpy matplotlib cvxpy simple-parsing pillow opencv-python pandas pytorch-lightning h5py

git submodule update --init --recursive 
cd dep/SCOD 
pip install -e pytorch-hessian-eigenthings/
pip install -e curvature/
pip install -e .
```

## Alternate download instructions for SCOD
If you are having trouble installing SCOD, you can try the following instead: 
```
git clone --recurse-submodules https://github.com/StanfordASL/SCOD.git
cd SCOD
pip install -e pytorch-hessian-eigenthings/
pip install -e curvature/
pip install -e .
```

## Generate training data
```
python main.py --runner gen --agent scp --num-simulation-time-steps 5000 --num-time-steps-ahead 1000 --num-rollouts 1 --num-goals 1 --world-seed 0
```

## Train NN agent
Generate training data, train NN agent, and evaluate NN agent: 
```
python main.py --runner train --agent nn --data-filepath datasets/simulation_output.hdf5
# check your lightning_logs dir to find the exact checkpoint to load
python main.py --runner test --agent nn --checkpoint-path <path_to_checkpoint.ckpt> --num-simulation-time-steps 5000 --num-rollouts 1 --num-goals 1 --world-seed 0
```

## Use trained NN in SCOD agent 
```
# check your lightning_logs dir to find the exact checkpoint to load
python main.py --runner test --agent scod --checkpoint-path <path_to_checkpoint.ckpt>--num-simulation-time-steps 5000 --num-rollouts 1 --num-goals 1 --world-seed 0  --data-filepath datasets/simulation_output.hdf5 
```

## Visualize training statistics with Tensorboard: 
```
tensorboard --logdir lightning_logs
```
