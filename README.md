# AA203 - Neural Car Controller
Train a neural network to emulate a MPC controller for Dubin's car

## Getting Started
With Anaconda / Miniconda:
```
conda create -n neural-car
conda activate neural-car

conda install pytorch cudatoolkit=10.2 -c pytorch
python -m pip install gym[box2d] numpy matplotlib cvxpy simple-parsing pillow opencv-python pandas
```

Evaluate neural net agent: 
```
python main.py --agent nn
```
