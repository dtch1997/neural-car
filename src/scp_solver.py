import cvxpy as cp 

def initialize_problem(num_time_steps):
    xpos = cp.Variable(num_time_steps)
    ypos = cp.Variable(num_time_steps)
    velocity = cp.Variable(num_time_steps)
    theta = cp.Variable(num_time_steps)
    kappa = cp.Variable(num_time_steps)
    jerk = cp.Variable(num_time_steps)
    pinch = cp.Variable(num_time_steps)

    # TODO: Initialize constraints
    constraints = []

    # Initialize objective
    input = cp.vstack([jerk, pinch])
    assert input.shape == (2, num_time_steps)
    input_norm = cp.norm(input, axis=0)
    assert input_norm.shape == (num_time_steps,)

    objective = cp.Minimize(cp.sum(input_norm))
    problem = cp.Problem(objective, constraints)
    return problem

if __name__ == "__main__":
    problem = initialize_problem(100)
    print(problem)

