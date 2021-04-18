import env as car_env
import cvxpy as cp

def initialize_problem(num_time_steps, duration : float):
    time_step_magnitude = duration / num_time_steps
    h = cp.Parameter(time_step_magnitude)

    xpos = cp.Variable(num_time_steps+1)
    ypos = cp.Variable(num_time_steps+1)
    velocity = cp.Variable(num_time_steps+1)
    theta = cp.Variable(num_time_steps+1)
    kappa = cp.Variable(num_time_steps+1)
    jerk = cp.Variable(num_time_steps)
    pinch = cp.Variable(num_time_steps)

    # TODO: Initialize constraints
    def curr(var: cp.Variable):
        return var[1:]
    def prev(var: cp.Variable)
        return var[:-1]

    constraints = [
        # curr(x) = prev(x) + h * (prev(V) * prev(cp.sin(theta)) +

    ]

    # Initialize objective
    input = cp.vstack([jerk, pinch])
    assert input.shape == (2, num_time_steps)
    input_norm = cp.norm(input, axis=0)
    assert input_norm.shape == (num_time_steps,)

    objective = cp.Minimize(cp.sum(input_norm))
    problem = cp.Problem(objective, constraints)
    return problem, jerk, pinch

if __name__ == "__main__":
    #problem = initialize_problem(100)
    #print(problem)
    env = car_env.CarRacing(
            allow_reverse=True,
            grayscale=1,
            show_info_panel=1,
            discretize_actions=None,
            num_obstacles=100,
            num_tracks=1,
            num_lanes=1,
            num_lanes_changes=4,
            max_time_out=0,
            frames_per_state=4)

    env.reset()  # Put the car at the starting position
    for _ in range(1000):
      env.render()
      action = env.action_space.sample() # your agent here (this takes random actions)
      print(action)
      observation, reward, done, info = env.step(action)

      if done:
        observation = env.reset()
    env.close
