import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt

N = 24  # time steps to look ahead
path = cvx.Variable((N, 2))  # initialize the y pos and y velocity
flap = cvx.Variable(N - 1, boolean=True)  # initialize the inputs, whether or not the bird should flap in each step
last_solution = [False, False, False]  # seed last solution
last_path = [(0, 0), (0, 0)]  # seed last path

PIPEGAPSIZE = 100  # gap between upper and lower pipe
PIPEWIDTH = 52
BIRDWIDTH = 34
BIRDHEIGHT = 24
BIRDDIAMETER = np.sqrt(BIRDHEIGHT**2 + BIRDWIDTH**2)  # the bird rotates in the game, so we use its maximum extent
SKY = 0  # location of sky
GROUND = (512 * 0.79) - 1  # location of ground
PLAYERX = 57  # location of bird

def getPipeConstraintsDistance(x, y, lowerPipes):
    constraints = []  # initialize pipe constraint list
    pipe_dist = 0  # initialize distance from pipe center
    for pipe in lowerPipes:
        dist_from_front = pipe['x'] - x - BIRDDIAMETER
        dist_from_back = pipe['x'] - x + PIPEWIDTH
        if (dist_from_front < 0) and (dist_from_back > 0):
            constraints += [
                y <= (pipe['y'] - BIRDDIAMETER),  # y above lower pipe
                y >= (pipe['y'] - PIPEGAPSIZE)  # y below upper pipe
            ]
            # Accumulate squared distance from pipe center
            pipe_center = pipe['y'] - PIPEGAPSIZE / 2
            pipe_dist += cvx.square(pipe_center - y)
    return constraints, pipe_dist

def our_solve(playery, playerVelY, lowerPipes):
    pipeVelX = -4  # speed in x
    playerAccY = 1  # player's downward acceleration
    playerFlapAcc = -14  # player's speed on flapping

    # Unpack path variables
    y = path[:, 0]
    vy = path[:, 1]

    constraints = []  # Initialize constraint list
    constraints += [y <= GROUND, y >= SKY]  # Constraints for sky and ground
    constraints += [y[0] == playery, vy[0] == playerVelY]  # Initial conditions

    obj = 0  # Initialize objective accumulator

    x = PLAYERX
    xs = [x]  # Initialize x positions list

    for t in range(N - 1):  # Look ahead
        dt = t // 15 + 1  # Let time get coarser further in the look ahead
        x -= dt * pipeVelX  # Update x
        xs += [x]  # Add to list

        # Add y velocity and position constraints
        constraints += [
            vy[t + 1] == vy[t] + playerAccY * dt + playerFlapAcc * flap[t],
            y[t + 1] == y[t] + vy[t + 1] * dt
        ]

        # Add pipe constraints and accumulate squared distance from pipe center
        pipe_constraints, pipe_dist = getPipeConstraintsDistance(x, y[t + 1], lowerPipes)
        constraints += pipe_constraints
        obj += pipe_dist  # Accumulate squared distances

    # Define the desired y position (center of the gap)
    desired_y_position = PIPEGAPSIZE / 2

    # Define a more efficient objective function using squared terms
    objective = cvx.Minimize(
        cvx.sum_squares(y - desired_y_position) + 100 * cvx.sum(flap)
    )

    # Initialize and solve the problem
    prob = cvx.Problem(objective, constraints)
    try:
        prob.solve(verbose=False, solver="GUROBI")  # Use Gurobi if available

        last_path = list(zip(xs, y.value))  # Store the path
        last_solution = np.round(flap.value).astype(bool)  # Store the solution
        return last_solution[0], last_path  # Return the next input and path for plotting
    except:
        try:
            last_solution = last_solution[1:]  # If no solution, use the last solution
            last_path = [((x - 4), y) for (x, y) in last_path[1:]]
            return last_solution[0], last_path
        except:
            return False, [(0, 0), (0, 0)]  # If multiple failures, do nothing
