from collections import defaultdict
from itertools import combinations

import gurobipy as gp
import numpy as np
from gurobipy import GRB

n_agents = 14
agents = np.arange(0, n_agents)
t_max = 100

# the formula for the solution to this problem is as follows:
# round down to the nearest even number
## floor(n_agents / 2)
# then muliply that by 1/t_max

# np.floor(n_agents / 4) / np.sqrt(t_max)


agent_combos = list(combinations(agents, r=2))

env = gp.Env()
model = gp.Model(env=env)

# build decision vars
hit_times = defaultdict(int)

# absolute value is non-linear so need to linearize with aux variables for gurobi to work
# Auxiliary variable for the expression inside the absolute value
expr_vars = defaultdict(int)
# Auxiliary variable for the absolute value itself
abs_vars = defaultdict(int)

for i in range(n_agents):
    hit_times[i] = model.addVar(vtype=GRB.BINARY, name=f"hit_time_{i}")
    # hit_times[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=t_max, name=f"hit_time_{i}")

model.update()

# build constraints
for i, combo in enumerate(agent_combos):
    expr_vars[combo] = model.addVar(lb=-GRB.INFINITY, name=f"expr_var_{i}")
    model.addConstr(
        expr_vars[combo] == (hit_times[combo[0]] - hit_times[combo[1]]) / t_max
    )

    abs_vars[i] = model.addVar(name=f"abs_var_{i}")
    model.addConstr(abs_vars[i] == gp.abs_(expr_vars[combo]))

# build objective
obj = 0
for i, combo in enumerate(agent_combos):
    obj += abs_vars[i]
    # obj += (hit_times[combo[0]] - hit_times[combo[1]]) ** 2
    print(combo[0], combo[1])

# solve
# Set y as the objective to minimize
model.setObjective(obj, GRB.MAXIMIZE)
model.optimize()

print("Max team-level objective", model.ObjVal)

print("agent times")
for i, time in hit_times.items():
    print(i, time.X)

# # this will produce the largest differences, but won't assign values to t_i, t_j directly
# # this doesn't capture the cross terms, treats them as independent

# # do this as a gurobi GP, idk how to do this as a scipy thing

# # all are in the range of 0 to t_max
# A = np.diag(np.ones(n_agents))
# lower_bounds = np.zeros(n_agents)
# upper_bounds = t_max * np.ones(n_agents)

# constraints = LinearConstraint(A, lower_bounds, upper_bounds)
# # Run the solver
# res = milp(c=c, constraints=constraints, integrality=integrality)

# if res.success:
#     print(f"Optimal x, y: {res.x}")
#     print(f"Max Objective Value: {-res.fun}")  # Negate back to positive
# else:
#     print("Solver failed.")

# print("Breakpoint ")
# __import__("ipdb").set_trace(context=5)
# # # Minimize: -3x - 2y (Equivalent to maximizing 3x + 2y)
# # c = np.array([-3, -2])

# # # Specify variable types: 1 means Integer, 0 means Continuous
# # integrality = np.array([1, 1])

# # # Constraints matrix (A)
# # #  1x + 2y <= 5   -->  -inf <=  1x + 2y <= 5
# # #  2x -  1y >= 1   -->     1 <=  2x - 1y <= inf
# A = np.array([[1, 2], [2, -1]])
# lower_bounds = np.array([-np.inf, 1])
# upper_bounds = np.array([5, np.inf])

# constraints = LinearConstraint(A, lower_bounds, upper_bounds)

# # Run the solver
# res = milp(c=c, constraints=constraints, integrality=integrality)

# if res.success:
#     print(f"Optimal x, y: {res.x}")
#     print(f"Max Objective Value: {-res.fun}")  # Negate back to positive
# else:
#     print("Solver failed.")
