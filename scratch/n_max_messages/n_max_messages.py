import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt


def get_max_sent_messages(n_agents: int, alpha_thres: float) -> float:
    # Create a new model
    model = gp.Model()
    model.setParam("OutputFlag", 0)  # Disables solver print output

    # Create variables
    alpha = {}
    beta = {}
    for i in range(n_agents):
        alpha[i] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"alpha_{i}")
        beta[i] = model.addVar(vtype=GRB.BINARY, name=f"beta_{i}")

    # enforce integer constraints on binary vars
    model.Params.IntegralityFocus = 1

    # constraints
    # outgoing attention sums to 1 for each agent
    sum = 0
    for i in range(n_agents):
        sum += alpha[i]
    model.addConstr(sum == 1, name="total attention")

    # Constants
    # M is chosen to be as small as possible given the bounds on alpha and beta
    eps = 0.00000001
    M = 1 + eps

    for i in range(n_agents):
        # If alpha >= alpha_thres, then beta = 1, otherwise beta = 0
        model.addConstr(
            alpha[i] >= alpha_thres + eps - M * (1 - beta[i]), name=f"bigM_constr1_{i}"
        )
        model.addConstr(alpha[i] <= alpha_thres + M * beta[i], name=f"bigM_constr2_{i}")

    model.update()

    # define objective
    obj = 0
    for i in range(n_agents):
        obj += beta[i]
    model.setObjective(obj, GRB.MAXIMIZE)

    # # infeasibility debugging
    # model.params.DualReductions = 0

    model.update()
    model.optimize()

    # also uncomment line above model.optimize()
    # if model.status == GRB.Status.INFEASIBLE:
    #     model.computeIIS()
    #     print("Irreducible Inconsistent Subset Infeasible Constraints\n")
    #     for c in model.getConstrs():
    #         if c.IISConstr:
    #             print(f"{c.constrName}")

    return model.ObjVal


max_n_agents = 20
agent_step = 1
alpha_step = 0.05

n_agents = np.arange(1, max_n_agents + agent_step, agent_step)
alpha_thresholds = np.arange(0, 1 + alpha_step, alpha_step)

# create a mesh grid for the data points to be processed
_xx, _yy = np.meshgrid(n_agents, alpha_thresholds)
n_agents_plot, alphas_plot = _xx.ravel(), _yy.ravel()

out = []
for n, alpha in zip(n_agents_plot, alphas_plot):
    out.append(get_max_sent_messages(n, alpha))

top = np.array(out)


bottom = np.zeros_like(top)

# create 3D figure
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection="3d")

width = agent_step / 2
depth = alpha_step / 2

plt.style.use("fast")
ax.bar3d(
    n_agents_plot - (width / 2),
    alphas_plot - (depth / 2),
    bottom,
    width,
    depth,
    top,
    shade=True,
)
ax.set_xlabel("N Agents")
ax.set_ylabel("Alpha threshold")
ax.set_zlabel("Max. Outgoing Messages per Agent")

plt.show()
# plt.savefig("n_max_messages.png")
