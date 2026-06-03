from abc import ABC, abstractmethod
from dataclasses import dataclass
import gurobipy as gp
from gurobipy import GRB


@abstractmethod
@dataclass
class ProblemConfigBase:
    """manages optimization problem configuration"""

    pass


@abstractmethod
@dataclass
class VariablesBase:
    """manages optimization problem decision variables"""

    pass


@abstractmethod
@dataclass
class ConstraintsBase:
    """manages optimization problem constraints. Most constraints
    can be left 'inside' the optimization model. Only use this for
    constraints that you need to directly access i.e., to update
    them during the optimization process, to send them to other
    optimization problems, etc."""

    pass


@abstractmethod
@dataclass
class SolutionBase:
    """manages optimization problem solution data"""

    pass


@abstractmethod
class OptimizationProblem(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.model: gp.Model

    # model building
    def build_model(self) -> None:
        """build computational model of the optimization problem"""
        # if you need to add license info, do it in another file and
        # import that as a dict called GUROBI_PARAMS and swap the env definition
        # env = gp.Env(params=GUROBI_PARAMS)
        env = gp.Env()
        self.model = gp.Model(self.__class__.__name__, env=env)

        # infeasibility debugging
        # self.model.params.DualReductions = 0
        self._build_variables()
        self.model.update()

        self._build_constraints()
        self.model.update()

        self._build_objective()
        self.model.update()

        # these should be instantiated in the child class
        # assert isinstance(self.opt_config, ProblemConfigBase)
        # assert isinstance(self.opt_vars, VariablesBase)
        # assert isinstance(self.opt_constrs, ConstraintsBase)
        # assert isinstance(self.opt_sol, SolutionBase)

    @abstractmethod
    def _build_variables(self) -> None:
        """define variables in the Gurobi model"""

    @abstractmethod
    def _build_constraints(self) -> None:
        """define constraints in the Gurobi model"""

    @abstractmethod
    def _build_objective(self) -> None:
        """define objective function in the Gurobi model"""

    # optimization
    def optimize(self) -> None:
        """run an optimization algorithm"""
        self.model.optimize()

        # infeasibility / unbounded debugging
        if self.model.Status == GRB.INF_OR_UNBD:
            self.model.params.DualReductions = 0

            # solve again to figure out whether it is infeasible or unbounded
            self.model.reset()
            self.model.optimize()

            if self.model.Status == GRB.INFEASIBLE:
                # infeasibility debugging
                # also uncomment the line in _build_model()
                self.model.computeIIS()
                print("Irreducible inconsistent subset of infeasible constraints\n")
                for c in self.model.getConstrs():
                    if c.IISConstr:
                        print(f"{c.ConstrName}")
            elif self.model.Status == GRB.UNBOUNDED:
                print(
                    "Objective function unbounded. Please re-check the data you put into the objective."
                )

    # solution building
    @abstractmethod
    def _build_solution(self) -> None:
        """organize solution data from the solved Gurobi model"""

    # other helper functions
    def check_if_optima_found(self, verbose=False) -> bool:
        status_str: str
        if self.model.Status == 2:
            status_str = "Optima found"
            return_val = True

        else:
            status_str = (
                "Gurobi model may be infeasible, unbounded, or have some other issue."
            )
            return_val = False

        if verbose:
            return_str: str = f"{self.model.ModelName} {status_str}\n"
            print(return_str)

        return return_val

    def get_objective_value(self) -> float:
        return self.model.ObjVal
