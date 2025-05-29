__version__ = "0.1.0.dev"

# Import necessary modules or packages
from .rapdhg import raPDHG
# from .mp_io import create_lp, create_qp, create_qp_from_gurobi
from .utils import plot_convergence, create_ot_problem
import gurobipy as gp

# def read_mps(filename):
#     model = gp.read(filename)
#     for v in model.getVars():
#         v.setAttr('vtype', 'C')
#     return create_qp_from_gurobi(model)

# Expose public API
__all__ = ["raPDHG", "create_ot_problem", "plot_convergence"]
