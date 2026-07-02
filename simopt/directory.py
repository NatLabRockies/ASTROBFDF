#!/usr/bin/env python
"""
Summary
-------
Provide dictionary directories listing solvers, problems, and models.
"""
from __future__ import annotations

# import solvers
from simopt.solvers.astrodf import ASTRODF
from simopt.solvers.adam import ADAM
from .solvers.astrobfdf import ASTROBFDF
from .solvers.bfbo import BFBO
from .solvers.bfsag import BFSAG
from .solvers.example_astrobfdf import EXASTROBFDF

# import models and problems
from .models.example import Example, ExampleProblem

from .models.forretal import Forretal, ForretalProblem
from .models.branin import Branin, BraninProblem
from .models.colville import Colville, ColvilleProblem
from .models.rosenbrockmf import Rosenbrock, RosenbrockProblem

from .models.sscontmf import SSContMF, SSContMinCostMF
from .models.dynamnews import DynamNews, DynamNewsMaxProfit
from .models.mm1queuemf import MM1QueueMF, MM1MinMeanSojournTimeMF

# Import base
from simopt.base import Model, Problem, Solver

# directory dictionaries
solver_directory: dict[str, "Solver"] = {
    "ASTRODF": ASTRODF,
    "ADAM": ADAM,
    "ASTROBFDF": ASTROBFDF,
    "EXASTROBFDF": EXASTROBFDF,
    "BFBO": BFBO,
    "BFSAG": BFSAG
}

solver_unabbreviated_directory: dict[str, "Solver"] = {
    "ASTRO-DF (SBCN)": ASTRODF,
    "ADAM (SBCN)": ADAM,
}

problem_directory: dict[str, "Problem"] = {
    "EXAMPLE-1": ExampleProblem,
    "FORRETAL-1": ForretalProblem,
    "BRANIN-1": BraninProblem,
    "COLVILLE-1": ColvilleProblem,
    "ROSENBROCK-1": RosenbrockProblem,
    "SSCONTMF-1": SSContMinCostMF,
    "MM1MF-1": MM1MinMeanSojournTimeMF,
    "DNEWS-1": DynamNewsMaxProfit
}

problem_unabbreviated_directory: dict[str, "Problem"] = {
}
model_directory: dict[str, "Model"] = {
    "EXAMPLE": Example,

    "FORRETAL": Forretal,
    "BRANIN": Branin,
    "COLVILLE": Colville,
    "ROSENBROCK": Rosenbrock,

    "SSCONTMF": SSContMF,
    "MM1MF": MM1QueueMF,
    "DNEWS-1": DynamNews
}
model_unabbreviated_directory: dict[str, "Model"] = {
}
model_problem_unabbreviated_directory: dict[str, str] = {
    "Min Deterministic Function + Noise (SUCG)": "EXAMPLE",
    "Max Profit for Continuous Newsvendor (SBCG)": "CNTNEWS",
    "Min Mean Sojourn Time for MM1 Queue (SBCG)": "MM1",
    "Min Total Cost for Facility Sizing (SSCG)": "FACSIZE",
    "Max Service for Facility Sizing (SDCN)": "FACSIZE",
    "Max Revenue for Revenue Management Temporal Demand (SDDN)": "RMITD",
    "Min Total Cost for (s, S) Inventory (SBCN)": "SSCONT",
    "Max Revenue for Iron Ore (SBDN)": "IRONORE",
    "Max Revenue for Continuous Iron Ore (SBCN)": "IRONORE",
    "Max Profit for Dynamic Newsvendor (SBDN)": "DYNAMNEWS",
    "Min Cost for Dual Sourcing (SBDN)": "DUALSOURCING",
    "Min Total Cost for Discrete Contamination (SSDN)": "CONTAM",
    "Min Total Cost for Continuous Contamination (SSCN)": "CONTAM",
    "Min Avg Difference for Chess Matchmaking (SSCN)": "CHESS",
    "Min Mean Longest Path for Stochastic Activity Network (SBCG)": "SAN",
    "Max Revenue for Hotel Booking (SBDN)": "HOTEL",
    "Max Revenue for Restaurant Table Allocation (SDDN)": "TABLEALLOCATION",
    "Max Log Likelihood for Gamma Parameter Estimation (SBCN)": "PARAMESTI",
    "Min Mean Longest Path for Fixed Stochastic Activity Network (SBCG)": "FIXEDSAN",
    "Min Total Cost for Communication Networks System (SDCN)": "NETWORK",
    "Min Total Departed Visitors for Amusement Park (SDDN)": "AMUSEMENTPARK",
}
model_problem_class_directory: dict[str, "Model"] = {
}
