"""
Summary
-------
BFBO: Bi-fidelity Bayesian Optimization.
See Forrester, Alexander IJ, András Sóbester, and Andy J. Keane. "Multi-fidelity optimization via surrogate modelling." Proceedings of the royal society a: mathematical, physical and engineering sciences 463.2088 (2007): 3251-3269. for details
"""
from __future__ import annotations

import numpy as np
from simopt.base import Solver, Problem, Solution
import numpy as np
import scipy
from scipy.optimize import minimize

from smt.applications.mfk import NestedLHS, MFK
import contextlib
import sys
import time

#from autograd import grad
#import autograd.numpy as anp


class BFBO(Solver):
    """
    An algorithm for first-order gradient-based optimization of
    stochastic objective functions, based on adaptive estimates of lower-order moments.

    Attributes
    ----------
    name : string
        name of solver
    objective_type : string
        description of objective types:
            "single" or "multi"
    constraint_type : string
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : string
        description of variable types:
            "discrete", "continuous", "mixed"
    gradient_needed : bool
        indicates if gradient of objective function is needed
    factors : dict
        changeable factors (i.e., parameters) of the solver
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)
    rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
        list of RNGs used for the solver's internal purposes

    Arguments
    ---------
    name : str
        user-specified name for solver
    fixed_factors : dict
        fixed_factors of the solver

    See also
    --------
    base.Solver
    """
    def __init__(self, name: str = "BFBO", fixed_factors: dict = {}):
        self.name = name
        self.objective_type = "single"
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.gradient_needed = False
        self.specifications = {
            "crn_across_solns": {
                "description": "use CRN across solutions?",
                "datatype": bool,
                "default": True
            },
            "costs_mf": {
                "description": "costs for multi-fidelity functions",
                "datatype": list,
                "default": [1, 0.3]
            },
            "r": {
                "description": "number of replications taken at each solution",
                "datatype": int,
                "default": 20
            }
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "r": self.check_r,
            "beta_1": self.check_beta_1,
            "beta_2": self.check_beta_2,
            "alpha": self.check_alpha,
            "epsilon": self.check_epsilon,
            "sensitivity": self.check_sensitivity
        }
        super().__init__(fixed_factors)

    def check_r(self):
        return self.factors["r"] > 0

    def check_beta_1(self):
        return self.factors["beta_1"] > 0 & self.factors["beta_1"] < 1

    def check_beta_2(self):
        return self.factors["beta_2"] > 0 & self.factors["beta_2"] < 1

    def check_alpha(self):
        return self.factors["alpha"] > 0

    def check_epsilon(self):
        return self.factors["epsilon"] > 0

    def check_sensitivity(self):
        return self.factors["sensitivity"] > 0

    
    def solve(self, problem: "Problem") -> tuple[list["Solution"], list[int]]:
        """
        Run a single macroreplication of a solver on a problem.

        Arguments
        ---------
        problem : Problem object
            simulation-optimization problem to solve
        crn_across_solns : bool
            indicates if CRN are used when simulating different solutions

        Returns
        -------
        recommended_solns : list of Solution objects
            list of solutions recommended throughout the budget
        intermediate_budgets : list of ints
            list of intermediate budgets when recommended solutions changes
        """
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0
        hf_solns = []
        lf_solns = []

        find_bounds_rng = self.rng_list[1]
        # Generate many dummy solutions without replication only to find a reasonable maximum radius
        dummy_solns = []
        for i in range(1000 * problem.dim):
            dummy_solns += [problem.get_random_solution(find_bounds_rng)]
        
        bound_max_arr = []
        for i in range(problem.dim):
            bound_max_arr += [min(max([sol[i] for sol in dummy_solns])-min([sol[i] for sol in dummy_solns]), 
                                  problem.upper_bounds[0] - problem.lower_bounds[0])]          
        bound_max = max(bound_max_arr)

        # Default values.
        r = self.factors["r"]
        costs = self.factors["costs_mf"]

        # Start with the initial solution.
        new_solution = self.create_new_solution(problem.factors["initial_solution"], problem)
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)
        problem.simulate(new_solution, r)
        expended_budget += r
        best_solution = new_solution
        hf_solns.append(new_solution)
        lf_solns.append(new_solution)
        
        # Upper bound and lower bound.        
        xlimits = np.array([np.array(problem.lower_bounds) + 0.01, np.array(problem.upper_bounds)]).T
        xlimits[:, 1] = np.where(np.isinf(xlimits[:, 1]), 10 * bound_max, xlimits[:, 1])
        xdoes = NestedLHS(nlevel=2, xlimits=xlimits, random_state=0)
        x_initial_l, x_initial_h = xdoes(problem.dim+1)

        if x_initial_l.size == 0 or x_initial_h.size == 0:
            raise ValueError("xdoes(3) returned an empty array.")

        for i in range(len(x_initial_l)):
            new_solution = self.create_new_solution(tuple(x_initial_l[i]), problem)
            problem.simulate(new_solution, r)
            expended_budget += r * costs[1]
            lf_solns.append(new_solution)
        
        
        for i in range(len(x_initial_h)):
            new_solution = self.create_new_solution(tuple(x_initial_h[i]), problem)
            problem.simulate(new_solution, r)
            expended_budget += r 
            hf_solns.append(new_solution)

        theta0 = x_initial_h.shape[1] * [1.0]

        sm = MFK(theta0=theta0, eval_noise=True, use_het_noise=True, propagate_uncertainty=False, n_start=1)
        
        ## Low-fidelity
        # stack x
        x_l_set = np.vstack([sol.x for sol in lf_solns])

        # duplicate x 
        all_x_l_set = np.repeat(x_l_set, r, axis=0)
        all_y_l_set = []  # List to store each arr

        for sol in lf_solns:
            y_l_set = sol.lf_f[sol.lf_f != 0]  # Remove zeros
            y_l_set = y_l_set.reshape(-1, 1)  # Ensure it's a column vector
            all_y_l_set.append(y_l_set)  # Store in list
        
        all_y_l_set = np.vstack(all_y_l_set)

        ## High-fidelity
        x_h_set = np.vstack([sol.x for sol in hf_solns])

        # duplicate x 
        all_x_h_set = np.repeat(x_h_set, r, axis=0)
        all_y_h_set = []  # List to store each arr
        
        for sol in hf_solns:
            y_h_set = sol.objectives[sol.objectives != 0]  # Remove zeros
            y_h_set = y_h_set.reshape(-1, 1)  # Ensure it's a column vector
            all_y_h_set.append(y_h_set)  # Store in list
        
        all_y_h_set = np.vstack(all_y_h_set)

        # Find the best solution
        best_solution = min(hf_solns, key=lambda soln: -1 * problem.minmax[0] * soln.objectives_mean)

        # Acquisition Function (Expected Improvement)
        def expected_improvement(x, model, y_best, xi=0.01):
            x = np.atleast_2d(x)
            mu, sigma = model.predict_values(x), model.predict_variances(x)
            sigma = np.sqrt(np.maximum(sigma, 1e-9))  # Avoid numerical errors

            # EI formula
            z = (y_best - mu - xi) / sigma
            ei = (y_best - mu - xi) * scipy.stats.norm.cdf(z) + sigma * scipy.stats.norm.pdf(z)
            return -ei  # Minimize negative EI

        
        while expended_budget < problem.factors["budget"]:
            # low-fidelity dataset names being integers from 0 to level-1
            sm.set_training_values(all_x_l_set, all_y_l_set, name=0)
            # high-fidelity dataset without name
            sm.set_training_values(all_x_h_set, all_y_h_set)

            # train the model
            with open('nul' if sys.platform == 'win32' else '/dev/null', 'w') as f, contextlib.redirect_stdout(f):
                sm.train()
            
            # Create x0 and bounds
            bounds = [(xlimits[i, 0], xlimits[i, 1]) for i in range(problem.dim)]
            x0 = np.random.uniform(xlimits[:, 0], xlimits[:, 1], size=problem.dim)

            # Find the next query point using EI
            with open('nul' if sys.platform == 'win32' else '/dev/null', 'w') as f, contextlib.redirect_stdout(f):
                #res = minimize(expected_improvement, x0=x0, args=(sm, best_solution.objectives_mean), bounds=bounds, method="L-BFGS-B")
                res = minimize(expected_improvement, x0=list(best_solution.x), args=(sm, best_solution.objectives_mean), bounds=bounds, method="L-BFGS-B")
                
            new_x = res.x
            
            # Create new solution based on new x
            new_solution = self.create_new_solution(tuple(new_x), problem)

            # Use r simulated observations to estimate the objective value.
            problem.simulate(new_solution, r)
            expended_budget += r + costs[1]*r

            if new_solution.objectives_mean != np.inf:
                for i in range(r):
                    all_x_l_set = np.vstack((all_x_l_set, new_x))
                    all_x_h_set = np.vstack((all_x_h_set, new_x))
                
                y_h_set = new_solution.objectives[new_solution.objectives != 0]  # Remove zeros
                y_h_set = y_h_set.reshape(-1, 1)  # Ensure it's a column vector
                all_y_h_set = np.vstack((all_y_h_set,y_h_set))

                y_l_set = new_solution.lf_f[new_solution.lf_f != 0]  # Remove zeros
                y_l_set = y_l_set.reshape(-1, 1)  # Ensure it's a column vector
                all_y_l_set = np.vstack((all_y_l_set,y_l_set))
            
            if (-1 * problem.minmax[0] * new_solution.objectives_mean < -1 * problem.minmax[0] * best_solution.objectives_mean):
                best_solution = new_solution
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)
        return recommended_solns, intermediate_budgets

''' 
# low-fidelity dataset names being integers from 0 to level-1
sm.set_training_values(all_x_l_set, all_y_l_set, name=0)
# high-fidelity dataset without name
sm.set_training_values(all_x_h_set, all_y_h_set)

with open('nul' if sys.platform == 'win32' else '/dev/null', 'w') as f, contextlib.redirect_stdout(f):
    sm._new_train_iteration(lvl=0)
    sm._new_train_iteration(lvl=1)
    sm._reinterpolate(lvl=0)
    sm._reinterpolate(lvl=0)
'''