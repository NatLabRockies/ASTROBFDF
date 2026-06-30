"""
Summary
-------
"""
from __future__ import annotations

import numpy as np
from simopt.base import Solver, Problem, Solution
from math import log, ceil, sqrt


class BFSAG(Solver):
    """The Nelder-Mead algorithm, which maintains a simplex of points that moves around the feasible
    region according to certain geometric operations: reflection, expansion,
    contraction, and shrinking.

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
    def __init__(self, name: str = "BFSAG", fixed_factors: dict = {}):
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
            "alpha": {
                "description": "step size",
                "datatype": float,
                "default": 0.5  # Changing the step size matters a lot.
            },
            "lr": {
                "description": "reflection coefficient > 0",
                "datatype": float,
                "default": 0.5
            },
            "N": {
                "description": "expansion coefficient > 1",
                "datatype": int,
                "default": 100
            },
            "Nl": {
                "description": "contraction coefficient > 0, < 1",
                "datatype": int,
                "default": 50
            },
            "Nh": {
                "description": "shrink factor > 0, < 1",
                "datatype": int,
                "default": 30
            },
            "sensitivity": {
                "description": "shrinking scale for bounds",
                "datatype": float,
                "default": 10**(-7)
            },
            "costs_mf": {
                "description": "costs for multi-fidelity functions",
                "datatype": list,
                "default": [1, 0.3]
            },
            "coefficient": {
                "description": "coefficient for the step size",
                "datatype": float,
                "default": 1.0
            }
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "alpha": self.check_alpha,
            "sensitivity": self.check_sensitivity,
            "initial_spread": self.check_initial_spread
        }
        super().__init__(fixed_factors)

    def check_alpha(self):
        return self.factors["alpha"] > 0

    def check_sensitivity(self):
        return self.factors["sensitivity"] > 0

    def check_initial_spread(self):
        return self.factors["initial_spread"] > 0

    def solve(self, problem: "Problem") -> tuple[list["Solution"], list[int]]:
        """
        Run a single macroreplication of the BF-SAG solver on a problem using finite differences
        and mrg32k3a random number generator.

        Returns
        -------
        recommended_solns : list of Solution objects
            list of recommended solutions throughout the budget
        intermediate_budgets : list of ints
            list of intermediate budgets when recommended solutions changes
        """
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0

        # Hyperparameters and factors
        lr = self.factors["lr"]
        N = self.factors["N"]
        Nl = self.factors["Nl"]
        Nh = self.factors["Nh"]
        costs = self.factors["costs_mf"]
        coefficient = self.factors["coefficient"]

        # Default random number generator.
        find_next_soln_rng = self.rng_list[1]
        # Generate many dummy solutions without replication only to find a reasonable maximum radius
        dummy_solns = []
        for i in range(1000 * problem.dim):    
            dummy_solns += [problem.get_random_solution(find_next_soln_rng)]

        # Range for each dimension is calculated and compared with box constraints range if given 
        # TODO: just use box constraints range if given

        stepsize_max_arr = []
        for i in range(problem.dim):
            stepsize_max_arr += [min(max([sol[i] for sol in dummy_solns])-min([sol[i] for sol in dummy_solns]), 
                                  problem.upper_bounds[0] - problem.lower_bounds[0])]          
        
        # TODO: update this so that it could be used for problems with decision variables at varying scales!
        stepsize_max = max(stepsize_max_arr)

        if not (np.isinf(problem.upper_bounds).any() or np.isinf(problem.lower_bounds).any()):
            lr = lr
        else:
            lr = coefficient * 10 ** (ceil(log(stepsize_max * 2, 10) - 1) / problem.dim)

        lower_bound = np.array(problem.lower_bounds)
        upper_bound = np.array(problem.upper_bounds)

        # Initialize at initial solution
        theta = np.array(problem.factors["initial_solution"])
        new_solution = self.create_new_solution(tuple(theta), problem)
        problem.simulate(new_solution, max(Nl,Nh))
        expended_budget += (Nl+Nh) * (costs[0] + costs[1])

        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)

        # Initialize gradient memory
        d = np.zeros((N, problem.dim))
        rng = self.rng_list[1]

        def draw_without_replacement(rng, N, k):
            """ Draw k unique indices from range(N) using mrg32k3a RNG. """
            indices = list(range(N))
            for i in range(N - 1, 0, -1):
                j = int(rng.random() * (i + 1))
                indices[i], indices[j] = indices[j], indices[i]
            return indices[:k]

        while expended_budget < problem.factors["budget"]:
            d_avg = np.zeros(problem.dim)
            n_l = 0
            n_h = 0
            solution_set_l = []
            solution_set_h = []

            new_x = new_solution.x
            indices = draw_without_replacement(rng, N, Nl + Nh)
            t_l = indices[:Nl]
            t_h = indices[Nl:]

            for i in range(N):
                # Prepare BdsCheck for finite difference
                forward = np.isclose(new_x, lower_bound, atol=self.factors["sensitivity"]).astype(int)
                backward = np.isclose(new_x, upper_bound, atol=self.factors["sensitivity"]).astype(int)
                BdsCheck = np.subtract(forward, backward)

                if i in t_l:
                    n_l += 1
                    d[i], solution_set_l = self.finite_diff_low_fidelity(new_solution, BdsCheck, problem, n_l, solution_set_l)
                    expended_budget += (2 * problem.dim - np.sum(BdsCheck != 0)) * 1 * costs[1]
                elif i in t_h:  # high-fidelity
                    n_h += 1
                    d[i], solution_set_h = self.finite_diff_high_fidelity(new_solution, BdsCheck, problem, n_h, solution_set_h)
                    expended_budget += (2 * problem.dim - np.sum(BdsCheck != 0)) * 1
                else:
                    if i == 0:
                        d[i] = np.zeros(problem.dim)
                    else:
                        d[i] = d[i-1]  # Reuse previous gradient # Weird

                d_avg += d[i] 
            d_avg = d_avg/N
            
            # Perform parameter update
            theta = theta - lr * d_avg
            theta = np.clip(theta, lower_bound, upper_bound)

            new_solution = self.create_new_solution(tuple(theta), problem)
            problem.simulate(new_solution, max(Nl,Nh))
            expended_budget += (Nl+Nh) * (costs[0] + costs[1])

            recommended_solns.append(new_solution)
            intermediate_budgets.append(expended_budget)

        return recommended_solns, intermediate_budgets

    # Finite difference for approximating gradients.
    def finite_diff_low_fidelity(self, new_solution, BdsCheck, problem, n_l, solution_set_l):
        alpha = self.factors['alpha']
        lower_bound = problem.lower_bounds
        upper_bound = problem.upper_bounds
        fn = -1 * problem.minmax[0] * new_solution.lf_f[n_l-1]
        new_x = new_solution.x
        # Store values for each dimension.
        FnPlusMinus = np.zeros((problem.dim, 3))
        grad = np.zeros(problem.dim)

        for i in range(problem.dim):
            # Initialization.
            x1 = list(new_x)
            x2 = list(new_x)
            # Forward stepsize.
            steph1 = alpha
            # Backward stepsize.
            steph2 = alpha

            # Check variable bounds.
            if x1[i] + steph1 > upper_bound[i]:
                steph1 = np.abs(upper_bound[i] - x1[i])
            if x2[i] - steph2 < lower_bound[i]:
                steph2 = np.abs(x2[i] - lower_bound[i])

            # Decide stepsize.
            # Central diff.
            if BdsCheck[i] == 0:
                FnPlusMinus[i, 2] = min(steph1, steph2)
                x1[i] = x1[i] + FnPlusMinus[i, 2]
                x2[i] = x2[i] - FnPlusMinus[i, 2]
            # Forward diff.
            elif BdsCheck[i] == 1:
                FnPlusMinus[i, 2] = steph1
                x1[i] = x1[i] + FnPlusMinus[i, 2]
            # Backward diff.
            else:
                FnPlusMinus[i, 2] = steph2
                x2[i] = x2[i] - FnPlusMinus[i, 2]

            if len(solution_set_l) == 2*problem.dim: 
                x1_solution = solution_set_l[2*i]
            else:
                x1_solution = self.create_new_solution(tuple(x1), problem)
            if BdsCheck[i] != -1:
                problem.simulate_up_to([x1_solution], n_l)
                fn1 = -1 * problem.minmax[0] * x1_solution.lf_f[n_l-1]
                # First column is f(x+h,y).
                FnPlusMinus[i, 0] = fn1
            if len(solution_set_l) < 2*problem.dim: solution_set_l.append(x1_solution)

            if len(solution_set_l) == 2*problem.dim: 
                x2_solution = solution_set_l[2*i+1]
            else:
                x2_solution = self.create_new_solution(tuple(x2), problem)
            if BdsCheck[i] != 1:
                problem.simulate_up_to([x2_solution], n_l)
                fn2 = -1 * problem.minmax[0] * x2_solution.lf_f[n_l-1]
                # Second column is f(x-h,y).
                FnPlusMinus[i, 1] = fn2
            if len(solution_set_l) < 2*problem.dim: solution_set_l.append(x2_solution)

            # Calculate gradient.
            if BdsCheck[i] == 0:
                grad[i] = (fn1 - fn2) / (2 * FnPlusMinus[i, 2])
            elif BdsCheck[i] == 1:
                grad[i] = (fn1 - fn) / FnPlusMinus[i, 2]
            elif BdsCheck[i] == -1:
                grad[i] = (fn - fn2) / FnPlusMinus[i, 2]

        return grad, solution_set_l
    
    # Finite difference for approximating gradients.
    def finite_diff_high_fidelity(self, new_solution, BdsCheck, problem, n_h, solution_set_h):
        alpha = self.factors['alpha']
        lower_bound = problem.lower_bounds
        upper_bound = problem.upper_bounds
        fn = -1 * problem.minmax[0] * new_solution.objectives[n_h-1]
        new_x = new_solution.x
        # Store values for each dimension.
        FnPlusMinus = np.zeros((problem.dim, 3))
        grad = np.zeros(problem.dim)

        for i in range(problem.dim):
            # Initialization.
            x1 = list(new_x)
            x2 = list(new_x)
            # Forward stepsize.
            steph1 = alpha
            # Backward stepsize.
            steph2 = alpha

            # Check variable bounds.
            if x1[i] + steph1 > upper_bound[i]:
                steph1 = np.abs(upper_bound[i] - x1[i])
            if x2[i] - steph2 < lower_bound[i]:
                steph2 = np.abs(x2[i] - lower_bound[i])

            # Decide stepsize.
            # Central diff.
            if BdsCheck[i] == 0:
                FnPlusMinus[i, 2] = min(steph1, steph2)
                x1[i] = x1[i] + FnPlusMinus[i, 2]
                x2[i] = x2[i] - FnPlusMinus[i, 2]
            # Forward diff.
            elif BdsCheck[i] == 1:
                FnPlusMinus[i, 2] = steph1
                x1[i] = x1[i] + FnPlusMinus[i, 2]
            # Backward diff.
            else:
                FnPlusMinus[i, 2] = steph2
                x2[i] = x2[i] - FnPlusMinus[i, 2]

                
            if len(solution_set_h) == 2*problem.dim: 
                x1_solution = solution_set_h[2*i+1]
            else:
                x1_solution = self.create_new_solution(tuple(x1), problem)
            if BdsCheck[i] != -1:
                problem.simulate_up_to([x1_solution], n_h)
                fn1 = -1 * problem.minmax[0] * x1_solution.objectives[n_h-1]
                # First column is f(x+h,y).
                FnPlusMinus[i, 0] = fn1
            if len(solution_set_h) < 2*problem.dim: solution_set_h.append(x1_solution)

            
            if len(solution_set_h) == 2*problem.dim: 
                x2_solution = solution_set_h[2*i+1]
            else:
                x2_solution = self.create_new_solution(tuple(x2), problem)
            if BdsCheck[i] != 1:
                problem.simulate_up_to([x2_solution], n_h)
                fn2 = -1 * problem.minmax[0] * x2_solution.objectives[n_h-1]
                # Second column is f(x-h,y).
                FnPlusMinus[i, 1] = fn2
            if len(solution_set_h) < 2*problem.dim: solution_set_h.append(x2_solution)

            # Calculate gradient.
            if BdsCheck[i] == 0:
                grad[i] = (fn1 - fn2) / (2 * FnPlusMinus[i, 2])
            elif BdsCheck[i] == 1:
                grad[i] = (fn1 - fn) / FnPlusMinus[i, 2]
            elif BdsCheck[i] == -1:
                grad[i] = (fn - fn2) / FnPlusMinus[i, 2]

        return grad, solution_set_h