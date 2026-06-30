"""
Summary
-------
Simulate a synthetic problem with a deterministic objective function
evaluated with noise.
"""
import numpy as np

from ..base import Model, Problem


class Example(Model):
    """
    A model that is a deterministic function evaluated with noise.

    Attributes
    ----------
    name : string
        name of model
    n_rngs : int
        number of random-number generators used to run a simulation replication
    n_responses : int
        number of responses (performance measures)
    factors : dict
        changeable factors of the simulation model
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)
    check_factor_list : dict
        switch case for checking factor simulatability

    Arguments
    ---------
    fixed_factors : dict
        fixed_factors of the simulation model

    See also
    --------
    base.Model
    """
    def __init__(self, fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = "EXAMPLE"
        self.n_rngs = 1
        self.n_responses = 1
        self.factors = fixed_factors
        self.specifications = {
            "x": {
                "description": "point to evaluate from 0 to 1",
                "datatype": tuple,
                "default": (0, 0)
            },
            "sigma_h": {
                "description": "high fidelity oracle s.d.",
                "datatype": float,
                "default": 50.0
            },
            "sigma_l": {
                "description": "low fidelity oracle s.d.",
                "datatype": float,
                "default": 50.0
            },
            "mean_h": {
                "description": "high fidelity oracle mean",
                "datatype": float,
                "default": 0.0
            },
            "mean_l": {
                "description": "low fidelity oracle mean",
                "datatype": float,
                "default": 0.0
            },
            "target_fidelity": {
                "description": "target fidelity level",
                "datatype": str,
                "default": "HF"
            },
            "deterministic_hfn": {
                "description": "High-fidelity deterministic function",
                "datatype": callable,
                "default": self.default_hf_function
            },
            "deterministic_lfn": {
                "description": "Low-fidelity deterministic function",
                "datatype": callable,
                "default": self.default_lf_function
            },
            "noise_rule_hf": {
                "description": "Rule to combine HF noise",
                "datatype": callable,
                "default": self.default_rule_hf
            },
            "noise_rule_lf": {
                "description": "Rule to combine LF noise",
                "datatype": callable,
                "default": self.default_rule_lf
            }

        }
        self.check_factor_list = {
            "x": self.check_x
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def default_hf_function(self, x):
        return x[0]**2
    
    def default_lf_function(self, x):
        return x[0]
    
    def default_rule_hf(self, det, hf_noise, lf_noise):
        return det + hf_noise
    
    def default_rule_lf(self, det, hf_noise, lf_noise):
        return det + (hf_noise + lf_noise)/2

    def check_x(self):
        # Assume f(x) can be evaluated at any x in R^d.
        return True

    def check_simulatable_factors(self):
        return True

    def replicate(self, rng_list):
        """
        Evaluate a deterministic function f(x) with stochastic noise.

        Arguments
        ---------
        rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
            "est_f(x)" = f(x) evaluated with stochastic noise
        """
        # Designate random number generator for stochastic noise.
        noise_rng = rng_list[0]
        x = np.array(self.factors["x"])
        hf_sigma2 = self.factors["sigma_h"]
        lf_sigma2 = self.factors["sigma_l"]
        hf_mean = self.factors["mean_h"]
        lf_mean = self.factors["mean_l"]
        noise = noise_rng.normalvariate(hf_mean,hf_sigma2)
        noise_lf = noise_rng.normalvariate(lf_mean,lf_sigma2)
        target_fidelity = self.factors.get("target_fidelity", "HF")

        deterministic_hfunc = self.factors.get(
            "deterministic_hfn",
            self.specifications["deterministic_hfn"]["default"]
        )
        deterministic_lfunc = self.factors.get(
            "deterministic_lfn",
            self.specifications["deterministic_lfn"]["default"]
        )

        rule_hf = self.factors.get("noise_rule_hf", self.specifications["noise_rule_hf"]["default"])
        rule_lf = self.factors.get("noise_rule_lf", self.specifications["noise_rule_lf"]["default"])

        if target_fidelity in ["HF"]:
            deterministic_hfn = deterministic_hfunc(x)
            hfn_eval_at_x = float(rule_hf(deterministic_hfn, noise, noise_lf))
            lfn_eval_at_x = 0.0  # LF not evaluated
        else:  # LF only
            deterministic_lfn = deterministic_lfunc(x)
            hfn_eval_at_x = 0.0  # HF not evaluated
            lfn_eval_at_x = float(rule_lf(deterministic_lfn, noise, noise_lf))
        
        # Compose responses and gradients.
        responses = {"est_hf(x)": hfn_eval_at_x, "est_lf(x)": lfn_eval_at_x}
        gradients = {}
        return responses, gradients


"""
Summary
-------
Minimize f(x).
"""


class ExampleProblem(Problem):
    """
    Base class to implement simulation-optimization problems.

    Attributes
    ----------
    name : string
        name of problem
    dim : int
        number of decision variables
    n_objectives : int
        number of objectives
    n_stochastic_constraints : int
        number of stochastic constraints
    minmax : tuple of int (+/- 1)
        indicator of maximization (+1) or minimization (-1) for each objective
    constraint_type : string
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : string
        description of variable types:
            "discrete", "continuous", "mixed"
    lower_bounds : tuple
        lower bound for each decision variable
    upper_bounds : tuple
        upper bound for each decision variable
    gradient_available : bool
        indicates if gradient of objective function is available
    optimal_value : tuple
        optimal objective function value
    optimal_solution : tuple
        optimal solution
    model : Model object
        associated simulation model that generates replications
    model_default_factors : dict
        default values for overriding model-level default factors
    model_fixed_factors : dict
        combination of overriden model-level factors and defaults
    model_decision_factors : set of str
        set of keys for factors that are decision variables
    rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
        list of RNGs used to generate a random initial solution
        or a random problem instance
    factors : dict
        changeable factors of the problem
            initial_solution : tuple
                default initial solution from which solvers start
            budget : int > 0
                max number of replications (fn evals) for a solver to take
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)

    Arguments
    ---------
    name : str
        user-specified name for problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See also
    --------
    base.Problem
    """
    def __init__(self, name="EXAMPLE-1", fixed_factors=None, model_fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        if model_fixed_factors is None:
            model_fixed_factors = {}
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "unconstrained"
        self.variable_type = "continuous"
        self.gradient_available = False
        self.model_default_factors = {}
        self.model_fixed_factors = {}
        self.model_decision_factors = {"x"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "initial solution",
                "datatype": tuple,
                "default": (0.5, 0.5)
            },
            "budget": {
                "description": "max # of replications for a solver to take",
                "datatype": int,
                "default": 100
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        self.dim = len(self.factors["initial_solution"])
        self.lower_bounds = (float('-inf'),) * self.dim
        self.upper_bounds = (float('inf'),) * self.dim
        # Instantiate model with fixed factors and overwritten defaults.
        self.model = Example(self.model_fixed_factors)
        self.optimal_value = None    # Change if f is changed.
        self.optimal_solution = None # Change if f is changed.


    def vector_to_factor_dict(self, vector):
        """
        Convert a vector of variables to a dictionary with factor keys

        Arguments
        ---------
        vector : tuple
            vector of values associated with decision variables

        Returns
        -------
        factor_dict : dictionary
            dictionary with factor keys and associated values
        """
        factor_dict = {
            "x": vector[:]
        }
        return factor_dict

    def factor_dict_to_vector(self, factor_dict):
        """
        Convert a dictionary with factor keys to a vector
        of variables.

        Arguments
        ---------
        factor_dict : dictionary
            dictionary with factor keys and associated values

        Returns
        -------
        vector : tuple
            vector of values associated with decision variables
        """
        vector = tuple(factor_dict["x"])
        return vector

    def response_dict_to_objectives(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of objectives.

        Arguments
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns
        -------
        objectives : tuple
            vector of objectives
        """
        objectives = (response_dict["est_hf(x)"],)
        return objectives

    def response_dict_to_lf_f(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of objectives.

        Arguments
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns
        -------
        objectives : tuple
            vector of objectives
        """
        lf_f = (response_dict["est_lf(x)"],)
        return lf_f

    def response_dict_to_stoch_constraints(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] <= 0

        Arguments
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns
        -------
        stoch_constraints : tuple
            vector of LHSs of stochastic constraint
        """
        stoch_constraints = None
        return stoch_constraints

    def deterministic_objectives_and_gradients(self, x):
        """
        Compute deterministic components of objectives for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_objectives : tuple
            vector of deterministic components of objectives
        det_objectives_gradients : tuple
            vector of gradients of deterministic components of objectives
        """
        det_objectives = (0,)
        det_objectives_gradients = ((0,) * self.dim,)
        return det_objectives, det_objectives_gradients

    def deterministic_stochastic_constraints_and_gradients(self, x):
        """
        Compute deterministic components of stochastic constraints
        for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_stoch_constraints : tuple
            vector of deterministic components of stochastic constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of
            stochastic constraints
        """
        det_stoch_constraints = None
        det_stoch_constraints_gradients = None
        return det_stoch_constraints, det_stoch_constraints_gradients

    def check_deterministic_constraints(self, x):
        """
        Check if a solution `x` satisfies the problem's deterministic
        constraints.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        satisfies : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        # Superclass method will check box constraints.
        # Can add other constraints here.
        return super().check_deterministic_constraints(x)

    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : mrg32k3a.mrg32k3a.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        #x = tuple([rand_sol_rng.uniform(0.25, 0.75) for _ in range(self.dim)])
        x = tuple(rand_sol_rng.mvnormalvariate(mean_vec=np.zeros(self.dim), cov=np.eye(self.dim), factorized=False))
        return x
