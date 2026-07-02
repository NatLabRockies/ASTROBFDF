"""
Summary
-------
ASTRO-BFDF for bi-fidelity simulation optimization problems.
Note that MFMC implies BFMC in this implementation since we focus on the bi-fidelity case.
"""
from numpy.linalg import pinv
from numpy.linalg import norm
import numpy as np
from math import log, ceil, sqrt
import warnings
from scipy.optimize import NonlinearConstraint, LinearConstraint
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

from ..base import Solver


class ASTROBFDF(Solver):
    """The ASTRO-BFDF solver.

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
    def __init__(self, name="ASTROBFDF", fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = name
        self.objective_type = "single"
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.gradient_needed = False
        self.specifications = {
            "crn_across_solns": {
                "description": "use CRN across solutions",
                "datatype": bool,
                "default": True
            },            
            "crn_solver": {
                "description": "use CRN for the solver",
                "datatype": bool,
                "default": True
            },
            "cor_lf_hf": {
                "description": "correlation threshold between low-fidelity and high-fidelity",
                "datatype": float,
                "default": 0.3
            },
            "costs_mf": {
                "description": "costs for multi-fidelity functions",
                "datatype": list,
                "default": [1, 0.3]
            },
            "eta_1": {
                "description": "threshhold for a successful iteration",
                "datatype": float,
                "default": 0.1
            },
            "eta_2": {
                "description": "threshhold for a very successful iteration",
                "datatype": float,
                "default": 0.8
            },
            "gamma_1": {
                "description": "trust-region radius increase rate after a very successful iteration",
                "datatype": float,
                "default": 1.5
            },
            "gamma_2": {
                "description": "trust-region radius decrease rate after an unsuccessful iteration",
                "datatype": float,
                "default": 0.5
            },
            "lambda_min": {
                "description": "minimum sample size",
                "datatype": int,
                "default": 5
            },
            "easy_solve": {
                "description": "solve the subproblem approximately with Cauchy point",
                "datatype": bool,
                "default": False
            },
            "coef_delta_initial": {
                "description": "the coefficient for Delta_0",
                "datatype": float,
                "default": 1
            }            
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "eta_1": self.check_eta_1,
            "eta_2": self.check_eta_2,
            "gamma_1": self.check_gamma_1,
            "gamma_2": self.check_gamma_2,
            "lambda_min": self.check_lambda_min,
            "ps_sufficient_reduction": self.check_ps_sufficient_reduction
        }
        super().__init__(fixed_factors)
    
    def check_eta_1(self):
        return self.factors["eta_1"] > 0

    def check_eta_2(self):
        return self.factors["eta_2"] > self.factors["eta_1"]

    def check_gamma_1(self):
        return self.factors["gamma_1"] > 1

    def check_gamma_2(self):
        return (self.factors["gamma_2"] < 1 and self.factors["gamma_2"] > 0)

    def check_lambda_min(self):
        return self.factors["lambda_min"] > 2
    
    def check_ps_sufficient_reduction(self):
        return self.factors["ps_sufficient_reduction"] >= 0

    # generate the coordinate vector corresponding to the variable number v_no
    def get_coordinate_vector(self, size, v_no):
        arr = np.zeros(size)
        arr[v_no] = 1.0
        return arr

    def planerot(self, x):
        if x[1] != 0:
            r = np.linalg.norm(x)
            G = np.vstack((x, np.array([-x[1], x[0]]))) / r
            x = np.array([r, 0])
        else:
            G = np.eye(2)
        return (G, x)

    def check_null_space(self, radius, lidn, problem, AffVector):
        d = problem.dim
        GPoints = np.zeros((d, d))
        if lidn != 0:
            R = np.array(AffVector) / radius  # The points we have so far
            Q, R = np.linalg.qr(R.T, mode='complete')  # Get QR of points so far
            R = np.atleast_2d(R)
            R = np.hstack((R, np.zeros((d, d - R.shape[1]))))
        else:
            Q = np.eye(d)
            R = np.zeros((d, d))
        
        GPoints[0:d - lidn, :] = Q[:, lidn:d].T
        
        return GPoints
    
    def get_affined_vector(self, x_k, delta_k, problem, visited_pts_list):
        d = problem.dim
        index_exist = []
        datapoints_exist = []
        reuseid = []
        index_exist = []

        for i in range(len(visited_pts_list)):
            if (np.linalg.norm(np.array(visited_pts_list[i].x) - np.array(x_k)) - delta_k < 0) and (np.linalg.norm(np.array(visited_pts_list[i].x) - np.array(x_k))!=0):
                index_exist.append(i)

        solutions_exist = [visited_pts for visited_pts in visited_pts_list\
                       if (np.linalg.norm(np.array(visited_pts.x) - np.array(x_k)) - delta_k < 0) and (np.linalg.norm(np.array(visited_pts.x) - np.array(x_k))!=0)]
        datapoints_exist = [list(visited_pts.x) for visited_pts in solutions_exist]
        
        # QR factorization
        Q = np.eye(d)  # Get initial null space
        R = np.zeros((d, d))
        lidn = 0 # Numbers of design points already included
        theta1 = 0.001  # Pivot threshold for validity
        AffVector = []
        
        for i in range(len(datapoints_exist)):
            D = (np.array(datapoints_exist[i]) - np.array(x_k)) / delta_k
            proj = np.linalg.norm(D.dot(Q[:, lidn:d]))
            if (proj >= theta1):  # add this index to AffIn
                lidn = lidn + 1
                AffVector.append(datapoints_exist[i])
                reuseid.append(index_exist[i])
                if (lidn == d):
                    valid = True
                    return AffVector, valid, lidn, reuseid
                
                # Update QR factorization:
                R[:, lidn - 1] = D.dot(Q)  # add D
                for j in np.arange(d - 1, lidn - 1, -1):
                    G, R[[j - 1, j], lidn - 1] = self.planerot(R[[j - 1, j], lidn - 1])
                    Q[:, [j - 1, j]] = Q[:, [j - 1, j]].dot(G.T)

        # if you get out of this loop then nmp<n
        valid = False
        
        return AffVector, valid, lidn, reuseid

    def make_affined_vector(self, x_k, delta_k, problem, visited_pts_list):
        d = problem.dim # problem dimension

        # STEP 1: Find affinely independent points & check if fully linear
        AffVector, valid, lidn, reuseid = self.get_affined_vector(x_k, delta_k, problem, visited_pts_list)

        if (not valid):  # Model is not valid, check if poised
            GPoints = self.check_null_space(delta_k, lidn, problem, AffVector)
            if lidn < d:  # Need to include additional points to obtain a model
                for j in range(0, d - lidn):
                    GPoints[j, :] = GPoints[j, :] / np.linalg.norm(GPoints[j, :])  # Make unit length.
                    AffVector.append(x_k+ delta_k*GPoints[j, :])

        return AffVector, lidn, reuseid

    def designset_diagonalHessian(self, x_k, delta_k, problem, Y_affine):
        Y_design_set = [[x_k]]
        epsilon = 0.01

        for i in range(0, problem.dim):
            Y_design_set.append(np.array([Y_affine[i]]))   

        for i in range(0, problem.dim):
            another_direction = x_k - (delta_k * np.array(np.array(Y_affine[i])-x_k) / np.linalg.norm(np.array(Y_affine[i])-x_k)) 
            if sum(x_k) != 0:
                # block constraints
                if another_direction[i] <= problem.lower_bounds[i]:
                    another_direction[i] = problem.lower_bounds[i] + epsilon
                if another_direction[i] >= problem.upper_bounds[i]:
                    another_direction[i] = problem.upper_bounds[i] - epsilon
            
            Y_design_set.append(np.array([another_direction]))
 
        return Y_design_set

    def move_to_origin(self, Y_design_set):
        translation_vector = -np.array(Y_design_set[0][0])
        Z_design_set = [point + translation_vector for point in Y_design_set]
        return Z_design_set

    # compute the local model value with a linear interpolation with a diagonal Hessian
    def evaluate_model(self, x_k, q):
        X = [1]
        X = np.append(X, np.array(x_k))
        X = np.append(X, np.array(x_k) ** 2)
        return np.matmul(X, q)

    # compute the sample size based on adaptive sampling stopping rule using the optimality gap
    def get_stopping_time(self, k, sig2, delta, kappa, dim):        
        if kappa == 0: kappa = 1
        lambda_k = max(self.factors["lambda_min"], 2 * log(dim + .5, 10)) * max(log(k + 0.1, 10) ** (1.01), 1)
    
        N_k = ceil(max(lambda_k, lambda_k * sig2 / (kappa ** 2 * max(delta ** 2, delta ** 4))))

        return N_k
    
    def var_mf(self, n, sigma2, c, cov):
        k = len(n)  # Number of terms

        # First term
        var_mf_hat = sigma2[0] / n[0]
        # Summation term
        for i in range(1, k):  
            coef = (1 / n[i - 1]) - (1 / n[i])
            term = (c[i-1]**2 * sigma2[i]) - (2 * c[i-1] * cov[i-1])
            var_mf_hat += coef * term

        return var_mf_hat


    def mfmc_sample_size(self, new_solution, o1_solution_set, problem, budget_limit, expended_budget, kappa, k, delta):
        """
        Compute adaptive sample sizes using Bi-Fidelity Monte Carlo (BFMC) allocation. (BFAS in the paper)
        Although we have used the term MFMC throughout, this implementation focuses on the bi-fidelity case.
        

        This method determines whether to use BFMC or crude Monte Carlo sampling for a given candidate solution and dynamically allocates high- and low-fidelity simulation replications 
        under the remaining budget. It uses variance and covariance estimates to minimize the 
        variance of the MFMC estimator while respecting cost constraints.

        Parameters
        ----------
        new_solution : Solution
            Current candidate solution at which additional sampling is required.
        o1_solution_set : list of Solution
            Low-fidelity solution set, used to compute correlation and MFMC adjustment.
        problem : Problem
            Simulation optimization problem instance providing simulation oracles and budget info.
        budget_limit : int
            Total allowed simulation budget.
        expended_budget : int
            Current amount of budget already used.
        kappa : float
            Parameter controlling adaptive sampling based on optimality gap and trust-region size.
        k : int
            Current iteration number (used in stopping rules).
        delta : float
            Current trust-region radius (affects sampling precision requirement).

        Returns
        -------
        new_solution : Solution
            Updated candidate solution after additional HF sampling and MFMC adjustment.
        expended_budget : int
            Updated budget after sampling.
        o1_solution_set : list of Solution
            Updated LF solution set (after sampling LF points as needed).

        Notes
        -----
        - The method solves a **small nonlinear optimization problem** to compute the optimal 
        number of HF and LF replications to minimize MFMC estimator variance under cost constraints.
        - Two possible sampling strategies:
            1. **MFMC mode:** If the variance reduction achieved by MFMC outweighs crude MC, 
            allocate replications according to MFMC formula and adjust function estimates.
            2. **Crude MC mode:** If MFMC is not beneficial or cost-infeasible, increment HF 
            replications using adaptive stopping rules.
        - Stopping conditions:
            - Target variance level satisfied,
            - Budget exhausted,
            - Adaptive threshold based on `get_stopping_time()`.
        - Updates solution attributes:
            - `mfmc_mean`: MFMC-adjusted estimate,
            - `mfmc_n_rep`: number of HF samples used,
            - `mfmc_o1_rep`: number of LF samples used,
            - `mfmc_c1`: MFMC coefficient.
        - Uses `var_mf()` for variance estimation of the MFMC estimator.

        See Also
        --------
        var_mf : Computes variance of the MFMC estimator for given sample sizes and coefficients.
        get_stopping_time : Determines adaptive sampling thresholds.
        construct_model : Calls `mfmc_sample_size()` when constructing HF model points.
        """
        
        # default values
        costs = self.factors["costs_mf"]
        lambda_min = self.factors["lambda_min"]
        lambda_max = budget_limit - expended_budget
        lambda_k = max(lambda_min, 2 * log(problem.dim + .5, 10)) * max(log(k + 0.1, 10) ** (1.01), 1)

        pilot_run = ceil(max(lambda_min, min(.5 * problem.dim, lambda_max)) - 1)

        o1_id = next((i for i, obj in enumerate(o1_solution_set) if np.allclose(obj.x, new_solution.x, rtol=1e-9)), -1)

        # pilot run
        if pilot_run - new_solution.n_reps > 0:
            ss_hf = pilot_run - new_solution.n_rep
            problem.simulate(new_solution, ss_hf)
            expended_budget += ss_hf * costs[0]
        
        if pilot_run - o1_solution_set[o1_id].n_reps > 0:
            ss_lf = pilot_run - o1_solution_set[o1_id].n_reps
            problem.simulate(o1_solution_set[o1_id], ss_lf)
            expended_budget += costs[1] * ss_lf
    
        fn = new_solution.objectives_mean
        o_o1_cov = new_solution.hf_lf_cov

        new_solution.MFMC_use = True
        
        while True:
            predicted_N_k = self.get_stopping_time(k, new_solution.objectives_var, delta, kappa, problem.dim)
            
            def subproblem(t):
                return costs[0]*t[0] + costs[1]*t[1]
            
            # Problem setup
            # Define the function in lambda form for NonlinearConstraint
            var_mf_f = lambda t: self.var_mf([t[0], t[1]],  # sample sizes
                                                [new_solution.objectives_var, o1_solution_set[o1_id].lf_f_var],  # sigma values
                                                [t[2]],  # coefficient 
                                                [o_o1_cov]   # covariance values
                                                ) 

            nlc = NonlinearConstraint(var_mf_f, -np.inf, (kappa**2)*max(delta**2,delta**4)/lambda_k)
            
            A = [[1,-1, 0],[1,0,0],[0,1,0]]
            lc = LinearConstraint(A, \
                                    lb=[-np.inf, new_solution.n_reps, max(new_solution.n_reps, float(o1_solution_set[o1_id].n_reps))], \
                                    ub=[0, np.inf, np.inf])
            initial_point = [new_solution.n_reps,max(new_solution.n_reps, float(o1_solution_set[o1_id].n_reps)), 1]
            
            solve_subproblem_approximate = minimize(subproblem, initial_point, method='trust-constr', constraints=(lc,nlc))
            n0, n1, c1 = solve_subproblem_approximate.x
                
            if solve_subproblem_approximate.fun <= costs[0]*predicted_N_k:
                # use MFMC
                new_solution.MFMC_use = True
                
                if new_solution.n_reps > o1_solution_set[o1_id].n_reps:
                    ss_lf = new_solution.n_reps - o1_solution_set[o1_id].n_reps + 1
                    problem.simulate(o1_solution_set[o1_id], ss_lf)
                    expended_budget += ss_lf * costs[1]

                # check whetehr we can finish the sampling
                fn = new_solution.objectives_mean - c1*new_solution.lf_f_mean  + c1*o1_solution_set[o1_id].lf_f_mean 
            
                if self.var_mf([new_solution.n_reps, o1_solution_set[o1_id].n_reps],
                               [new_solution.objectives_var, o1_solution_set[o1_id].lf_f_var],
                                [c1],[o_o1_cov]) \
                    <= (kappa**2)*max(delta**2,delta**4)/lambda_k or expended_budget > budget_limit:
                    break
                
                if new_solution.n_reps >= ceil(n0)-1:
                    problem.simulate(o1_solution_set[o1_id], ceil(1/costs[1]))
                    expended_budget += ceil(1/costs[1])*costs[1]
                        
                    fn = new_solution.objectives_mean - c1*new_solution.lf_f_mean + c1*o1_solution_set[o1_id].lf_f_mean 

                else:
                    problem.simulate(new_solution, 1)
                    expended_budget += 1 * costs[0]

                    fn = new_solution.objectives_mean - c1*new_solution.lf_f_mean + c1*o1_solution_set[o1_id].lf_f_mean
                    
                    # update covariance terms based on increasing sample sizes for the HF function
                    o_o1_cov = new_solution.hf_lf_cov
                    
            else:
                # use the crude MC - increase n
                new_solution.MFMC_use = False
                fn = new_solution.objectives_mean
                sig2 = new_solution.objectives_var
                
                if new_solution.n_reps >= self.get_stopping_time(k, sig2, delta, kappa, problem.dim) or \
                    new_solution.n_reps >= lambda_max or expended_budget >= budget_limit:
                    break

                problem.simulate(new_solution, 1)
                expended_budget += 1 * costs[0]
                    
            if expended_budget > budget_limit:
                break
        
        mfmc_var = self.var_mf([new_solution.n_reps, o1_solution_set[o1_id].n_reps],
                               [new_solution.objectives_var, o1_solution_set[o1_id].lf_f_var],
                                [c1],[o_o1_cov])
        
        cmc_var = new_solution.objectives_var

        if mfmc_var < cmc_var:
            fn = new_solution.objectives_mean - c1*new_solution.lf_f_mean  + c1*o1_solution_set[o1_id].lf_f_mean 

        # Save new variables for MFMC
        new_solution.mfmc_mean = fn
        new_solution.mfmc_n_rep = new_solution.n_reps
        new_solution.mfmc_o1_rep = o1_solution_set[o1_id].n_reps
        new_solution.mfmc_c1 = c1

        return new_solution, expended_budget, o1_solution_set

    # construct the "qualified" local model for each iteration k with the center point x_k
    # reconstruct with new points in a shrunk trust-region if the model fails the criticality condition
    # the criticality condition keeps the model gradient norm and the trust-region size in lock-step
    def construct_model(self, x_k, delta_k, k, Y, problem, expended_budget, kappa, new_solution, visited_pts_list, 
                        o1_solution_set, io, low_or_high):
        """
        Construct a local surrogate model within the trust region.

        This method builds a fully linear quadratic model using interpolation points and simulation evaluations. 
        The model approximates either the high-fidelity or low-fidelity function (or both) depending on the stage of the algorithm 
        and is used for solving the trust-region subproblem.

        Parameters
        ----------
        x_k : tuple
            Current center of the trust region (incumbent solution).
        delta_k : list
            Trust-region radius for model construction. 
            If `io='inner'`, uses delta for LF optimization; otherwise, uses HF delta.
        k : int
            Current iteration number.
        Y : list
            Current interpolation set (may be updated according to the algorithm step).
        problem : Problem
            Simulation optimization problem instance.
        expended_budget : int
            Budget consumed so far; updated as new simulations are performed.
        kappa : float
            Hyperparameter used in adaptive sampling.
        new_solution : Solution
            Current incumbent solution with evaluated function values and variance estimates.
        visited_pts_list : list of Solution
            Previously evaluated HF solutions for reuse in the interpolation set.
        o1_solution_set : list of Solution
            Low-fidelity visited solution set.
        io : str
            Indicates model construction context:
            - 'inner': build LF-based model (used in low-fidelity subproblem),
            - 'outer': build HF-based model or LF-based model when alpha_k < alpha_th.
        low_or_high : int
            Specifies which fidelity is being modeled:
            - 0: high-fidelity,
            - 1: low-fidelity.

        Returns
        -------
        fval : list of float
            Function estimates for interpolation points.
        Y : list
            Updated interpolation set.
        q : ndarray
            Coefficients of the local quadratic model (constant, linear, and diagonal terms).
        grad : ndarray
            Gradient vector of the surrogate model.
        Hessian : ndarray
            Diagonal approximation of the Hessian of the surrogate model.
        expended_budget : int
            Updated simulation budget after sampling additional design points.
        visited_pts_list : list of Solution
            Updated visited HF solution set.
        o1_solution_set : list of Solution
            Updated visited LF solution set.

        Notes
        -----
        - The interpolation set consists of 2d+1 points: the center point and symmetric perturbations along coordinate directions (or affine-independent points if available).
        - When insufficient design points exist, additional points are generated to ensure affine independence and validity of the model.
        - Sampling strategy:
            - For HF models (low_or_high=0), adaptive sampling with MFMC is applied.
            - For LF models (low_or_high=1), LF evaluations are sampled with CRN and adaptive rules.
        - This method enforces box constraints when generating new design points.
        - The coefficients (q, grad, Hessian) are obtained by solving a least-squares system using the pseudo-inverse of the interpolation matrix.

        See Also
        --------
        get_model_coefficients : Computes model coefficients from interpolation points.
        solve_subproblem : Uses the constructed model to propose the next candidate solution.
        iterate : Calls `construct_model()` to build HF and LF models each iteration.
        """    
        costs = self.factors["costs_mf"]
        budget = problem.factors["budget"]
        lambda_max = budget - expended_budget
        MFMC_use = True
        fval = []

        if io == 'inner':
            delta_k = delta_k[low_or_high]
        else:
            delta_k = delta_k[0]

        if low_or_high == 0:
            Y_affine, lidn, reuseid = self.make_affined_vector(x_k, delta_k, problem, visited_pts_list) # Linearly independent d number of design points selection
            Y = self.designset_diagonalHessian(x_k, delta_k, problem, Y_affine)
            Z = self.move_to_origin(Y)
        elif io == 'outer' and low_or_high == 1:
            Z = self.move_to_origin(Y)
        else:
            # Construct the interpolation set
            Y = self.get_coordinate_basis_interpolation_points(x_k, delta_k, problem)
            Z = self.get_coordinate_basis_interpolation_points(np.zeros(problem.dim), delta_k, problem)

        # Evaluate the function estimate for the interpolation points
        # High-fidleity function estimate
        if low_or_high == 0:
            for i in range(2 * problem.dim + 1):
                # for x_0, we don't need to simulate the new solution
                if (k == 1) and (i == 0):
                    if new_solution.MFMC_use == True:
                        # use mfmc
                        fval.append(-1 * problem.minmax[0] * new_solution.mfmc_mean)
                        CRN_n, CRN_n1, CRN_c1 = (new_solution.mfmc_n_rep, new_solution.mfmc_o1_rep, new_solution.mfmc_c1)
                    else:
                        # use crude mc
                        MFMC_use = False
                        fval.append(-1 * problem.minmax[0] * new_solution.objectives_mean)
                        CRN_crude_n = new_solution.n_reps

                # reuse the replications for x_k (center point, i.e., the incumbent solution) 
                elif (i == 0):
                    new_solution, expended_budget, o1_solution_set = \
                        self.mfmc_sample_size(new_solution, o1_solution_set, problem, budget, expended_budget, kappa, k, delta_k)
                    
                    if new_solution.MFMC_use == True:
                        fval.append(-1 * problem.minmax[0] * new_solution.mfmc_mean)
                        
                        CRN_n, CRN_n1, CRN_c1 = (new_solution.mfmc_n_rep, new_solution.mfmc_o1_rep, new_solution.mfmc_c1)
                    else:
                        # use crude mc
                        MFMC_use = False
                        fval.append(-1 * problem.minmax[0] * new_solution.objectives_mean)
                        
                        CRN_crude_n = new_solution.n_reps

                # else if reuse design points, reuse the replications
                elif i <= lidn:
                    if MFMC_use == True:
                        o1_id = next((j for j, obj in enumerate(o1_solution_set) if np.allclose(obj.x, visited_pts_list[reuseid[i-1]].x, rtol=1e-9)), -1)

                        reuse_n = visited_pts_list[reuseid[i-1]].n_reps
                        
                        if CRN_n-reuse_n > 0:
                            problem.simulate(visited_pts_list[reuseid[i-1]], CRN_n-reuse_n)
                            expended_budget += (CRN_n-reuse_n) * costs[0]
                        
                        if o1_id == -1:
                            o1_solution = self.create_new_solution(tuple(Y[i][0]), problem)
                            o1_solution_set.append(o1_solution)
                            problem.simulate(o1_solution,CRN_n1)
                            expended_budget += CRN_n1 * costs[1]
                            o1_n1_mean = o1_solution.lf_f_mean 
                        else:
                            reuse_n1 = o1_solution_set[o1_id].n_reps

                            if CRN_n1 -reuse_n1 > 0:
                                problem.simulate(o1_solution_set[o1_id],CRN_n1 -reuse_n1)
                                expended_budget += (CRN_n1 -reuse_n1)*costs[1]
 
                            o1_n1_mean = o1_solution_set[o1_id].lf_f_mean 
                        
                        # MFMC                        
                        visited_pts_list[reuseid[i-1]].mfmc_n_rep = CRN_n
                        visited_pts_list[reuseid[i-1]].mfmc_o1_rep = CRN_n1
                        visited_pts_list[reuseid[i-1]].mfmc_c1 = CRN_c1
                        visited_pts_list[reuseid[i-1]].mfmc_mean = visited_pts_list[reuseid[i-1]].objectives_mean - CRN_c1*visited_pts_list[reuseid[i-1]].lf_f_mean \
                                                                + CRN_c1*o1_n1_mean
                        
                        fval.append(-1 * problem.minmax[0] * visited_pts_list[reuseid[i-1]].mfmc_mean)

                    else: # use CMC
                        if CRN_crude_n-visited_pts_list[reuseid[i-1]].n_reps > 0:
                            ss_hf = CRN_crude_n-visited_pts_list[reuseid[i-1]].n_reps
                            problem.simulate(visited_pts_list[reuseid[i-1]], ss_hf)
                            expended_budget += ss_hf * costs[0]
                        fval.append(-1 * problem.minmax[0] * visited_pts_list[reuseid[i-1]].objectives_mean)
                        
                # for new points, run the simulation with pilot run
                else:
                    design_set_solution = self.create_new_solution(tuple(Y[i][0]), problem)
                    visited_pts_list.append(design_set_solution)
                    if MFMC_use == True:
                        problem.simulate(design_set_solution, CRN_n)

                        new_o1_soln = self.create_new_solution(tuple(Y[i][0]), problem)
                        problem.simulate(new_o1_soln, CRN_n1)
                        o1_solution_set.append(new_o1_soln)

                        expended_budget += CRN_n*costs[0] + CRN_n1*costs[1]
                        
                        design_set_solution.mfmc_n_rep = CRN_n
                        design_set_solution.mfmc_o1_rep = CRN_n1
                        design_set_solution.mfmc_c1 = CRN_c1
                        design_set_solution.mfmc_use = True
                        design_set_solution.mfmc_mean = design_set_solution.objectives_mean - CRN_c1*design_set_solution.lf_f_mean \
                                                                + CRN_c1*new_o1_soln.lf_f_mean
                        
                        fval.append(-1 * problem.minmax[0] * design_set_solution.mfmc_mean)
                    else:
                        problem.simulate(design_set_solution, CRN_crude_n)
                        expended_budget += CRN_crude_n * costs[0]

                        design_set_solution.mfmc_use = False

                        fval.append(-1 * problem.minmax[0] * design_set_solution.objectives_mean)
        
        elif io == 'outer' and low_or_high == 1:
            for i in range(2 * problem.dim + 1):
                if (i == 0):
                    soln_id = next((j for j, obj in enumerate(o1_solution_set) if np.allclose(obj.x, new_solution.x, rtol=1e-9)), -1)
                    sample_size = o1_solution_set[soln_id].n_reps
                    sig2 = o1_solution_set[soln_id].lf_f_var

                    # adaptive sampling
                    while True:
                        if sample_size >= self.get_stopping_time(k, sig2, delta_k, kappa, problem.dim) or \
                                sample_size >= lambda_max or expended_budget >= budget: #NOTE: expended_budget needs to be update at every iteration
                            break
                        problem.simulate(o1_solution_set[soln_id], 1)
                        expended_budget += costs[1]
                        sample_size += 1
                        sig2 = o1_solution_set[soln_id].lf_f_var

                    CRN_low = sample_size
                    fval.append(-1 * problem.minmax[0] * o1_solution_set[soln_id].lf_f_mean)
                else:
                    soln_id = next((j for j, obj in enumerate(o1_solution_set) if np.allclose(obj.x, tuple(Y[i][0]), rtol=1e-5)), -1)
                    
                    if soln_id == -1:
                        new_o1_soln = self.create_new_solution(tuple(Y[i][0]), problem)
                        problem.simulate(new_o1_soln, CRN_low)
                        o1_solution_set.append(new_o1_soln)

                        expended_budget += costs[1] * CRN_low

                        fval.append(-1 * problem.minmax[0] * o1_solution_set[soln_id].lf_f_mean)
                    else:
                        sample_size = o1_solution_set[soln_id].n_reps
                        if CRN_low > sample_size:
                            problem.simulate(o1_solution_set[soln_id], int(CRN_low-sample_size))
                            expended_budget += costs[1] * int(CRN_low-sample_size)

                        fval.append(-1 * problem.minmax[0] * o1_solution_set[soln_id].lf_f_mean)
                  
        # low_or_high == 1
        else:
            for i in range(2 * problem.dim + 1):
                if (i == 0):
                    #NOTE: Need to use BFMC for this case
                    # update the high-fidelity function estimate at center point which will be use for the sufficient reduction test
                    new_solution, expended_budget, o1_solution_set = \
                        self.mfmc_sample_size(new_solution, o1_solution_set, problem, budget, expended_budget, kappa, k, delta_k)

                    soln_id = next((j for j, obj in enumerate(o1_solution_set) if np.allclose(obj.x, new_solution.x, rtol=1e-9)), -1)
                    sample_size = o1_solution_set[soln_id].n_reps
                    sig2 = o1_solution_set[soln_id].lf_f_var

                    # adaptive sampling
                    while True:
                        if sample_size >= self.get_stopping_time(k, sig2, delta_k, kappa, problem.dim) or \
                                sample_size >= lambda_max or expended_budget >= budget: #NOTE: expended_budget needs to be update at every iteration
                            break
                        problem.simulate(o1_solution_set[soln_id], 1)
                        expended_budget += costs[1]
                        sample_size += 1
                        sig2 = o1_solution_set[soln_id].lf_f_var

                    CRN_low = sample_size

                    fval.append(-1 * problem.minmax[0] * o1_solution_set[soln_id].lf_f_mean)
                # for new points, run the simulation with pilot run
                else:
                    design_set_solution = self.create_new_solution(tuple(Y[i][0]), problem)
                    
                    # use CRN
                    problem.simulate(design_set_solution, CRN_low)
                    expended_budget += CRN_low*costs[1]
                    fval.append(-1 * problem.minmax[0] * design_set_solution.lf_f_mean)

        # construct the model and obtain the model coefficients
        q, grad, Hessian = self.get_model_coefficients(Z, fval, problem)

        return fval, Y, q, grad, Hessian, expended_budget, visited_pts_list, o1_solution_set

    # compute the model coefficients using (2d+1) design points and their function estimates
    def get_model_coefficients(self, Y, fval, problem):
        M = []
        for i in range(0, 2 * problem.dim + 1):
            M.append(1)
            M[i] = np.append(M[i], np.array(Y[i]))
            M[i] = np.append(M[i], np.array(Y[i]) ** 2)

        q = np.matmul(pinv(M), fval)  # pinv returns the inverse of your matrix when it is available and the pseudo inverse when it isn't.
        grad = q[1:problem.dim + 1]
        grad = np.reshape(grad, problem.dim)
        Hessian = q[problem.dim + 1 : 2 * problem.dim + 1]
        Hessian = np.reshape(Hessian, problem.dim)
        return q, grad, Hessian

    # compute the interpolation points (2d+1) using the coordinate basis
    def get_coordinate_basis_interpolation_points(self, x_k, delta, problem):
        Y = [[x_k]]
        epsilon = 0.01
        for i in range(0, problem.dim):
            plus = Y[0] + delta * self.get_coordinate_vector(problem.dim, i)
            minus = Y[0] - delta * self.get_coordinate_vector(problem.dim, i)

            if sum(x_k) != 0:
                # block constraints
                if minus[0][i] <= problem.lower_bounds[i]:
                    minus[0][i] = problem.lower_bounds[i] + epsilon
                if plus[0][i] >= problem.upper_bounds[i]:
                    plus[0][i] = problem.upper_bounds[i] - epsilon

            Y.append(plus)
            Y.append(minus)
        return Y

    # compute the interpolation points (2d+1) using the rotated coordinate basis (reuse one design point)
    def get_rotated_basis_interpolation_points(self, x_k, delta, problem, rotate_matrix, reused_x):
        Y = [[x_k]]
        epsilon = 0.01
        for i in range(0, problem.dim):
            if i == 0:
                plus = [np.array(reused_x)]
            else:
                plus = Y[0] + delta * rotate_matrix[i]
            minus = Y[0] - delta * rotate_matrix[i]

            if sum(x_k) != 0:
                # block constraints
                for j in range(problem.dim):
                    if minus[0][j] <= problem.lower_bounds[j]:
                        minus[0][j] = problem.lower_bounds[j] + epsilon
                    elif minus[0][j] >= problem.upper_bounds[j]:
                        minus[0][j] = problem.upper_bounds[j] - epsilon
                    if plus[0][j] <= problem.lower_bounds[j]:
                        plus[0][j] = problem.lower_bounds[j] + epsilon
                    elif plus[0][j] >= problem.upper_bounds[j]:
                        plus[0][j] = problem.upper_bounds[j] - epsilon

            Y.append(plus)
            Y.append(minus)
        return Y
    
    def solve_subproblem(self, problem, corr_k, new_x, new_solution, fval, q, grad, Hessian, delta_k, \
                         expended_budget, visited_pts_list, o1_solution_set):  
        """
        Solve the trust-region subproblem based on the local surrogate model.

        Given a quadratic surrogate model constructed from simulation evaluations, 
        this method determines the next candidate solution by solving approximately the trust-region subproblem. 
        It evaluates the predicted versus actual improvement to compute the success ratio and determines whether the trust region should expand or shrink.

        Parameters
        ----------
        problem : Problem
            Simulation-optimization problem instance (provides simulation oracle, bounds, etc.).
        corr_k : float
            Current correlation parameter between low- and high-fidelity models.
        new_x : ndarray
            Current incumbent solution (center of the trust region).
        new_solution : Solution
            Current solution object representing the incumbent.
        fval : list of float
            Function estimates at the interpolation points used in the model.
        q : ndarray
            Coefficients of the local quadratic model (constant, linear, and diagonal terms).
        grad : ndarray
            Gradient vector of the surrogate model.
        Hessian : ndarray
            Diagonal approximation of the Hessian of the surrogate model.
        delta_k : float
            Current trust-region radius.
        expended_budget : int
            Budget consumed so far.
        visited_pts_list : list of Solution
            Solutions previously evaluated, used for bookkeeping.
        o1_solution_set : list of Solution
            Low-fidelity visited solution set.

        Returns
        -------
        I_h : bool
            Indicator of iteration success:
            - False: successful step (accept candidate and possibly expand trust region),
            - True: unsuccessful step (reject candidate and shrink trust region).
        corr_k : float
            Updated correlation parameter after evaluating the candidate solution.
        candidate_x : ndarray
            Candidate solution proposed by solving the subproblem.
        candidate_solution : Solution
            Candidate solution object with updated evaluations.
        expended_budget : int
            Updated simulation budget after sampling candidate solution.
        fval_tilde : float
            Function estimate at the candidate solution.
        fval_center : float
            Function estimate at the current incumbent (for success ratio calculation).
        o1_solution_set : list of Solution
            Updated low-fidelity solution set after evaluating candidate solution.

        Notes
        -----
        - If `easy_solve=True`, the method uses the Cauchy point for an approximate solution. 
        Otherwise, the subproblem is solved using SciPy's `trust-constr` method with a norm constraint on the step (||s|| ≤ delta_k). 
        - Box constraints are enforced after computing the candidate point.
        - After solving the subproblem:
            - The candidate is evaluated using either MFMC-adjusted sampling or crude Monte Carlo.
            - The success ratio (rho) is computed to determine whether the candidate step is accepted.
        - This function does not update trust-region radii directly; `iterate()` performs that step.

        See Also
        --------
        iterate : Controls the algorithm loop and updates trust-region radii based on `solve_subproblem()` results.
        construct_model : Builds the surrogate model whose coefficients are used in this subproblem.
        """      
        # default values
        eta_1 = self.factors["eta_1"]
        costs = self.factors["costs_mf"]
        easy_solve = self.factors["easy_solve"]

        # solve the local model (subproblem)
        if easy_solve:
            # Cauchy reduction
            if np.dot(np.multiply(grad, Hessian), grad) <= 0:
                tau = 1
            else:
                tau = min(1, norm(grad) ** 3 / (delta_k * np.dot(np.multiply(grad, Hessian), grad)))
            grad = np.reshape(grad, (1, problem.dim))[0]
            candidate_x = new_x - tau * delta_k * grad / norm(grad)
        else:
            # Search engine - solve subproblem
            def subproblem(s):
                return fval[0] + np.dot(s, grad) + np.dot(np.multiply(s, Hessian), s)

            con_f = lambda s: norm(s)
            nlc = NonlinearConstraint(con_f, 0, delta_k)
            solve_subproblem_approximate = minimize(subproblem, np.zeros(problem.dim), method='trust-constr', constraints=nlc)
            candidate_x = new_x + solve_subproblem_approximate.x

        # handle the box constraints
        for i in range(problem.dim):
            if candidate_x[i] <= problem.lower_bounds[i]:
                candidate_x[i] = problem.lower_bounds[i] + 0.01
            elif candidate_x[i] >= problem.upper_bounds[i]:
                candidate_x[i] = problem.upper_bounds[i] - 0.01

        # store the solution (and function estimate at it) to the subproblem as a candidate for the next iterate
        candidate_solution = self.create_new_solution(tuple(candidate_x), problem)
        candidate_o1_solution = self.create_new_solution(tuple(candidate_x), problem)
        visited_pts_list.append(candidate_solution)
        o1_solution_set.append(candidate_o1_solution)
        
        # calculate success ratio
        if new_solution.MFMC_use == True:
            # sampling
            problem.simulate(candidate_solution, new_solution.mfmc_n_rep)
            problem.simulate(candidate_o1_solution, new_solution.mfmc_o1_rep)
            expended_budget += new_solution.mfmc_n_rep*costs[0] + costs[1]*new_solution.mfmc_o1_rep

            c1 = new_solution.mfmc_c1

            candidate_solution.MFMC_use = True
            candidate_solution.mfmc_n_rep = new_solution.mfmc_n_rep
            candidate_solution.mfmc_o1_rep = new_solution.mfmc_o1_rep
            candidate_solution.mfmc_c1 = c1
            candidate_solution.mfmc_mean = candidate_solution.objectives_mean - c1*candidate_solution.lf_f_mean \
                                            + c1*candidate_o1_solution.lf_f_mean 
            
            fval_tilde = -1 * problem.minmax[0] * candidate_solution.mfmc_mean
            fval_center = -1 * problem.minmax[0] * new_solution.mfmc_mean
        else:
            # sampling
            problem.simulate(candidate_solution, new_solution.n_reps)
            expended_budget += new_solution.n_reps * costs[0]
            
            candidate_solution.MFMC_use = False
            candidate_solution.mfmc_n_rep = new_solution.n_reps
            candidate_solution.mfmc_o1_rep = 0
            candidate_solution.mfmc_mean = np.inf

            fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean
            fval_center = -1 * problem.minmax[0] * new_solution.objectives_mean
        
        # compute the success ratio rho
        if (self.evaluate_model(np.zeros(problem.dim), q) - self.evaluate_model(np.array(candidate_x) - np.array(new_x), q)) <= 0:
            rho = 0
        else:
            rho = (fval_center - fval_tilde) / (self.evaluate_model(np.zeros(problem.dim), q) - self.evaluate_model(candidate_x - new_x, q))

        # successful: expand and accept
        if rho >= eta_1:
            I_h = False
        # unsuccessful: shrink and reject
        else:
            I_h = True
        
        return I_h, corr_k, candidate_x, candidate_solution, expended_budget, fval_tilde, fval_center, o1_solution_set

    # run one iteration of trust-region algorithm by bulding and solving a local model and updating the current incumbent and trust-region radius, and saving the data
    def iterate(self, k, corr_k, delta, delta_max, problem, visited_pts_list, new_x, expended_budget, budget_limit, recommended_solns, intermediate_budgets, kappa, new_solution, o1_solution_set):
        """
        Perform one iteration of the ASTRO-BFDF trust-region algorithm.

        This method constructs local surrogate models using high- and/or low-fidelity simulation outputs, solves the corresponding trust-region subproblems, and updates:
        - the trust-region radii (Delta)
        - the correlation parameter (corr_k)
        - the incumbent solution and recommended solutions list (final outputs of ASTRO-BFDF)

        It is called repeatedly by the main optimization loop until the simulation budget is exhausted.

        Parameters
        ----------
        k : int
            Current iteration number.
        corr_k : float
            Current correlation parameter between low- and high-fidelity models; updated after each iteration.
        delta : list of float
            Trust-region radii: [Delta_HF, Delta_LF].
        delta_max : float
            Maximum allowable trust-region radius.
        problem : Problem
            Simulation optimization problem instance (provides simulation oracle, bounds, etc.).
        visited_pts_list : list of Solution
            List of all previously evaluated solutions for reuse in model construction.
        new_x : ndarray or tuple
            Current incumbent solution (center of the trust region).
        expended_budget : float
            Total budget consumed so far (across HF and LF evaluations).
        budget_limit : float
            Maximum allowable simulation budget.
        recommended_solns : list of Solution
            Solutions recommended at various checkpoints (stored during optimization).
        intermediate_budgets : list of int
            Budgets corresponding to recommended solutions for plotting progress curves.
        kappa : float
            Hyperparameter for adaptive sampling rules.
        new_solution : Solution
            Current solution object representing the incumbent.
        o1_solution_set : list of Solution
            Low-fidelity solution set used in constructing bi-fidelity models and applying MFMC.

        Returns
        -------
        corr_k : float
            Updated correlation parameter after the iteration.
        delta : list of float
            Updated trust-region radii: [Delta_HF, Delta_LF].
        recommended_solns : list of Solution
            Updated list of recommended solutions.
        intermediate_budgets : list of int
            Updated list of intermediate budgets.
        expended_budget : int
            Updated budget after this iteration.
        new_x : ndarray
            Updated incumbent solution for the next iteration.
        kappa : float
            Hyperparameter for adaptive sampling.
        new_solution : Solution
            Updated incumbent solution object with simulation results.
        visited_pts_list : list of Solution
            Updated list of all visited points.
        o1_solution_set : list of Solution
            Updated low-fidelity visited solution set.

        Notes
        -----
        - The iteration follows a nested structure:
            1. Construct local models (high- and/or low-fidelity) using interpolation sets.
            2. Solve subproblems approximately (trust-region subproblem).
            3. Evaluate the candidate point using BFAS (def mfmc_sample_size).
            4. Check success ratio and update trust-region radii and correlation parameter.
        - The method internally enforces box constraints and adjusts trust-region steps accordingly.
        - Users typically do NOT call `iterate()` directly; it is invoked by `solve()`.

        See Also
        --------
        solve : Main optimization driver that repeatedly calls iterate() until budget exhaustion.
        """
        
        # default values
        eta_1 = self.factors["eta_1"]
        costs = self.factors["costs_mf"]
        gamma_1 = self.factors["gamma_1"]
        gamma_2 = self.factors["gamma_2"]
        corr_th = self.factors["cor_lf_hf"]
        lambda_min = self.factors["lambda_min"]
        lambda_max = budget_limit - expended_budget
        lambda_k = max(lambda_min, 2 * log(problem.dim + .5, 10)) * max(log(k + 0.1, 10) ** (1.01), 1)

        I_h = True
        Y = []

        # The first iteration need to determine the sample sizes for the initial point by BFAS mannually to determine kappa based on the function estimate at the initial point
        pilot_run = ceil(max(lambda_min, min(.5 * problem.dim, lambda_max)) - 1)

        if k == 1:
            new_solution = self.create_new_solution(tuple(new_x), problem)
            o1_solution = self.create_new_solution(tuple(new_x), problem)

            o1_solution_set.append(o1_solution)

            if len(visited_pts_list) == 0:
                visited_pts_list.append(new_solution)

            # pilot run
            problem.simulate(new_solution, pilot_run)
            expended_budget += pilot_run * costs[0]
            
            # pilot run for low-fidelity
            problem.simulate(o1_solution, pilot_run + 1)
            expended_budget += costs[1]*(pilot_run+1)

            fn = new_solution.objectives_mean
            o_o1_cov = new_solution.hf_lf_cov

            new_solution.MFMC_use = True
            kappa = fn / (delta[0])
            
            while True:
                predicted_N_k = self.get_stopping_time(k, new_solution.objectives_var, delta[0], kappa, problem.dim)

                def subproblem(t):
                    return costs[0]*t[0] + costs[1]*t[1]
                
                var_mf_f = lambda t: self.var_mf([t[0], t[1]],  # sample sizes
                                                    [new_solution.objectives_var, o1_solution.lf_f_var],  # sigma values
                                                    [t[2]],  # coefficient 
                                                    [o_o1_cov]   # covariance values
                                                    ) 

                nlc = NonlinearConstraint(var_mf_f, -np.inf, (kappa**2)*max(delta[0]**2,delta[0]**4)/lambda_k)
                
                A = [[1,-1, 0],[1,0,0],[0,1,0]]
                lc = LinearConstraint(A, \
                                        lb=[-np.inf, new_solution.n_reps, max(new_solution.n_reps, float(o1_solution.n_reps))], \
                                        ub=[0, np.inf, np.inf])
                initial_point = [new_solution.n_reps,max(new_solution.n_reps, float(o1_solution.n_reps)), 1]
                
                solve_subproblem_approximate = minimize(subproblem, initial_point, method='trust-constr', constraints=(lc,nlc))
                n0, n1, c1 = solve_subproblem_approximate.x              
                
                if solve_subproblem_approximate.fun <= costs[0]*predicted_N_k:
                    # use MFMC
                    new_solution.MFMC_use = True
                    
                    if new_solution.n_reps > o1_solution.n_reps:
                        ss_lf = new_solution.n_reps - o1_solution.n_reps + 1
                        problem.simulate(o1_solution, ss_lf)
                        expended_budget += ss_lf*costs[1]

                    # check whetehr we can finish the sampling
                    fn = new_solution.objectives_mean - c1*new_solution.lf_f_mean  + c1*o1_solution.lf_f_mean 
                    if fn < 1 and fn > -1:
                        kappa = 1/(delta[0]**2)
                    else:
                        kappa = fn/(delta[0]**2)
                    
                    if self.var_mf([new_solution.n_reps, o1_solution.n_reps],
                                   [new_solution.objectives_var, o1_solution.lf_f_var],
                                   [c1],[o_o1_cov]) \
                        <= (kappa**2)*max(delta[0]**2,delta[0]**4)/lambda_k or expended_budget > budget_limit:
                        break
                    
                    if new_solution.n_reps >= ceil(n0)-1:
                        problem.simulate(o1_solution, ceil(1/costs[1]))
                        expended_budget += ceil(1/costs[1])*costs[1]
                        
                        fn = new_solution.objectives_mean - c1*new_solution.lf_f_mean  + c1*o1_solution.lf_f_mean

                    else:
                        problem.simulate(new_solution, 1)
                        expended_budget += 1 * costs[0]

                        fn = new_solution.objectives_mean - c1*new_solution.lf_f_mean  + c1*o1_solution.lf_f_mean 
                        
                        # update covariance terms based on increasing sample sizes for the HF function
                        o_o1_cov = new_solution.hf_lf_cov
                        
                else:
                    # use the crude MC - increase n
                    new_solution.MFMC_use = False

                    fn = new_solution.objectives_mean
                    sig2 = new_solution.objectives_var
                    if fn < 1 and fn > -1:
                        kappa = 1/(delta[0]**2)
                    else:
                        kappa = fn/(delta[0]**2)
                    
                    if new_solution.n_reps >= self.get_stopping_time(k, sig2, delta[0], fn/(delta[0]**2), problem.dim) or \
                        new_solution.n_reps >= lambda_max or expended_budget >= budget_limit:
                        break

                    problem.simulate(new_solution, 1)
                    expended_budget += 1 * costs[0]
                
        
            mfmc_var = self.var_mf([new_solution.n_reps, o1_solution.n_reps],
                                [new_solution.objectives_var, o1_solution.lf_f_var],
                                [c1],[o_o1_cov])
            cmc_var = new_solution.objectives_var

            if mfmc_var < cmc_var:
                fn = new_solution.objectives_mean - c1*new_solution.lf_f_mean  + c1*o1_solution.lf_f_mean

            # Save new variables for MFMC
            new_solution.mfmc_mean = fn
            new_solution.mfmc_n_rep = new_solution.n_reps
            new_solution.mfmc_o1_rep = o1_solution.n_reps
            new_solution.mfmc_c1 = c1

            recommended_solns.append(new_solution)
            intermediate_budgets.append(0)
            recommended_solns.append(new_solution)
            intermediate_budgets.append(expended_budget)
        
        # ASTRO-LFDF
        while corr_k > corr_th:
            io = 'inner'
            # Construct LF-based local model
            fval_low, Y, q_low, grad_low, Hessian_low, expended_budget, visited_pts_list, o1_solution_set =\
                self.construct_model(new_x, delta, k, Y, problem, expended_budget, kappa, new_solution, visited_pts_list,
                        o1_solution_set, io, low_or_high=1)
                
            I_h, corr_k, candidate_x, candidate_solution, expended_budget, o1_fval_tilde, o1_fval_center, o1_solution_set =\
                self.solve_subproblem(problem, corr_k, new_x, new_solution, fval_low, q_low, grad_low, Hessian_low, 
                                      delta[1], expended_budget,  visited_pts_list, o1_solution_set)
            
            # Successful iteration with LF-based local model
            if (I_h == False) and (o1_fval_center-o1_fval_tilde >= eta_1*0.0001*delta[0]**2):
                break
            # Unsuccessful iteration with LF-based local model
            else: 
                corr_k = gamma_2 * corr_k
                delta[1] = gamma_2 * delta[1]

        # Success in ASTRO-LFDF / Update the incumbent, the trust region raddi, and the correlation parameter
        if (I_h == False):
            new_x = candidate_x
            new_solution = candidate_solution
            recommended_solns.append(candidate_solution)
            intermediate_budgets.append(expended_budget)
            delta[1] = min(gamma_1 * delta[1], delta_max)
            delta[0] = max(delta[0], delta[1])
            corr_k = min(gamma_1 * corr_k, 1)
        else:
            io = 'outer'
            # Construct both LF-based and HF-based models
            # HF-based local model
            fval_high, Y_high, q_high, grad_high, Hessian_high, expended_budget, visited_pts_list, o1_solution_set =\
                self.construct_model(new_x, delta, k, Y, problem, expended_budget, kappa, new_solution, visited_pts_list,
                                        o1_solution_set, io, low_or_high=0)
            
            # LF-based local model
            fval_1, __, q_1, grad_1, Hessian_1, expended_budget, visited_pts_list, o1_solution_set =\
                self.construct_model(new_x, delta, k, Y_high, problem, expended_budget, kappa, new_solution, visited_pts_list, 
                                        o1_solution_set, io, low_or_high=1)
            
            # Solve the subproblem with the LF-based local model: generate X_k^{s,\ell} in the paper (candidate_x1)
            __, __, candidate_x1, candidate_solution1, expended_budget, fval_tilde1, fval_center1, o1_solution_set  =\
                self.solve_subproblem(problem, corr_k, new_x, new_solution, fval_1, q_1, grad_1, Hessian_1, 
                                        delta[0], expended_budget, visited_pts_list, o1_solution_set)
            
            # Calculate the success ratio (\rhohat^l_k) for the cnadidate point from the LF-based local model
            if (self.evaluate_model(np.zeros(problem.dim), q_high) - self.evaluate_model(np.array(candidate_x1) - np.array(new_x), q_high)) <= 0:
                rho1 = 0
            else:
                rho1 = (fval_center1 - fval_tilde1) / (self.evaluate_model(np.zeros(problem.dim), q_high) - self.evaluate_model(candidate_x1 - new_x, q_high))

            # Update Correlation
            if (rho1 >= eta_1) and (fval_center1-fval_tilde1 >= eta_1*0.0001*delta[0]**2):
                corr_k = min(gamma_1 * corr_k, 1)
            else:
                corr_k = gamma_2 * corr_k
            
            # Solve the subproblem with the HF-based local model: generate X_k^{s,h} in the paper (candidate_x_high)
            I_h_withhigh, __, candidate_x_high, candidate_solution_high, expended_budget, fval_tilde_high, __, o1_solution_set =\
                self.solve_subproblem(problem, corr_k, new_x, new_solution, fval_high, q_high, grad_high, Hessian_high,
                                    delta[0], expended_budget, visited_pts_list, o1_solution_set)

            
            # Check whether which one gives better solution
            # Define the list of function values and corresponding candidates
            fval_list = [fval_tilde_high, fval_tilde1]
            candidates = [(candidate_x_high, candidate_solution_high),
                        (candidate_x1, candidate_solution1)]

            # Find the index of the minimum function value
            min_index = fval_list.index(min(fval_list))
            candidate_x, candidate_solution = candidates[min_index]

            # Update the incumbent, the trust region raddi, and the correlation parameter
            if not I_h_withhigh and 1000 * norm(grad_high) >= delta[0]:
                new_x = candidate_x
                new_solution = candidate_solution
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)
                delta[0] = min(gamma_1 * delta[0], delta_max)
            else:
                delta[0] = min(gamma_2 * delta[0], delta_max)
                delta[1] = min(delta[1], delta[0])

        return corr_k, delta, recommended_solns, intermediate_budgets, expended_budget, new_x, kappa, new_solution, visited_pts_list, o1_solution_set
       
    # start the search and stop when the budget is exhausted
    def solve(self, problem):
        """
        Run a single macroreplication of the ASTRO-BFDF algorithm on the given problem.

        This is the main driver of the ASTRO-BFDF solver. It initializes algorithm parameters, 
        sets up the trust-region framework, and repeatedly calls `iterate()` until the simulation budget is exhausted. 
        Throughout the process, it tracks and stores the recommended solutions and their corresponding intermediate budgets.

        Parameters
        ----------
        problem : Problem
            A simulation-optimization problem instance that provides:
            - `simulate()`: simulation oracle for evaluating solutions,
            - `dim`: problem dimension,
            - `factors`: includes the available simulation budget and initial solution,
            - `lower_bounds` and `upper_bounds`: variable bounds for trust-region updates.

        Returns
        -------
        recommended_solns : list of Solution
            Solutions recommended by the solver at various checkpoints during optimization.
            Includes the final recommended solution at termination.
        intermediate_budgets : list of int
            Budgets corresponding to the checkpoints when solutions were recommended.

        Notes
        -----
        - The method implements the **outer loop** of the ASTRO-BFDF algorithm:
            1. Compute an initial trust-region radius (`delta_initial`) and maximum radius (`delta_max`).
            2. Initialize correlation parameter and incumbent solution.
            3. Repeatedly call `iterate()` until the simulation budget is exhausted.
        - Trust-region radii are initialized based on problem bounds or sampled solutions when explicit bounds are not provided.
        - Common stopping criterion: expended simulation budget exceeds the given budget limit.
        internally within `iterate()` and `construct_model()`.

        See Also
        --------
        iterate : Performs a single trust-region iteration, constructing surrogate models and solving subproblems.
        construct_model : Builds local surrogate models using simulation evaluations.
        """

        budget_limit = problem.factors["budget"]        
        delta_coefficient = self.factors["coef_delta_initial"]
        # Designate random number generator for random sampling 
        find_next_soln_rng = self.rng_list[1]
        
        # Generate many dummy solutions without replication only to find a reasonable maximum radius
        dummy_solns = []
        for i in range(1000 * problem.dim):    
            dummy_solns += [problem.get_random_solution(find_next_soln_rng)]

        # Range for each dimension is calculated and compared with box constraints range if given 
        # TODO: just use box constraints range if given

        delta_max_arr = []
        for i in range(problem.dim):
            delta_max_arr += [min(max([sol[i] for sol in dummy_solns])-min([sol[i] for sol in dummy_solns]), 
                                  problem.upper_bounds[0] - problem.lower_bounds[0])]          
        # TODO: update this so that it could be used for problems with decision variables at varying scales!
        delta_max = max(delta_max_arr)
        
        # Reset iteration and data storage arrays
        visited_pts_list = []
        k = 0        
        corr_k = 1
        
        if not (np.isinf(problem.upper_bounds).any() or np.isinf(problem.lower_bounds).any()):
            delta_max = problem.upper_bounds[0] - problem.lower_bounds[0]
            delta_initial = delta_coefficient * 0.1 * min(np.array(problem.upper_bounds) - np.array(problem.lower_bounds))
        else:
            delta_initial = delta_coefficient * 10 ** (ceil(log(delta_max * 2, 10) - 1) / problem.dim)

        delta = [delta_initial, delta_initial]
        new_x = problem.factors["initial_solution"]
        expended_budget, kappa = 0, 0
        new_solution, recommended_solns, intermediate_budgets = [], [], [] 
        o1_solution_set = []

        while expended_budget < budget_limit:
            k += 1
            corr_k, delta, recommended_solns, intermediate_budgets, expended_budget, new_x, kappa, new_solution, visited_pts_list,\
                 o1_solution_set = \
                self.iterate(k, corr_k, delta, delta_max, problem, visited_pts_list, new_x,\
                              expended_budget, budget_limit, recommended_solns, intermediate_budgets, kappa, new_solution,\
                              o1_solution_set)
            
        return recommended_solns, intermediate_budgets
        