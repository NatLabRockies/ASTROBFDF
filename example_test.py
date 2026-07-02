"""
Example script: Apply ASTRO-BFDF to a custom bi-fidelity problem.

This script demonstrates:
1. How to define custom high- and low-fidelity functions.
2. How to specify noise rules for HF and LF oracles.
3. How to set experiment parameters: initial solution, budget, costs
4. How to configure solvers and run them.
5. How to normalize and plot results for comparison.

Output:
- Stores .pickle files in experiments/outputs/
- Generates plots (progress curves) for solver comparison.

Modify sections marked with "### USER CAN MODIFY ###" for your own problem.
"""

import sys
import os.path as o
import os
import numpy as np
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))) # type:ignore

from simopt.experiment_base import ProblemSolver, plot_area_scatterplots, post_normalize, plot_progress_curves, plot_solvability_cdfs, read_experiment_results, plot_solvability_profiles, plot_terminal_scatterplots, plot_terminal_progress

# ==============================
# Function Settings
# ==============================
### USER CAN MODIFY ###
def det_hf_func(x):
    """
    High-fidelity deterministic function.
    
    Parameters
    ----------
    x : tuple or list
        Input decision vector (e.g., (x1, x2, ...)).
    
    Returns
    -------
    float
        High-fidelity function value at x.
    """
    return x[0]**2

### USER CAN MODIFY ###
def det_lf_func(x):
    
    """
    Low-fidelity deterministic function.
    
    Parameters
    ----------
    x : tuple or list
        Input decision vector.
    
    Returns
    -------
    float
        Low-fidelity approximation of the high-fidelity function.
    """
    return x

### USER CAN MODIFY ###
def noise_rule_hf(det_hf, hf_noise, lf_noise):
    """
    Noise combination rule for high-fidelity oracle.
    
    Parameters
    ----------
    det_hf : float
        Deterministic high-fidelity function value.
    hf_noise : float
        Random noise sampled for the HF oracle.
    lf_noise : float
        Random noise sampled for the LF oracle (not used in default HF rule).
    
    Returns
    -------
    float
        High-fidelity function value with noise applied.
    
    Notes
    -----
    Default rule: f_hf(x) = deterministic value + HF noise.
    """
    return det_hf + hf_noise

### USER CAN MODIFY ###
def noise_rule_lf(det_lf, hf_noise, lf_noise):
    """
    Noise combination rule for low-fidelity oracle.
    
    Parameters
    ----------
    det_lf : float
        Deterministic low-fidelity function value (or HF if same formula).
    hf_noise : float
        Random noise sampled for HF oracle.
    lf_noise : float
        Random noise sampled for LF oracle.
    
    Returns
    -------
    float
        Low-fidelity function value with noise applied.
    
    Notes
    -----
    Default rule: f_lf(x) = deterministic value + average(HF noise, LF noise).
    """
    return det_lf + (hf_noise + lf_noise)/2

def main():
    ### USER CAN MODIFY ###
    # ==============================
    # Experiment Settings
    # ==============================
    '''
    Note: Other single-fidelity solvers, e.g., ASTRO-DF and ADAM, can be used without modification. 
          For example, solvers = ["EXASTROBFDF", "ASTRODF", "ADAM"]
    '''
    M = 1                                                           # Number of macro-replications (solution sequences)
    solvers = ["EXASTROBFDF"]                                       # List of solvers to run
    problem_dimension = 1
    initial_solution = (10,)*problem_dimension                      # Initial point for the problem (tuple)

    ### USER CAN MODIFY ###
    # ==============================
    # Model Settings
    # ==============================
    hf_variance = 0.1    # High-fidelity oracle noise variance
    lf_variance = 0.1    # Low-fidelity oracle noise variance
    hf_mean = 0.0        # High-fidelity oracle noise mean
    lf_mean = 0.0        # Low-fidelity oracle noise mean

    ### USER CAN MODIFY ###
    # ==============================
    # Solver Settings (ASTRO-BFDF)
    # ==============================
    initial_delta = [1,1]     # [delta_0^h, delta_0^l]: delta_0^h = TR size for HF, delta_0^l = TR size for LF
    delta_max = 20            # Low-fidelity oracle noise variance

    ### USER CAN MODIFY ###
    # ==============================
    # Budget and Cost Settings
    # ==============================
    budget = 200            # Total budget for simulations
    cost = [0.1, 0.01]      # [c1, c2]:
                            #   c1 = cost per one HF replication (per HF call / per replication)
                            #   c2 = cost per one LF replication (per LF call / per replication)
                            #   Require: c1 >= c2

    batch_sizes = [10, 20]  # [b1, b2]: default batch sizes = [1, 10]
                            #   b1 = HF batch size (number of HF replications added per adaptive-sampling update)
                            #   b2 = LF batch size (number of LF replications added per adaptive-sampling update)
                            #   Larger batch -> fewer stopping-rule checks (lower overhead) but more overshoot risk.
                            #   Smaller batch -> finer control but higher overhead.

    # Example Run #####################################################################################
    N = 200
    L = 200

    model_fixed_factors = {"x": (0,)*problem_dimension,
                           "sigma_h": hf_variance,
                           "sigma_l": lf_variance,
                           "mean_h": hf_mean,
                           "mean_l": lf_mean,
                           "deterministic_hfn": det_hf_func,
                           "deterministic_lfn": det_lf_func,
                           "noise_rule_hf": noise_rule_hf,
                           "noise_rule_lf": noise_rule_lf}
    problem_fixed_factors = {"initial_solution": initial_solution,
                             "budget": budget}

    # Temporarily store experiments on the same problem for post-normalization.
    experiments_same_problem = []
    solver_fixed_factors = {}
    for solver in solvers:
        solver_name = solver
        if solver == "EXASTROBFDF":
            solver_fixed_factors = {"costs_mf": cost, "delta": initial_delta, "delta_max": delta_max, "batch_size": batch_sizes}

        # Loop over solvers:
        new_experiment = ProblemSolver(solver_name=solver_name,
                                        solver_rename=solver,
                                        solver_fixed_factors=solver_fixed_factors,
                                        problem_name="EXAMPLE-1",
                                        problem_fixed_factors=problem_fixed_factors,
                                        model_fixed_factors=model_fixed_factors
                                        )
        
        # Run experiment with M.
        new_experiment.run(n_macroreps=M)

        # Post replicate experiment with N.
        new_experiment.post_replicate(n_postreps=N)
        experiments_same_problem.append(new_experiment)

    # Post-normalize experiments with L.
    # Provide NO proxies for f(x0), f(x*), or f(x).
    post_normalize(experiments=experiments_same_problem, n_postreps_init_opt=L)

    # Example Plotting #####################################################################################
    experiments = []

    for solver in solvers:
        experiments_same_solver = []

        solver_display = solver
        if solver == "EXASTROBFDF":
            solver_display = "ASTRO-BFDF"
        elif solver == "ASTRODF":
            solver_display = "ASTRO-DF"

        file_name = f"{solver}_on_EXAMPLE-1"
        
        # Load experiment.
        new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")

        # Rename problem to produce nicer plot labels.
        new_experiment.problem.name = fr"EXAMPLE"
        new_experiment.solver.name = solver_display
        experiments_same_solver.append(new_experiment)

        experiments.append(experiments_same_solver)

    # PLOTTING
    n_solvers = len(experiments)
    n_problems = len(experiments[0])

    CI_param = True

    for i in range(n_problems):
        plot_progress_curves([experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type="mean", all_in_one=True, plot_CIs=CI_param, print_max_hw=False, normalize=False)

if (__name__ == "__main__"):
    main()