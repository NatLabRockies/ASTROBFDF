'''
Generate Figure 4b
'''
import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

from simopt.experiment_base import post_normalize, read_experiment_results, plot_solvability_profiles, plot_progress_curves

# Problems factors used in experiments
hf_sigma2 = [5,10,15]
lf_sigma2 = [5,10,15]
correlation = [0.1, 0.5, 0.9]

# RUNNING AND POST-PROCESSING EXPERIMENTS
L = 200
solvers = ["ASTROBFDF_c=1_coef=2.0",  "ASTRODF_coef=2.0", "ADAM_lr=1_n=10", "BFSAG_c=1_lr=0.01_n=10", "BFBO_c=1_n=10"]


problems = ["FORRETAL", "COLVILLE", "ROSENBROCK", "BRANIN"]

for problem in problems:
    for hf_sig2 in hf_sigma2:
        for sig2 in lf_sigma2:
            for cor in correlation:
                model_fixed_factors = {"sigma2": sig2,
                                       "sigma1": hf_sig2,
                                        "cor": cor}
                #problem_fixed_factors = {"budget": 200}
                problem_rename = f"{problem}-1_hf_sig2={hf_sig2}_lf_sig2={sig2}_cor={cor}"

                # Temporarily store experiments on the same problem for post-normalization.
                experiments_same_problem = []
                solver_fixed_factors = {}
                for solver in solvers:
                    solver_name = solver                    
                    problem_rename = f"{problem}-1_hf_sig2={hf_sig2}_lf_sig2={sig2}_cor={cor}"
                    file_name = f"{solver}_on_{problem_rename}"
                    # Load experiment.
                    new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
                    experiments_same_problem.append(new_experiment)

                # Post-normalize experiments with L.
                # Provide NO proxies for f(x0), f(x*), or f(x).
                post_normalize(experiments=experiments_same_problem, n_postreps_init_opt=L)

        
# LOAD DATA FROM .PICKLE FILES TO PREPARE FOR PLOTTING.
experiments = []

# Load .pickle files of past results.
# Load all experiments for a given solver, for all solvers.
for solver in solvers:
    experiments_same_solver = []

    solver_display = solver
    if solver == "ASTROBFDF_c=1_coef=2.0":
        solver_display = "ASTRO-BFDF"
    elif solver == "ASTRODF_coef=2.0":
        solver_display = "ASTRO-DF"
    elif solver == "ADAM_lr=1_n=10":
        solver_display = "ADAM"
    elif solver == "BFSAG_c=1_lr=0.01_n=10":
        solver_display = "BFSAG"
    elif solver == "BFBO_c=1_n=10":
        solver_display = "BFBO"


    for problem in problems:
        # Load IRONORECONT .pickle files
        for hf_sig2 in hf_sigma2:
            for sig2 in lf_sigma2:
                for cor in correlation:
                    problem_rename = f"{problem}-1_hf_sig2={hf_sig2}_lf_sig2={sig2}_cor={cor}"
                    file_name = f"{solver}_on_{problem_rename}"
                    # Load experiment.
                    new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
                    # Rename problem to produce nicer plot labels.
                    new_experiment.problem.name = fr"{problem}-1 with hf_sig2={hf_sig2} and lf_sig2={sig2} and cor={cor}"
                    new_experiment.solver.name = solver_display
                    experiments_same_solver.append(new_experiment)

    experiments.append(experiments_same_solver)

# PLOTTING
n_solvers = len(experiments)
n_problems = len(experiments[0])

CI_param = True
alpha = 0.01

plot_solvability_profiles(experiments, plot_type="cdf_solvability", solve_tol=alpha, all_in_one=True, plot_CIs=CI_param, print_max_hw=False)
