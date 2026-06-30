"""
Generate Figure 10
"""
import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

from simopt.experiment_base import ProblemSolver, plot_area_scatterplots, post_normalize, plot_progress_curves, plot_solvability_cdfs, read_experiment_results, plot_solvability_profiles, plot_terminal_scatterplots, plot_terminal_progress


# Problems factors used in experiments
# MM1
mu_set = [1.0]
lambda_setup = [1.0, 5.0]

# RUNNING AND POST-PROCESSING EXPERIMENTS
L = 200
solvers = ["ASTROBFDF_coef=1", "ASTRODF_coef=1"] 
problems = ["MM1MF", "SSCONTMF"]

experiments = []


# MM1
for mu in mu_set:
    for ld in lambda_setup:
        model_fixed_factors = {"lambda": ld,
                                "mu": mu}

        # Temporarily store experiments on the same problem for post-normalization.
        experiments_same_problem = []
        solver_fixed_factors = {}
        for solver in solvers:
            solver_name = solver

            problem_rename = f"MM1MF-1_mu={mu}_ld={ld}"
            file_name = f"{solver}_on_{problem_rename}"
            # Load experiment.
            new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
            experiments_same_problem.append(new_experiment)

        # Post-normalize experiments with L.
        # Provide NO proxies for f(x0), f(x*), or f(x).
        post_normalize(experiments=experiments_same_problem, n_postreps_init_opt=L)

for solver in solvers:
    experiments_same_solver = []

    solver_display = solver
    if solver == "ASTROBFDF_coef=1":
        solver_display = "ASTRO-BFDF"
    elif solver == "ASTRODF_coef=1":
        solver_display = "ASTRO-DF"

    for problem in problems:
        if problem == "MM1MF":
            # Load IRONORECONT .pickle files
            for mu in mu_set:
                for ld in lambda_setup:
                    problem_rename = f"{problem}-1_mu={mu}_ld={ld}"
                    file_name = f"{solver}_on_{problem_rename}"
                    # Load experiment.
                    new_experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
                    # Rename problem to produce nicer plot labels.
                    new_experiment.problem.name = fr"{problem}-1 with $mu={mu}$ and lambda={ld}"
                    new_experiment.solver.name = solver_display
                    experiments_same_solver.append(new_experiment)

    experiments.append(experiments_same_solver)

# PLOTTING
n_solvers = len(experiments)
n_problems = len(experiments[0])

CI_param = True
alpha = 0.01

for i in range(n_problems):
    plot_progress_curves([experiments[solver_idx][i] for solver_idx in range(n_solvers)], plot_type="mean", all_in_one=True, plot_CIs=CI_param, print_max_hw=False)
