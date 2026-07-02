"""
Generate Figure 7b
"""
import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))) # type:ignore

from simopt.experiment_base import ProblemSolver, plot_area_scatterplots, post_normalize, plot_progress_curves, plot_solvability_cdfs, read_experiment_results, plot_solvability_profiles, plot_terminal_scatterplots, plot_terminal_progress

def main():
    hf_sigma2 = [1]
    lf_sigma2 = [1]

    correlation = [0]
    
    # RUNNING AND POST-PROCESSING EXPERIMENTS
    M = 1
    N = 100
    L = 200

    solvers = ["ASTROBFDF", "ASTRODF"] # adam sample size
    
    problems = ["BRANIN"]

    # SYN
    for problem in problems:
        for hf_sig2 in hf_sigma2:
            for sig2 in lf_sigma2:
                for cor in correlation:
                    model_fixed_factors = {"sigma2": sig2,
                                        "sigma1": hf_sig2,
                                            "cor": cor}
                    if problem == "BRANIN":
                        problem_fixed_factors = {"initial_solution": (6, 6)}
                    elif problem == "COLVILLE":
                        problem_fixed_factors = {"budget": 1000}
                    else:
                        problem_fixed_factors = {}
                    problem_rename = f"{problem}-1_hf_sig2={hf_sig2}_lf_sig2={sig2}_cor={cor}"

                    # Temporarily store experiments on the same problem for post-normalization.
                    experiments_same_problem = []
                    solver_fixed_factors = {}
                    for solver in solvers:
                        solver_name = solver
                        if solver == "ASTROBFDF":
                            solver_fixed_factors = {"costs_mf": [1, 0.1]}
                        

                        # Loop over solvers:
                        new_experiment = ProblemSolver(solver_name=solver_name,
                                                        solver_rename=solver,
                                                        solver_fixed_factors=solver_fixed_factors,
                                                        problem_name=problem+"-1",
                                                        problem_rename=problem_rename,
                                                        problem_fixed_factors=problem_fixed_factors,
                                                        model_fixed_factors=model_fixed_factors
                                                        )
                        # Run experiment with M.
                        new_experiment.run(n_macroreps=M)
                        # Post replicate experiment with N.
                        new_experiment.post_replicate(n_postreps=N)
                        experiments_same_problem.append(new_experiment)
                        #print(new_experiment.all_recommended_xs)
                    
                        if solver == "ASTROBFDF":
                            trajectory_astrobfdf = new_experiment.all_recommended_xs
                        else:
                            trajectory_astrodf = new_experiment.all_recommended_xs
                    # Post-normalize experiments with L.
                    # Provide NO proxies for f(x0), f(x*), or f(x).
                    post_normalize(experiments=experiments_same_problem, n_postreps_init_opt=L)


    """
    Generate Figure 5
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Define the range of x_i values
    x0 = np.linspace(-6, 11, 50)
    x1 = np.linspace(-6, 11, 50)
    X0, X1 = np.meshgrid(x0, x1)

    # Define the correlation coefficient
    cor = 0  # You can change this value as needed

    # Calculate deterministic_hfn for each combination of x0 and x1
    deterministic_hfn = (
        X1 - (5.1 * X0**2 / (4 * np.pi**2)) + (5 * X0 / np.pi) - 6 +
        10 * (1 - 1 / (8 * np.pi)) * np.cos(X0) + 10
    )


    # Extract points
    traj_bfdf = np.array(trajectory_astrobfdf[0])
    x_bfdf, y_bfdf = traj_bfdf[:, 0], traj_bfdf[:, 1]

    traj_df = np.array(trajectory_astrodf[0])
    x_df, y_df = traj_df[:, 0], traj_df[:, 1]

    # Plot
    plt.figure(figsize=(8, 7))
    contour = plt.contourf(X0, X1, deterministic_hfn, levels=50, cmap='viridis')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)

    # Overlay trajectories with larger markers
    plt.plot(x_bfdf, y_bfdf, color='red', marker='o', markersize=12, linewidth=2, label='ASTRO-BFDF')
    plt.plot(x_df, y_df, color='blue', marker='^', markersize=12, linewidth=2, label='ASTRO-DF')

    # Final point as star
    plt.plot(x_bfdf[-1], y_bfdf[-1], color='yellow', marker='*', markersize=15, linestyle='None', label='ASTRO-BFDF Last Point')
    plt.plot(x_df[-1], y_df[-1], color='orange', marker='*', markersize=15, linestyle='None', label='ASTRO-DF Last Point')

    # Labels and title
    plt.xlabel('x[1]', fontsize=16)
    plt.ylabel('x[2]', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig("experiments/plots/figure7b.pdf", format='pdf')



if (__name__ == "__main__"):
    main()

