"""
Generate Figure 12a
"""
import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), ".."))) # type:ignore

from simopt.experiment_base import ProblemSolver, post_normalize 

def main():
    # Problems factors used in experiments
    # SSCONT
    demand_means = [400.0]
    lead_means = [3.0]

    # RUNNING AND POST-PROCESSING EXPERIMENTS
    M = 1
    N = 100
    L = 200

    solvers = ["ASTROBFDF", "ASTRODF"]

    # SSCONT
    for dm in demand_means:
        for lm in lead_means:
            model_fixed_factors = {"demand_mean": dm,
                                "lead_mean": lm}
            # Default budget for (s,S) inventory problem = 1000 replications.
            # RS with sample size of 100 will get through only 10 iterations.
            problem_fixed_factors = {"budget": 1000}
            problem_rename = f"SSCONTMF-1_dm={dm}_lm={lm}"

            # Temporarily store experiments on the same problem for post-normalization.
            experiments_same_problem = []
            solver_fixed_factors = {}
            for solver in solvers:
                solver_name = solver
                    
                if solver == "ASTROBFDF":
                    solver_name = "ASTROBFDF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 1}
                elif solver == "ASTRODF":
                    solver_name = "ASTRODF"
                    solver_fixed_factors = {"costs_mf": [1, 0.3], "coef_delta_initial": 1}
            
                # Loop over solvers:
                new_experiment = ProblemSolver(solver_name=solver_name,
                                            solver_rename=solver,
                                            solver_fixed_factors=solver_fixed_factors,
                                            problem_name="SSCONTMF-1",
                                            problem_rename=problem_rename,
                                            problem_fixed_factors=problem_fixed_factors,
                                            model_fixed_factors=model_fixed_factors
                                            )
                # Run experiment with M.
                new_experiment.run(n_macroreps=M)
                # Post replicate experiment with N.
                new_experiment.post_replicate(n_postreps=N)
                experiments_same_problem.append(new_experiment)
                
                if solver == "ASTROBFDF":
                    trajectory_astrobfdf = new_experiment.all_recommended_xs
                else:
                    trajectory_astrodf = new_experiment.all_recommended_xs

            # Post-normalize experiments with L.
            # Provide NO proxies for f(x0), f(x*), or f(x).
            post_normalize(experiments=experiments_same_problem, n_postreps_init_opt=L)

    """
    Generate Figure 9 (High and Low Fidelity Only)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    factors = {
        "demand_mean": 400.0,
        "lead_mean": 3.0,
        "backorder_cost": 4.0,
        "holding_cost": 1.0,
        "fixed_cost": 36.0,
        "variable_cost": 2.0,
        "s": 550.0,
        "S": 600.0,
        "n_days": 120,
        "warmup": 20
    }

    s_values = np.linspace(100, 2000, 100)
    S_values = np.linspace(100, 2000, 100)
    sample_size = 50
    object = []

    for i in range(len(s_values)):
        for j in range(len(S_values)):
            objective_o = 0
            for z in range(sample_size):
                np.random.seed(z)

                demand_mean = factors["demand_mean"]
                lead_mean = factors["lead_mean"]
                backorder_cost = factors["backorder_cost"]
                holding_cost = factors["holding_cost"]
                fixed_cost = factors["fixed_cost"]
                variable_cost = factors["variable_cost"]
                s = s_values[i]
                S = S_values[j]
                n_days = factors["n_days"]
                warmup = factors["warmup"]

                demands = np.random.exponential(scale=demand_mean, size=n_days + warmup)
                start_inv = np.zeros(n_days + warmup)
                start_inv[0] = s
                end_inv = np.zeros(n_days + warmup)
                orders_received = np.zeros(n_days + warmup)
                inv_pos = np.zeros(n_days + warmup)
                orders_placed = np.zeros(n_days + warmup)
                orders_outstanding = np.zeros(n_days + warmup)

                for day in range(n_days + warmup):
                    end_inv[day] = start_inv[day] - demands[day]
                    inv_pos[day] = end_inv[day] + orders_outstanding[day]

                    orders_placed[day] = np.max(((inv_pos[day] < s) * (S - inv_pos[day])), 0)

                    if orders_placed[day] > 0:
                        lead = np.random.poisson(lead_mean)
                        for future_day in range(day + 1, day + lead + 1):
                            if future_day < n_days + warmup:
                                orders_outstanding[future_day] += orders_placed[day]
                        if day + lead + 1 < n_days + warmup:
                            orders_received[day + lead + 1] += orders_placed[day]

                    if day < n_days + warmup - 1:
                        start_inv[day + 1] = end_inv[day] + orders_received[day + 1]

                # High-fidelity
                avg_order_costs = np.mean(fixed_cost * (orders_placed[warmup:] > 0) +
                                        variable_cost * orders_placed[warmup:])
                avg_holding_costs = np.mean(holding_cost * end_inv[warmup:] * [end_inv[warmup:] > 0])
                on_time_rate = 1 - np.sum(np.min(np.vstack((demands[warmup:], demands[warmup:] - start_inv[warmup:])), axis=0)
                                        * ((demands[warmup:] - start_inv[warmup:]) > 0)) / np.sum(demands[warmup:])
                avg_backorder_costs = backorder_cost * (1 - on_time_rate) * np.sum(demands[warmup:]) / float(n_days)
                objective_o += avg_backorder_costs + avg_order_costs + avg_holding_costs

                if s > S:
                    penalty = 500
                    objective_o += penalty

            object.append(objective_o / sample_size)

    # Convert to arrays and reshape
    object_array = np.array(object).reshape(100, 100)

    # Extract points
    trajectory_astrobfdf = [[(x0, x0 + x1) for (x0, x1) in inner_list] for inner_list in trajectory_astrobfdf]
    traj_bfdf = np.array(trajectory_astrobfdf[0])
    x_bfdf, y_bfdf = traj_bfdf[:, 0], traj_bfdf[:, 1]

    trajectory_astrodf = [[(x0, x0 + x1) for (x0, x1) in inner_list] for inner_list in trajectory_astrodf]
    traj_df = np.array(trajectory_astrodf[0])
    x_df, y_df = traj_df[:, 0], traj_df[:, 1]

    # Plot
    plt.figure(figsize=(8, 7))
    contour = plt.contourf(s_values, S_values, object_array, levels=50, cmap='binary')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)

    # Overlay trajectories with larger markers
    plt.plot(y_bfdf, x_bfdf, color='red', marker='o', markersize=12, linewidth=2, label='ASTRO-BFDF')
    plt.plot(y_df, x_df, color='blue', marker='^', markersize=12, linewidth=2, label='ASTRO-DF')

    # Final point as star
    plt.plot(y_bfdf[-1], x_bfdf[-1], color='yellow', marker='*', markersize=15, linestyle='None', label='ASTRO-BFDF Last Point')
    plt.plot(y_df[-1], x_df[-1], color='orange', marker='s', markersize=15, linestyle='None', label='ASTRO-DF Last Point')

    # Labels and title
    plt.xlabel('S', fontsize=16)
    plt.ylabel('s', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig("experiments/plots/figure12a.pdf", format='pdf')

if (__name__ == "__main__"):
    main()