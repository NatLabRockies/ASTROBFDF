"""
Generate Figure 11b
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
    "s": 500.0, 
    "n_days": 120,
    "warmup": 20
}

S_values = np.linspace(600, 2000, 100)
sample_size = 1
objective_high = []
objective_low = []

for j, S in enumerate(S_values):
    objective_o = 0
    objective_o2 = 0
    for z in range(sample_size):
        np.random.seed(z)
        #np.random.seed(j)

        demand_mean = factors["demand_mean"]
        lead_mean = factors["lead_mean"]
        backorder_cost = factors["backorder_cost"]
        holding_cost = factors["holding_cost"]
        fixed_cost = factors["fixed_cost"]
        variable_cost = factors["variable_cost"]
        s = factors["s"]  # Fixed
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

        # Low-fidelity
        o2 = warmup + 30
        avg_order_costs_o2 = np.mean(fixed_cost * (orders_placed[warmup:o2] > 0) +
                                     variable_cost * orders_placed[warmup:o2])
        avg_holding_costs_o2 = np.mean(holding_cost * end_inv[warmup:o2] * [end_inv[warmup:o2] > 0])
        on_time_rate_o2 = 1 - np.sum(np.min(np.vstack((demands[warmup:o2], demands[warmup:o2] - start_inv[warmup:o2])), axis=0)
                                     * ((demands[warmup:o2] - start_inv[warmup:o2]) > 0)) / np.sum(demands[warmup:o2])
        avg_backorder_costs_o2 = backorder_cost * (1 - on_time_rate_o2) * np.sum(demands[warmup:o2]) / float(o2 - warmup)
        objective_o2 += avg_backorder_costs_o2 + avg_order_costs_o2 + avg_holding_costs_o2

        if s > S:
            penalty = 500
            objective_o += penalty
            objective_o2 += penalty

    objective_high.append(objective_o / sample_size)
    objective_low.append(objective_o2 / sample_size)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(S_values, objective_high, label="HF function", linewidth=3)
plt.plot(S_values, objective_low, label="LF function", linewidth=3, linestyle='--')
plt.xlabel("$S$", fontsize=20)
plt.ylabel("Objective Value", fontsize=20)
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig("experiments/plots/figure11b.pdf", format='pdf', dpi=300)
plt.show()
