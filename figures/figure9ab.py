"""
Generate Figure 9a and 9b
"""
import numpy as np
import matplotlib.pyplot as plt

factors = {
    "lambda": 1.0,
    "mu": 1.0,
    "people": 200,
    "warmup": 20
}

# Set the random seed
seed_value = 1
np.random.seed(seed_value)

# Calculate total number of arrivals to simulate.
total = factors["warmup"] + factors["people"]
lf_t = factors["warmup"]+15

mu_values = np.linspace(1, 6, 50)
result = []
result_lf = []

for i in range(len(mu_values)):
    
    # Generate all interarrival and service times up front.
    arrival_times = np.random.exponential(scale = 1/factors["lambda"], size=total)
    service_times = np.random.exponential(scale = 1/mu_values[i], size=total)
    costs = 0.2*mu_values[i]**2
    
    cust_mat = np.zeros((total, 10))
    cust_mat[:, 0] = np.cumsum(arrival_times)
    cust_mat[:, 1] = service_times
    # Input entries for first customer's queueing experience.
    cust_mat[0, 2] = cust_mat[0, 0] + cust_mat[0, 1]
    cust_mat[0, 3] = cust_mat[0, 1]
    cust_mat[0, 4] = 0
    cust_mat[0, 5] = 0
    cust_mat[0, 6] = -cust_mat[0, 1] / factors["mu"]
    cust_mat[0, 7] = 0
    cust_mat[0, 8] = 0
    cust_mat[0, 9] = 0
    # Fill in entries for remaining customers' experiences.
    for i in range(1, total):
        cust_mat[i, 2] = (max(cust_mat[i, 0], cust_mat[i - 1, 2])
                          + cust_mat[i, 1])
        cust_mat[i, 3] = cust_mat[i, 2] - cust_mat[i, 0]
        cust_mat[i, 4] = cust_mat[i, 3] - cust_mat[i, 1]
        cust_mat[i, 5] = (sum(cust_mat[i - int(cust_mat[i - 1, 5]) - 1:i, 2]
                              > cust_mat[i, 0]))
        cust_mat[i, 6] = (-sum(cust_mat[i - int(cust_mat[i, 5]):i + 1, 1])
                          / factors["mu"])
        cust_mat[i, 7] = (-sum(cust_mat[i - int(cust_mat[i, 5]):i, 1])
                          / factors["mu"])
        cust_mat[i, 8] = np.nan  # ... to be derived
        cust_mat[i, 9] = np.nan  # ... to be derived
    # Compute average sojourn time and its gradient.
    mean_sojourn_time = np.mean(cust_mat[factors["warmup"]:, 3])
    mean_sojourn_time_lf = np.mean(cust_mat[factors["warmup"]:lf_t, 3])
    
    result.append(mean_sojourn_time+costs)
    result_lf.append(mean_sojourn_time_lf+costs)

plt.figure(figsize=(7, 4.8))

plt.rc('font', size=14)  # Change the font size for labels and legends
plt.rc('xtick', labelsize=12)  # Change the font size for x-axis tick labels
plt.rc('ytick', labelsize=12)

plt.plot(mu_values, result, color='C0', linewidth=2, label='HF function')
plt.plot(mu_values, result_lf, color='C1', linewidth=2, linestyle='--', label='LF function')

plt.legend(loc="upper right")
plt.xlabel('mu', fontsize=18)
plt.ylabel('objective function value', fontsize=18)
plt.grid(True)

plt.savefig("experiments/plots/figure9a.pdf", format='pdf')


################ WITH CRN ###################################################
# Calculate total number of arrivals to simulate.
total = factors["warmup"] + factors["people"]
lf_t = factors["warmup"]+20

# Set the random seed
#seed_value = 1
#np.random.seed(seed_value)

mu_values = np.linspace(1, 6, 50)
result = []
result_lf = []

for z in range(len(mu_values)):
    mean_tempt = []
    mean_tempt_lf = []
    costs = 0.2*mu_values[z]**2
    for j in range(10):
        # Set the random seed
        seed_value = j
        np.random.seed(seed_value)
        
        # Generate all interarrival and service times up front.
        arrival_times = np.random.exponential(scale = 1/factors["lambda"], size=total)
        service_times = np.random.exponential(scale = 1/mu_values[z], size=total)
        
        cust_mat = np.zeros((total, 10))
        cust_mat[:, 0] = np.cumsum(arrival_times)
        cust_mat[:, 1] = service_times
        # Input entries for first customer's queueing experience.
        cust_mat[0, 2] = cust_mat[0, 0] + cust_mat[0, 1]
        cust_mat[0, 3] = cust_mat[0, 1]
        cust_mat[0, 4] = 0
        cust_mat[0, 5] = 0
        cust_mat[0, 6] = -cust_mat[0, 1] / mu_values[z]
        cust_mat[0, 7] = 0
        cust_mat[0, 8] = 0
        cust_mat[0, 9] = 0
        # Fill in entries for remaining customers' experiences.
        for i in range(1, total):
            cust_mat[i, 2] = (max(cust_mat[i, 0], cust_mat[i - 1, 2])
                              + cust_mat[i, 1])
            cust_mat[i, 3] = cust_mat[i, 2] - cust_mat[i, 0]
            cust_mat[i, 4] = cust_mat[i, 3] - cust_mat[i, 1]
            cust_mat[i, 5] = (sum(cust_mat[i - int(cust_mat[i - 1, 5]) - 1:i, 2]
                                  > cust_mat[i, 0]))
            cust_mat[i, 6] = (-sum(cust_mat[i - int(cust_mat[i, 5]):i + 1, 1])
                              / factors["mu"])
            cust_mat[i, 7] = (-sum(cust_mat[i - int(cust_mat[i, 5]):i, 1])
                              / factors["mu"])
            cust_mat[i, 8] = np.nan  # ... to be derived
            cust_mat[i, 9] = np.nan  # ... to be derived
            
        # Compute average sojourn time and its gradient.
        mean_sojourn_time = np.mean(cust_mat[factors["warmup"]:, 3])
        mean_sojourn_time_lf = np.mean(cust_mat[factors["warmup"]:lf_t, 3])
        mean_tempt.append(mean_sojourn_time)
        mean_tempt_lf.append(mean_sojourn_time_lf)
    result.append(np.mean(mean_tempt)+costs)
    result_lf.append(np.mean(mean_tempt_lf)+costs)


plt.figure(figsize=(7, 4.8))

plt.rc('font', size=14)  # Change the font size for labels and legends
plt.rc('xtick', labelsize=12)  # Change the font size for x-axis tick labels
plt.rc('ytick', labelsize=12)

plt.plot(mu_values, result, color='C0', linewidth=2, label='HF function')
plt.plot(mu_values, result_lf, color='C1', linewidth=2, linestyle='--', label='LF function')

plt.legend(loc="upper right")
plt.xlabel('mu', fontsize=18)
plt.ylabel('objective function value', fontsize=18)
plt.grid(True)

plt.savefig("experiments/plots/figure9b.pdf", format='pdf')

