import random
import numpy as np
import matplotlib.pyplot as plt
def simulate_packet_arrival(path, edge_probability):
    """
    Simulate the arrival of a packet from source to target along a path based on edge probabilities.

    Args:
    - path (list of edges): The path from source to target, represented as a list of edges.
    - edge_probabilities (dict): A dictionary where keys are edges (as tuples) and values are edge probabilities.

    Returns:
    - bool: True if the packet successfully arrived, False otherwise.
    """
    probability = 1.0

    for edge in path:
        # if edge in edge_probabilities:
        #     probability *= edge_probabilities[edge]
        # else:
        #     # If the edge probability is not provided, assume a default value, e.g., 1.0 (always succeeds).
        #     probability *= 1.0
        probability *= edge_probability

    # Simulate packet arrival based on calculated probability
    return random.random() < probability

def trial_monte_carlo(total_trial_number, path, edge_probability) :
    average_log = []
    x = np.arange(1, total_trial_number, 1)
    for trial_number in x :
        log_packet_arrival = []
        for trial in range(1,trial_number+1) :

            # Simulate packet arrival.
            packet_arrived = simulate_packet_arrival(path, edge_probability)
            log_packet_arrival.append(packet_arrived)

        # Calculate the average of elements in log_packet_arrival
        average_packet_arrival = sum(log_packet_arrival) / len(log_packet_arrival)
        average = average_packet_arrival
        average_log.append(average)
    average_success_over_trials = average_log[-1]
    return average_log, average_success_over_trials

# Define the path from source to target as a list of edges (as tuples).
path = [(1, 2), (2, 3), (3, 4)]

# Define edge probabilities as a dictionary.
probability_p = 0.7
edge_probabilities = {
    (1, 2): probability_p,  # Probability of success for edge (1, 2) is p
    (2, 3): probability_p,  # Probability of success for edge (2, 3) is p
    (3, 4): probability_p  # Probability of success for edge (3, 4) is p
}
print("Edge probabilies = ", edge_probabilities)
edge_probability = probability_p
total_trial_number = round(10000*1/probability_p)
average_log, average_success_over_trials = trial_monte_carlo(total_trial_number, path, edge_probability)


x = np.arange(1, total_trial_number, 1)
plt.plot(x,average_log)
plt.title("Monte-Carlo simulation for probability p = 0.7 and 14000+ trials")
plt.xlabel("Total trials")
plt.ylabel("Average ratio of packets received")
plt.savefig("p07_10000attempts_montecarlo.png")
print("Last value of p other : ", average_success_over_trials)
plt.show()
# # Print the result.
# if packet_arrived:
#     print("The packet successfully arrived!")
# else:
#     print("The packet did not arrive.")