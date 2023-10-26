import random
import matplotlib.pyplot as plt
import numpy as np

# Define pathsets as a list of edges.
pathsets = [
    [(1, 2), (2, 4)],
    [(1, 3), (3, 4)]
]

def simulate_packet_arrival(source, destination, pathsets, edge_probabilities, num_packets):
    success_count = 0
    success_rates = []

    for packet_number in range(1, num_packets + 1):
        success = False
        attempts = 0

        while not success and attempts < len(pathsets):
            chosen_pathset = pathsets[attempts]
            success_probability = 1.0  # Initialize success probability for the pathset

            for edge in chosen_pathset:
                if edge in edge_probabilities:
                    success_probability *= edge_probabilities[edge]

            # Simulate packet transmission on the chosen pathset based on success probability
            if random.random() < success_probability:
                success = True
            attempts += 1

        if success:
            success_count += 1

        success_rate = success_count / packet_number
        success_rates.append(success_rate)

    return success_rates

source = 1  # Modify source and destination to match your node labels
destination = 4
num_packets = 10000

# Vary the success probability (probability_p) between 0 and 1
success_probability_values = np.linspace(0, 1, num=101)  # 101 values from 0 to 1

average_success_rates = []

for probability_p in success_probability_values:
    edge_probabilities = {
        (1, 2): probability_p,
        (2, 4): probability_p,
        (1, 3): probability_p,
        (3, 4): probability_p
        # (4, 5): probability_p,
        # (5, 6): probability_p,
        # (6, 8): probability_p,
        # (5, 7): probability_p,
        # (7, 8): probability_p
    }
    success_rates = simulate_packet_arrival(source, destination, pathsets, edge_probabilities, num_packets)
    average_success_rate = sum(success_rates) / len(success_rates)
    average_success_rates.append(average_success_rate)

# Plotting the success rates for different success probabilities
plt.plot(success_probability_values, average_success_rates)
plt.xlabel("Success Probability (probability_p)")
plt.ylabel("Average Success Rate")
plt.title("Packet Arrival Success Rate vs. Success Probability")
plt.yscale('log')  # Set the y-axis to a logarithmic scale
plt.grid(True)
plt.show()
