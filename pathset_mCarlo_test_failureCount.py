import random
import matplotlib.pyplot as plt
import numpy as np

# Define pathsets as a list of edges.
pathsets = [
    [(1, 2), (2, 4)],
    [(1, 3), (3, 4)]
]

def simulate_packet_arrival(source, destination, pathsets, edge_probabilities, num_packets):
    failure_count = 0  # Initialize the failure count
    failure_rates = []

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

        if not success:
            failure_count += 1  # Increment failure count if the packet was not successfully transmitted

        failure_rate = failure_count / packet_number
        failure_rates.append(failure_rate)

    return failure_rates  # Return the failure rates

source = 1  # source and destination match node labels
destination = 4
num_packets = 10000

# Vary the success probability (probability_p) between 0 and 1
success_probability_values = np.linspace(0, 1, num=101)  # 101 values from 0 to 1

average_failure_rates = []

for probability_p in success_probability_values:
    edge_probabilities = {
        (1, 2): probability_p,
        (2, 4): probability_p,
        (1, 3): probability_p,
        (3, 4): probability_p
    }
    failure_rates = simulate_packet_arrival(source, destination, pathsets, edge_probabilities, num_packets)
    average_failure_rate = sum(failure_rates) / len(failure_rates)
    average_failure_rates.append(average_failure_rate)

# Calculate the values of the polynomial p^4 - 4p^3 + 4p^2 for the same range of success probabilities
polynomial_values = success_probability_values**4 - 4 * success_probability_values**3 + 4 * success_probability_values**2

# Plotting the failure rates and the polynomial
plt.plot(success_probability_values, average_failure_rates, label="Packet Arrival Failure Rate")
plt.plot(success_probability_values, 1 - polynomial_values, label="1 - (p^4 - 4p^3 + 4p^2)", linestyle='dashed')
plt.xlabel("Success Probability (probability_p)")
plt.ylabel("Values")
plt.title(f"Packet Arrival Failure Rate vs. Polynomial Comparison for {num_packets} packets")
plt.yscale('log')  # Set the y-axis to a logarithmic scale
plt.legend()
plt.grid(True)
plt.savefig(f"monte-carlo_failure_{num_packets}packets.png")
# plt.show()
