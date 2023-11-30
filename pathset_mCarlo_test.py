import random
import matplotlib.pyplot as plt
import numpy as np

# Define pathsets as a list of edges. NOW CUTSETS
pathsets = [
    # [(1, 2), (2, 4)],
    # [(1, 3), (3, 4)]



    [(1, 2), (2, 4)],
    [(1,4)],
    [(1, 3), (3, 4)]

    #Collection K
    # [(1, 2, 0)]
    # [(1, 2, 0), (2, 3, 0)],
    # [(1, 2, 0), (2, 3, 1)],
    # [(1, 2, 0), (1, 2, 1)],
    # [(2, 3, 0), (2, 3, 1)]
    # [(1, 2, 0), (2, 3, 0), (2, 3, 1)],

    # [(1, 2), (3, 4)],
    # [(2, 4), (3, 4)],
    # [(1, 2), (1, 3), (2, 4)],
    # [(1, 2), (1, 3), (2, 3)],
    # [(1, 2), (1, 3), (3, 4)],
    # [(1, 2), (2, 4), (3, 4)],
    # [(1, 2), (2, 3), (3, 4)],
    # [(1, 3), (2, 4), (2, 3)],
    # [(1, 3), (2, 4), (3, 4)],
    # [(2, 4), (2, 3), (3, 4)],
    # [(1, 2), (1, 3), (2, 4), (2, 3)],
    # [(1, 2), (1, 3), (2, 4), (3, 4)],
    # [(1, 2), (1, 3), (2, 3), (3, 4)],
    # [(1, 2), (2, 4), (2, 3), (3, 4)],
    # [(1, 3), (2, 4), (2, 3), (3, 4)],
    # [(1, 2), (1, 3), (2, 4), (2, 3), (3, 4)]



]

def simulate_packet_arrival(source, destination, pathsets, edge_probabilities, num_packets):
    failure_count = 0
    failure_rates = []

    for packet_number in range(1, num_packets + 1):
        failure = 0
        attempts = 0
        tx_states = []
        while attempts < len(pathsets):
            # For each pathset or cutset
            chosen_pathset = pathsets[attempts]
            failure_probability = 1.0  # Initialize failure probability for the pathset
            failure = 0
            failure_state = []
            for edge in chosen_pathset:
                edge_probability = edge_probabilities.get(edge, 0)  # Fetch edge probability or default to 0
                random_number_list = np.random.uniform(0,1,1)
                random_number = random_number_list[-1]
                if random_number <= edge_probability:
                    failure = 1                                        
                    failure_state.append(failure)                    
            if sum(failure_state) >= 1 :
                # packet did not arrive!
                #failure_count +=1
                tx_states.append(0)
                # break
            if sum(failure_state) == 0 : 
                # packet arrived and one transmission is successful!get out of while loop               
                tx_states.append(1)
                break
                    
            # if edge_probability == 0.38 :
            #     edge_probability, 
            # increment number of attempts and change the considered pathset/cutset :   
            attempts += 1
        if sum(tx_states) == 0:
            # No transmission succeeded, increment failure_count
            failure_count +=1
        if (packet_number == num_packets-1) :
            print("HERE 2")
        failure_rate = failure_count / num_packets
        failure_rates.append(failure_rate)

    return failure_rates

source = 1  # source and destination match node labels
destination = 3
num_packets = 150000

# Vary the failure probability (probability_p) between 0 and 1
failure_probability_values = np.linspace(0, 1, num=101)  # 101 values from 0 to 1

average_failure_rates = []

for probability_p in failure_probability_values:
    edge_probabilities = {
        # (1, 2, 0): probability_p,
        # (1, 2, 1): probability_p,
        # (2, 3, 0): probability_p,
        # (2, 3, 1): probability_p,
        
        (1, 2): probability_p,
        (2, 4): probability_p,
        (1, 3): probability_p,
        (3, 4): probability_p,
        (1, 4): probability_p
        # (4, 5): probability_p,
        # (5, 6): probability_p,
        # (6, 8): probability_p,
        # (5, 7): probability_p,
        # (7, 8): probability_p
    }
    if probability_p == 0.45 :
        print("HERE")
    failure_rates = simulate_packet_arrival(source, destination, pathsets, edge_probabilities, num_packets)
    average_failure_rate = failure_rates[-1]#sum(failure_rates) / len(failure_rates)
    average_failure_rates.append(average_failure_rate)
    print(f"Probability: {probability_p}, Simulated Failure Rate: {average_failure_rate}, Polynomial Value: {probability_p**5 - 4*probability_p**4 + 4*probability_p**3}")
# Calculate the values of the polynomial p^4 - 4p^3 + 4p^2 for the same range of failure probabilities
# polynomial_values = failure_probability_values**4 - 4 * failure_probability_values**3 + 4 * failure_probability_values**2
# polynomial_values = failure_probability_values**2  - failure_probability_values**3 + failure_probability_values
polynomial_values = failure_probability_values**5 - 4*failure_probability_values**4 + 4*failure_probability_values**3

# Plotting the failure rates and the polynomial
plt.plot(failure_probability_values, average_failure_rates, label="Packet Arrival failure Rate")
plt.plot(failure_probability_values, polynomial_values, label="p^5 -4p^4 +4p^3", linestyle='dashed')
plt.xlabel("failure Probability (probability_p)")
plt.ylabel("Values")
plt.title(f"Packet Arrival failure Rate vs. Polynomial Comparison for {num_packets} packets")
plt.yscale('log')  # Set the y-axis to a logarithmic scale
plt.legend()
plt.grid(True)
plt.savefig(f"monte-carlo_failure_{num_packets}packets.png")
# plt.show()