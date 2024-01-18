import random
import matplotlib.pyplot as plt
import numpy as np

# Define pathsets as a list of edges. NOW CUTSETS
# pathsets = [
#     # [(1, 2), (2, 4)],
#     # [(1, 3), (3, 4)]

#     [(1, 2), (2, 4)],
#     [(1, 2), (2, 10), (10,4)],
#     [(1, 3), (3, 10), (10,4)]

#     # [(1, 2), (2, 4)],
#     # [(1,4)],
#     # [(1, 3), (3, 4)],
#     # [(1,5), (5,4)]

#     #Collection K
#     # [(1, 2, 0)]
#     # [(1, 2, 0), (2, 3, 0)],
#     # [(1, 2, 0), (2, 3, 1)],
#     # [(1, 2, 0), (1, 2, 1)],
#     # [(2, 3, 0), (2, 3, 1)]
#     # [(1, 2, 0), (2, 3, 0), (2, 3, 1)],

#     # [(1, 2), (3, 4)],
#     # [(2, 4), (3, 4)],
#     # [(1, 2), (1, 3), (2, 4)],
#     # [(1, 2), (1, 3), (2, 3)],
#     # [(1, 2), (1, 3), (3, 4)],
#     # [(1, 2), (2, 4), (3, 4)],
#     # [(1, 2), (2, 3), (3, 4)],
#     # [(1, 3), (2, 4), (2, 3)],
#     # [(1, 3), (2, 4), (3, 4)],
#     # [(2, 4), (2, 3), (3, 4)],
#     # [(1, 2), (1, 3), (2, 4), (2, 3)],
#     # [(1, 2), (1, 3), (2, 4), (3, 4)],
#     # [(1, 2), (1, 3), (2, 3), (3, 4)],
#     # [(1, 2), (2, 4), (2, 3), (3, 4)],
#     # [(1, 3), (2, 4), (2, 3), (3, 4)],
#     # [(1, 2), (1, 3), (2, 4), (2, 3), (3, 4)]



# ]

# pathsets = [[ (2, 4), (4, 5), (5, 6), (6, 8)], [(2, 4), (4, 5), (5, 7), (7, 8)], [(1, 3), (3, 4), (4, 5), (5, 6), (6, 8)], [(1, 3), (3, 4), (4, 5), (5, 7), (7, 8)]]
# pathsets = [[(1, 2), (2, 3), (3, 5)], [(1, 2), (2, 4), (4, 5)]] #TESTGRAPHA
# pathsets = [[(2, 3), (3, 5)], [(2, 4), (4, 5)]] #TESTGRAPHB
# pathsets = [[(1, 2), (2, 3), (3, 5), (5, 6)], [(1, 2), (2, 4), (4, 5), (5, 6)]] #TESTGRAPHC
# pathsets = [[(1, 2), (2, 3), (3, 5), (5, 7)], [(1, 2), (2, 4), (4, 6), (6, 7)]] #TESTGRAPHD
# pathsets = [[(1, 2), (2, 3), (3, 5), (5, 7), (7, 9), (9, 11), (11, 13), (13, 15)], [(1, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 15)]] #TESTGRAPHE
pathsets = [[(1, 2), (2, 4), (4, 5), (5, 6), (6, 8)], [(1, 2), (2, 4), (4, 5), (5, 7), (7, 8)], [(1, 3), (3, 4), (4, 5), (5, 6), (6, 8)], [(1, 3), (3, 4), (4, 5), (5, 7), (7, 8)]] #TESTGRAPHF
# pathsets = [[(1, 2), (2, 6)], [(1, 3), (3, 6)], [(1, 4), (4, 6)], [(1, 5), (5, 6)]] #TESTGRAPHG
# pathsets = [[(1, 2), (2, 3)]] #TESTGRAPHH

def verify_number_already_attributed_to_edge(edge_to_test, random_number, edge_number_association, pathsets) :
    # Verify if edge already has a random number associated with it
    if edge_to_test not in edge_number_association:
        # Associate the random number to it
        edge_number_association[edge_to_test] = random_number
        # print("Now edge:", edge_to_test, "has been attributed the random number:", random_number)
    else:
        a = 1
        # Edge already has a random number associated with it
        # print("Edge already has a random number:", edge_number_association[edge_to_test])
    return edge_number_association


def simulate_packet_arrival(source, destination, pathsets, edge_probabilities, num_packets):
    failure_count = 0
    failure_rates = []

    for packet_number in range(1, num_packets + 1):
        failure = 0
        attempts = 0
        tx_states = []
        # For every packet
        edge_number_association = {}
        # Uniquely associate random failure probability to every edge of the pathsets
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
                edge_number_association = verify_number_already_attributed_to_edge(edge, random_number, edge_number_association, pathsets)
                if edge_number_association.get(edge) <= edge_probability:
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
num_packets = 50000

# Vary the failure probability (probability_p) between 0 and 1
failure_probability_values = np.linspace(0, 1, num=101)  # 101 values from 0 to 1

average_failure_rates = []

for probability_p in failure_probability_values:
    edge_probabilities = {
        # (1, 2, 0): probability_p,
        # (1, 2, 1): probability_p,
        # (2, 3, 0): probability_p,
        # (2, 3, 1): probability_p,
        
        # (1, 2): probability_p,
        # (2, 4): probability_p,
        # (1, 3): probability_p,
        # (3, 4): probability_p

        # (1, 2): probability_p,
        # (2, 4): probability_p,
        # (1, 3): probability_p,
        # (3, 10): probability_p,
        # (2, 10): probability_p,
        # (10, 4): probability_p,

        (1, 2): probability_p,
        (1, 3): probability_p,
        (2, 4): probability_p,
        (3, 4): probability_p,
        (4, 5): probability_p,
        (5, 6): probability_p,
        (6, 8): probability_p,
        (5, 7): probability_p,
        (7, 8): probability_p

        # (1, 2): probability_p,
        # (1, 3): probability_p,
        # (1, 4): probability_p,
        # (1, 5): probability_p,
        # (2, 6): probability_p,
        # (3, 6): probability_p,
        # (4, 6): probability_p,
        # (5, 6): probability_p



        # (1, 2): probability_p,
        # (2, 3): probability_p,
        # (2, 4): probability_p,
        # (3, 5): probability_p,
        # (4, 5): probability_p


        # (1, 2): probability_p,
        # (2, 3): probability_p


        # (1, 2): probability_p,
        # (2, 3): probability_p,
        # (3, 5): probability_p,
        # (5, 7): probability_p,
        # (7, 9): probability_p,
        # (9, 11): probability_p,
        # (11, 13): probability_p,
        # (13, 15): probability_p,
        # (2, 4): probability_p,
        # (4, 6): probability_p,
        # (6, 8): probability_p,
        # (8, 10): probability_p,
        # (10, 12): probability_p,
        # (12, 14): probability_p,
        # (14, 15): probability_p
    }
    if probability_p == 0.45 :
        print("HERE")
    failure_rates = simulate_packet_arrival(source, destination, pathsets, edge_probabilities, num_packets)
    average_failure_rate = failure_rates[-1]#sum(failure_rates) / len(failure_rates)
    average_failure_rates.append(average_failure_rate)
    print(f"Probability: {probability_p}, Simulated Failure Rate: {average_failure_rate}, Polynomial Value: {probability_p**9 -9*probability_p**8 +32*probability_p**7 - 56*probability_p**6 +46*probability_p**5 -6*probability_p**4 -16*probability_p**3 +8*probability_p**2 +probability_p**1}")
# Calculate the values of the polynomial p^4 - 4p^3 + 4p^2 for the same range of failure probabilities
# polynomial_values = failure_probability_values**4 - 4 * failure_probability_values**3 + 4 * failure_probability_values**2
# polynomial_values = failure_probability_values**2  - failure_probability_values**3 + failure_probability_values
# polynomial_values =4*failure_probability_values**2 -2*failure_probability_values**3 -4*failure_probability_values**4 +4*failure_probability_values**5 -failure_probability_values**6  #failure_probability_values**5 - 4*failure_probability_values**4 + 4*failure_probability_values**3
# polynomial_values = failure_probability_values**6 -6*failure_probability_values**5 +13*failure_probability_values**4 -12*failure_probability_values**3 +3*failure_probability_values**2 +2*failure_probability_values**1
# polynomial_values = -failure_probability_values**7 + 7*failure_probability_values**6 -21*failure_probability_values**5 +33*failure_probability_values**4 -27*failure_probability_values**3 +9*failure_probability_values**2 +failure_probability_values**1
# polynomial_values = -failure_probability_values**15 +15*failure_probability_values**14 -105*failure_probability_values**13 +455*failure_probability_values**12 -1365*failure_probability_values**11 + 3003*failure_probability_values**10 -5005*failure_probability_values**9 +6433*failure_probability_values**8 -6419*failure_probability_values**7 + 4949*failure_probability_values**6 -2891*failure_probability_values**5 +1225*failure_probability_values**4 -343*failure_probability_values**3 +49*failure_probability_values**2 +failure_probability_values**1
polynomial_values = failure_probability_values**9 -9*failure_probability_values**8 +32*failure_probability_values**7 - 56*failure_probability_values**6 +46*failure_probability_values**5 -6*failure_probability_values**4 -16*failure_probability_values**3 +8*failure_probability_values**2 +failure_probability_values**1
# polynomial_values = failure_probability_values**8 -8*failure_probability_values**7 +24*failure_probability_values**6 -32*failure_probability_values**5 +16*failure_probability_values**4
# polynomial_values = -failure_probability_values**2 +2*failure_probability_values**1
# polynomial_values = -failure_probability_values**5 +5*failure_probability_values**4 -8*failure_probability_values**3 +4*failure_probability_values**2 +failure_probability_values**1


# Plotting the failure rates and the polynomial
plt.plot(failure_probability_values, average_failure_rates, label="Packet Arrival failure Rate")
plt.plot(failure_probability_values, polynomial_values, label="p^9 - 9p^8 + 32p^7 - 56p^6 + 46p^5 - 6p^4 - 16p^3 + 8p^2 + p", linestyle='dashed')
plt.xlabel("failure Probability (probability_p)")
plt.ylabel("Values")
plt.title(f"Packet Arrival failure Rate vs. Polynomial Comparison for {num_packets} packets")
plt.yscale('log')  # Set the y-axis to a logarithmic scale
plt.legend()
plt.grid(True)
plt.savefig(f"monte-carlo_failure_{num_packets}packets_TESTGRAPHF_NEW.png")
# plt.show()