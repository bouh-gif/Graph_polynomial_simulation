import networkx as nx
from itertools import combinations
from itertools import chain, combinations
from networkx.algorithms.connectivity import build_auxiliary_node_connectivity
from networkx.algorithms.connectivity import minimum_st_edge_cut
from networkx.algorithms.connectivity import minimum_edge_cut
from networkx.algorithms.flow import build_residual_network
from networkx.algorithms.flow import minimum_cut
import matplotlib.pyplot as plt
from collections import Counter
import sympy
import numpy
import matplotlib.pyplot as plt
from itertools import chain
from networkx.algorithms.connectivity import k_edge_components
import random

# GLOBAL VARIABLE, to be run once
legend = []

# Class of function for polynomial evaluation :
class OutagePolynomial(sympy.Function):
  @classmethod
  # defining subclass function with 'eval()' class method
  def eval(cls, p, minimum_cut_size, number_of_edges, coefs):
    m = minimum_cut_size
    n = number_of_edges
    result = []
    UpperB = []
    for element in p:
        # Core of the function : the algebraic expression
        # m=1 # size of any minimum cut-set (see collection M)
        # n=3 # number of edges
        # #values of coefficients A_i, given by the number of cut-sets of size i, with i between m and n included
        # coefs = [1, 3, 1]
        # changement de variable : x = p/(1-p) et p = element
        # x = element/(1.0-element)
        UpperB.append((1-(1-element)**2)**2)
        index = list(range(m, n+1))
        sum=0
        for  i in index:
            # Evaluate sum = A(x) = A(p/(1-p)) :
            A_i = coefs[i-m]
            # sum = sum + A_i * pow(x, i)
            sum = sum + A_i*(pow(element, i))*(pow(1-element, n-i))
        # Evaluate O(p)
        # result = pow((1.0-p), n) * sum

        result.append(sum)
    return result, UpperB
  @classmethod
  # defining subclass function with 'eval()' class method
  def plot(cls, x, y, UpperB, string_expanded_polynomial, marker=None):
    ### PLOT
    # convert y-axis to Logarithmic scale
    plt.yscale("log")
    if marker==None:
        # plotting the points
        plt.plot(x, y)
    else:
        # plotting the points
        plt.plot(x, y, marker, ms = 1)
    # plt.plot(x,UpperB)
    # legend
    legend.append(string_expanded_polynomial)
    plt.legend(legend)
    # naming the x axis
    plt.xlabel('p')
    # naming the y axis
    plt.ylabel('O(p)')

    # giving a title to my graph
    plt.title('Outage polynomial O(p)')

    # Add gridlines to the plot
    plt.grid(visible=True, which='major', linestyle='-')
    plt.grid(visible=True, which='minor', linestyle='--')
    # grid(b=True, which='major', color='b', linestyle='-')

    # Save the file and show the figure
    plt.savefig("plotted_polynomial.png")


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def find_all_minimum_cuts(graph, source, target, capacity_key):

    # Find all possible sets of edges in the graph
    all_edge_combinations = powerset(graph.edges(keys=True, data=True))
    all_minimum_cuts = []

    for edge_combination in all_edge_combinations:
        # Create a new graph and remove the selected edges
        temp_graph = graph.copy()
        temp_graph.remove_edges_from(edge_combination)

        # Check if the graph is disconnected between the source and target nodes
        if not nx.has_path(temp_graph, source, target):
            all_minimum_cuts.append(edge_combination)

    return all_minimum_cuts


def get_collection(all_min_cuts) :

    K = [] # list of all cut sets
    L = [] # List of all minimal cut sets
    M = [] # List of all minimum cut sets
    for min_cut_set in all_min_cuts:
        K.append(min_cut_set)
        print(min_cut_set)
        list_edge_combination = list(powerset(min_cut_set))
        list_edge_combination.remove(min_cut_set)
        # Verify if edge_combination is a cut-set, i.e. if edge_combination can be found in all_min_cuts :
        flag_is_present = 0
        for edge_combination in list_edge_combination:

            for one_min_cut in all_min_cuts:
                if edge_combination == one_min_cut:
            #     # don't add current min_cut_set value to L because edge_combination already exists as a minimum cut set
                    flag_is_present = 1
                    break
        # If edge_combination is not a cut-set for every member of list_edge_combination, add min_cut_set to L :
        if flag_is_present == 0 :
            L.append(min_cut_set)
    # Create collection of minimum cut-sets M : a minimum cut-set is a cut-set that is of minimum size
    minimum_size = 10000000000000000000000000 #big value, will be replaced when entering below for loop
    for cut_set in L:
        temp_size = len(cut_set)
        if temp_size < minimum_size :
            minimum_size = temp_size
    for cut_set in L:
        if len(cut_set) == minimum_size:
            M.append(cut_set)
    return K, L, M
def display_collection(graph_name, K, L, M):
    with open('log.txt', 'a') as log_file:
        print("FOR GRAPH : ", graph_name, file=log_file)
        print("Here is K the collection of all s-t separating cut-sets: a cut-set is a subset of edges whose removal from the network disconnects nodes s and node t  ", file=log_file)
        for cut_set in K:
            print(cut_set, file=log_file)
        print("Here is L the collection of all minimal cut-sets in the graph : a minimal cut-set is a subset of edges whose removal from the network disconnects nodes s and node t and from which no subset can be called a cut-set : ", file=log_file)
        for minimal_cut_set in L:
            print(minimal_cut_set, file=log_file)
        print("Here is M the collection of all minimum cut-sets in the graph : a minimum cut-set is a subset of edges whose removal from the network disconnects nodes s and node t and from which no subset can be called a cut-set and is of minimum size : ", file=log_file)
        for minimum_cut_set in M:
            print(minimum_cut_set, file=log_file)


def get_n_m_values(G, M) :
    number_of_edges = G.number_of_edges() # noted n in Kschischang's paper
    minimum_cut_size = len(M[0]) # noted m in Kschischang's paper
    return number_of_edges, minimum_cut_size
def find_coefs(K) :
    # Finding coefficients for A(x) in reliability polynomial : Am = number of cut-sets of size m, Am+1 = number of cut-sets of size m+1 etc...
    # Iterating over collection K
    temp_list_of_cut_set_sizes = []
    for cut_set in K:
        # get size of cut-set and store it if not already stored
        temp_list_of_cut_set_sizes.append(len(cut_set))
    # order list in ascending order
    list_of_cut_set_sizes = sorted(temp_list_of_cut_set_sizes)
    # coefficients for A(x) 
    coefs = list(dict(Counter(list_of_cut_set_sizes)).values())
    return coefs

def get_polynomial_expression(coefs, minimum_cut_size, number_of_edges) :  
    m = minimum_cut_size
    n = number_of_edges
    x, y, z = sympy.symbols('x y z')
    polynomial = 0
    indices = list(range(m, n+1))
    # list_indices = [index for index, _ in enumerate(indices)]
    for i in indices :
        polynomial = polynomial + coefs[i-m]*pow(x,i)*pow((1-x),(n-i))
        # print(i)
    string_expanded_polynomial = ((str(sympy.expand(polynomial)).replace('x', 'p')).replace('**', '^')).replace('*', '·')
    return string_expanded_polynomial
def multiply_two_lists(list1, list2, list3) :
    # Multiplying two lists
    # of same length
    res_list = []
    for i in range(0, len(list1)):
        res_list.append(1 - ((1-list1[i]) * (1-list2[i]) * (1-list3[i])))    
    return res_list
    
def draw_graph(G, path) :
    ### DRAWING THE GRAPH
    # Draw and save the graph
    plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference
    pos = nx.planar_layout(G)
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='c', node_size=100, alpha=1)
    # Draw edges as separate curves
    for u, v, key in G.edges(keys=True):
        connection_style = "arc3,rad=" + str(0.3 * key)
        plt.annotate("",
                    xy=pos[u], xycoords='data',  # Source node
                    xytext=pos[v], textcoords='data',  # Target node
                    arrowprops=dict(arrowstyle="->", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle=connection_style),
                    )        
    # Draw node labels
    node_labels = {n: n for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    # Turn off axis
    plt.axis('off')
    # Save the file and show the figure 
    plt.savefig(path)
    # plt.show()

# def compute_pathsets(graph, source, target):
#     paths = list(nx.all_simple_paths(graph, source=source, target=target))
#     return paths

def compute_all_edge_pathsets(graph, source, target):
    all_edge_pathsets = []

    def dfs(node, current_path):
        if node == target:
            all_edge_pathsets.append(tuple(current_path))
        for neighbor in graph.neighbors(node):
            edge = (node, neighbor)
            if edge not in current_path:
                dfs(neighbor, current_path + [edge])

    dfs(source, [])
    return all_edge_pathsets

# def compute_and_visualize_pathsets(graph, source, target, pathsets):
#     pos = nx.spring_layout(graph)
#     plt.figure(figsize=(8, 6))
    
#     # Draw nodes
#     nx.draw_networkx_nodes(graph, pos, node_color='c', node_size=100, alpha=1)
    
#     # Draw edges as separate curves
#     for u, v, key in graph.edges(keys=True):
#         connection_style = "arc3,rad=" + str(0.3 * key)
#         plt.annotate("",
#                     xy=pos[u], xycoords='data',  # Source node
#                     xytext=pos[v], textcoords='data',  # Target node
#                     arrowprops=dict(arrowstyle="->", color="0.5",
#                                     shrinkA=5, shrinkB=5,
#                                     patchA=None, patchB=None,
#                                     connectionstyle=connection_style),
#                     )
    
#     # Draw node labels
#     node_labels = {n: n for n in graph.nodes}
#     nx.draw_networkx_labels(graph, pos, labels=node_labels)
    
#     # Visualize pathsets
#     for i, path in enumerate(pathsets):
#         color = plt.cm.viridis(i / len(pathsets))  # Use a colormap for path coloring
#         for j in range(len(path) - 1):
#             u, v = path[j], path[j + 1]
#             plt.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=color)
    
#     # Turn off axis
#     plt.axis('off')
    
#     plt.show()

def compute_and_visualize_edge_pathsets_with_highlighted_arrows(G, s_node, t_node, all_pathsets, pathset, one_edge_probability):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='c', node_size=100, alpha=1)
    
    # Draw edges as separate curves
    for u, v, key in G.edges(keys=True):
        connection_style = "arc3,rad=" + str(0.3 * key)
        plt.annotate("",
                    xy=pos[u], xycoords='data',  # Source node
                    xytext=pos[v], textcoords='data',  # Target node
                    arrowprops=dict(arrowstyle="->", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle=connection_style),
                    )
    
    # Draw node labels
    node_labels = {n: n for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    
    # Visualize pathsets
    for i, path in enumerate(all_pathsets):
        color = plt.cm.viridis(i / len(all_pathsets))  # Use a colormap for path coloring
        for j in range(len(path) - 1):
            u, v = path[j], path[j + 1]
            if path in pathset:
                plt.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="red", linewidth=2)  # Highlight in red
            else:
                plt.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=color)
    
    # Turn off axis
    plt.axis('off')
    
    plt.show()

def simulate_packet_arrival(path, edge_probabilities, one_edge_probability):
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
        probability *= one_edge_probability
        # else:
        #     # If the edge probability is not provided, assume a default value, e.g., 1.0 (always succeeds).
        #     probability *= 1.0
        # probability *= edge_probability

    # Simulate packet arrival based on calculated probability
    return random.random() < probability


def trial_monte_carlo(total_trial_number, path, edge_probabilities, one_edge_probability) :
    average_log = []
    x = numpy.arange(1, total_trial_number, 1)
    for trial_number in x :
        log_packet_arrival = []
        for trial in range(1,trial_number+1) :

            # Simulate packet arrival.
            packet_arrived = simulate_packet_arrival(path, edge_probabilities, one_edge_probability)
            log_packet_arrival.append(packet_arrived)

        # Calculate the average of elements in log_packet_arrival
        average_packet_arrival = sum(log_packet_arrival) / len(log_packet_arrival)
        average = average_packet_arrival
        average_log.append(average)
    average_success_over_trials = average_log[-1]
    return average_log, average_success_over_trials

def create_dict_edge_probabilities(all_pathsets, one_edge_probability) :
    # one_edge_probability = 0.9  # change this probability as needed

    # Create a dictionary to associate every edge with the edge-probability pair
    edge_probabilities = {}

    # Iterate through all_edge_pathsets and populate the edge_probabilities dictionary
    for pathset in all_pathsets:
        for edge in pathset:
            edge_probabilities[edge] = one_edge_probability
    return edge_probabilities

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
                random_number_list = numpy.random.uniform(0,1,1)
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
        # if (packet_number == num_packets-1) :
        #     print("HERE 2")
        failure_rate = failure_count / num_packets
        failure_rates.append(failure_rate)

    return failure_rates



### CREATE THE GRAPH G0
G = nx.MultiDiGraph(directed=True)
# Nodes 
G.add_node(1, subset='source')
G.add_node(8, subset='target')
s_node = 1 # source node
t_node = 8 # target node
G.add_node(2, subset='others')
G.add_node(3, subset='others')
G.add_node(4, subset='others')
G.add_node(5, subset='others')
G.add_node(6, subset='others')
G.add_node(7, subset='others')
# Edges 
G.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (1, 3, 0, {'capacity': "e2"}), (2, 4, 0, {'capacity': "e3"}), (3, 4, 0, {'capacity': "e4"})])
G.add_edges_from([(4, 5, 0, {'capacity': "e5"}), (5, 6, 0, {'capacity': "e6"}), (5, 7, 0, {'capacity': "e7"}), (6, 8, 0, {'capacity': "e8"}), (7, 8, 0, {'capacity': "e9"})])

### CREATE THE GRAPH G1
G1 = nx.MultiDiGraph(directed=True)
# Nodes 
G1.add_node(1, subset='source')
G1.add_node(3, subset='target')
s_node1 = 1 # source node
t_node1 = 4 # target node
G1.add_node(2, subset='others')
G1.add_node(3, subset='others')
# G1.add_node(4, subset='others')
# G1.add_node(5, subset='others')
# G1.add_node(6, subset='others')
# G1.add_node(7, subset='others')
# G1.add_node(8, subset='others')
# G1.add_node(9, subset='others')
# G1.add_node(10, subset='others')
# G1.add_node(11, subset='others')
# G1.add_node(12, subset='others')
# G1.add_node(13, subset='others')
# G1.add_node(14, subset='others')
# Edges 
G1.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (1, 3, 0, {'capacity': "e2"}), (3, 4, 0, {'capacity': "e4"}), (2, 4, 0, {'capacity': "e3"})])
# G1.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (1, 3, 0, {'capacity': "e2"}), (3, 10, 0, {'capacity': "e310"}), (2, 4, 0, {'capacity': "e3"}), (2, 10, 0, {'capacity': "e210"}), (10, 4, 0, {'capacity': "e104"})])
# G1.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (2, 3, 0, {'capacity': "e2"}), (3, 5, 0, {'capacity': "e310"}), (2, 4, 0, {'capacity': "e3"}), (4, 6, 0, {'capacity': "e210"}), (5, 7, 0, {'capacity': "e211"}), (6, 7, 0, {'capacity': "e212"})])
# G1.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (2, 3, 0, {'capacity': "e2"}), (3, 5, 0, {'capacity': "e310"}), (5, 7, 0, {'capacity': "e57"}), (7, 9, 0, {'capacity': "e79"}), (9, 11, 0, {'capacity': "e911"}), (11, 13, 0, {'capacity': "e1113"}), (13, 15, 0, {'capacity': "e1315"}),  (2, 4, 0, {'capacity': "e3"}), (4, 6, 0, {'capacity': "e210"}), (6, 8, 0, {'capacity': "e68"}), (8, 10, 0, {'capacity': "e810"}), (10, 12, 0, {'capacity': "e1012"}), (12, 14, 0, {'capacity': "e1214"}), (14, 15, 0, {'capacity': "e1415"}) ])
# G1.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (1, 3, 0, {'capacity': "e2"}), (1, 4, 0, {'capacity': "e3"}), (1, 5, 0, {'capacity': "e4"}), (2, 6, 0, {'capacity': "e5"}), (3, 6, 0, {'capacity': "e6"}), (4, 6, 0, {'capacity': "e7"}), (5, 6, 0, {'capacity': "e8"})])
# G1.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (2, 3, 0, {'capacity': "e2"})])


# G1.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (1, 3, 0, {'capacity': "e2"}), (2, 4, 0, {'capacity': "e3"}), (3, 4, 0, {'capacity': "e4"}), (4, 5, 0, {'capacity': "e5"}), (5, 6, 0, {'capacity': "e6"}), (5, 7, 0, {'capacity': "e7"}), (6, 8, 0, {'capacity': "e8"}), (7, 8, 0, {'capacity': "e9"})])
# G1.add_edges_from([(4, 5, 0, {'capacity': "e5"}), (5, 6, 0, {'capacity': "e6"}), (5, 7, 0, {'capacity': "e7"}), (6, 8, 0, {'capacity': "e8"}), (7, 8, 0, {'capacity': "e9"})])

### CREATE THE GRAPH G2
G2 = nx.MultiDiGraph(directed=True)
# Nodes 
G2.add_node(4, subset='source')
G2.add_node(5, subset='target')
s_node2 = 4 # source node
t_node2 = 5 # target node

# Edges 
# G2.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (1, 3, 0, {'capacity': "e2"}), (2, 4, 0, {'capacity': "e3"}), (3, 4, 0, {'capacity': "e4"})])
G2.add_edges_from([(4, 5, 0, {'capacity': "e5"})])

### CREATE THE GRAPH G3
G3= nx.MultiDiGraph(directed=True)
# Nodes 
G3.add_node(5, subset='source')
G3.add_node(8, subset='target')
s_node3 = 5 # source node
t_node3 = 8 # target node
G3.add_node(6, subset='others')
G3.add_node(7, subset='others')
# Edges 
# G2.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (1, 3, 0, {'capacity': "e2"}), (2, 4, 0, {'capacity': "e3"}), (3, 4, 0, {'capacity': "e4"})])
G3.add_edges_from([(5, 6, 0, {'capacity': "e6"}), (5, 7, 0, {'capacity': "e7"}), (6, 8, 0, {'capacity': "e8"}), (7, 8, 0, {'capacity': "e9"})])

#### GRAPH 0 HANDLING
### CALL TO CUSTOM FLOW COMPUTING FUNCTION
# Find all minimum edge cuts in the graph
all_min_cuts = find_all_minimum_cuts(G, source=s_node, target=t_node, capacity_key='capacity')

### EVALUATING K, L, AND M (collection of cut-sets, minimal cut-sets and minimum cut-sets)
K, L, M = get_collection(all_min_cuts)
display_collection("G", K,L,M)
number_of_edges, minimum_cut_size = get_n_m_values(G, M)
coefs = find_coefs(K)
string_expanded_polynomial = get_polynomial_expression(coefs, minimum_cut_size, number_of_edges)
p = numpy.linspace(0.01, 0.99, num=101)
O_p, UpperB = OutagePolynomial.eval(p, minimum_cut_size, number_of_edges, coefs)
# plt.scatter(p, O_p, marker="*") 
OutagePolynomial.plot(p, O_p, UpperB, string_expanded_polynomial)

#### GRAPH 1 HANDLING
# Find all minimum edge cuts in the graph
all_min_cuts1 = find_all_minimum_cuts(G1, source=s_node1, target=t_node1, capacity_key='capacity')

### EVALUATING K, L, AND M (collection of cut-sets, minimal cut-sets and minimum cut-sets)
K1, L1, M1 = get_collection(all_min_cuts1)
display_collection("G1", K1, L1, M1)
number_of_edges1, minimum_cut_size1 = get_n_m_values(G1, M1)
coefs1 = find_coefs(K1)
string_expanded_polynomial1 = get_polynomial_expression(coefs1, minimum_cut_size1, number_of_edges1)
p1 = numpy.linspace(0.01, 0.99, num=101)
O_p1, UpperB1 = OutagePolynomial.eval(p1, minimum_cut_size1, number_of_edges1, coefs1)
OutagePolynomial.plot(p1, O_p1, UpperB1, string_expanded_polynomial1)


#### GRAPH 2 HANDLING
# Find all minimum edge cuts in the graph
all_min_cuts2 = find_all_minimum_cuts(G2, source=s_node2, target=t_node2, capacity_key='capacity')

### EVALUATING K, L, AND M (collection of cut-sets, minimal cut-sets and minimum cut-sets)
K2, L2, M2 = get_collection(all_min_cuts2)
display_collection("G2", K2, L2, M2)
number_of_edges2, minimum_cut_size2 = get_n_m_values(G2, M2)
coefs2 = find_coefs(K2)
string_expanded_polynomial2 = get_polynomial_expression(coefs2, minimum_cut_size2, number_of_edges2)
p2 = numpy.linspace(0.01, 0.99, num=101)
O_p2, UpperB2 = OutagePolynomial.eval(p2, minimum_cut_size2, number_of_edges2, coefs2)
OutagePolynomial.plot(p2, O_p2, UpperB2, string_expanded_polynomial2)

# O_p12 = multiply_two_lists(O_p1, O_p2)
# UpperB12 = [] # Upper bound for this case is nothing for now, to be changed later on
# OutagePolynomial.plot(p2, O_p12, UpperB2, " Two polynomial multiplication")

#### GRAPH 3 HANDLING
# Find all minimum edge cuts in the graph
all_min_cuts3 = find_all_minimum_cuts(G3, source=s_node3, target=t_node3, capacity_key='capacity')

### EVALUATING K, L, AND M (collection of cut-sets, minimal cut-sets and minimum cut-sets)
K3, L3, M3 = get_collection(all_min_cuts3)
display_collection("G3", K3, L3, M3)
number_of_edges3, minimum_cut_size3 = get_n_m_values(G3, M3)
coefs3 = find_coefs(K3)
string_expanded_polynomial3 = get_polynomial_expression(coefs3, minimum_cut_size3, number_of_edges3)
p3 = numpy.linspace(0.01, 0.99, num=101)
O_p3, UpperB3 = OutagePolynomial.eval(p3, minimum_cut_size3, number_of_edges3, coefs3)
OutagePolynomial.plot(p3, O_p3, UpperB3, string_expanded_polynomial3)

# THREE OUTAGE MULTIPLICATION
O_p123 = multiply_two_lists(O_p1, O_p2, O_p3)
UpperB123 = [] # Upper bound for this case is nothing for now, to be changed later on
# plt.scatter(p2, O_p123, 'o:r') 
# marker = '^k:'
# OutagePolynomial.plot(p2, O_p123, UpperB2, " Three OUTAGE polynomial combination", marker='k:')

# Graph 0
# pathsets0 = compute_pathsets(G, s_node, t_node)
# compute_and_visualize_pathsets(G, s_node, t_node, pathsets0)
one_edge_probability = 0.7
all_pathsets = compute_all_edge_pathsets(G, s_node, t_node)
# print("Here is the list of pathsets : ")
# print(all_pathsets[0])
# print(all_pathsets[1])
# print(all_pathsets[2])
# print(all_pathsets[3])
edge_probabilities = create_dict_edge_probabilities(all_pathsets[0], one_edge_probability) 
# compute_and_visualize_edge_pathset(G, all_pathsets[0], edge_probabilities, s_node, t_node)
# compute_and_visualize_edge_pathsets_with_highlighted_arrows(G, s_node, t_node, all_pathsets, all_pathsets[0], one_edge_probability)

# # Initialize a list to store pathsets for each graph
# monte_carlo_average_success_over_trials = []
# for one_edge_probability in p :
#     ### MONTE-CARLO Confirmation of results
#     # Graph 0

#     all_pathsets = compute_all_edge_pathsets(G, s_node, t_node)
#     edge_probabilities = create_dict_edge_probabilities(all_pathsets[0], one_edge_probability) 

#     total_trial_number = round(100*1/one_edge_probability)
#     average_log, average_success_over_trials = trial_monte_carlo(total_trial_number, all_pathsets, edge_probabilities, one_edge_probability)
#     monte_carlo_average_success_over_trials.append(average_success_over_trials)

# OutagePolynomial.plot(p, monte_carlo_average_success_over_trials, UpperB2, " Monte-carlo probability", marker='m:')
# print("all_pathsets : ",all_pathsets[0])


### MONTE CARLO CURV CREATION

pathsets = []

for pathset in all_pathsets:
    converted_pathset = []
    for i in range(len(pathset)):
        # if i == 0:
        converted_pathset.append(pathset[i])
        # elif pathset[i][0] != converted_pathset[-1][1]:
        #     converted_pathset.append(pathset[i])
    pathsets.append(converted_pathset)

print(pathsets)
source = 1  # source and destination match node labels
destination = 3
num_packets = 50000

# Vary the failure probability (probability_p) between 0 and 1
failure_probability_values = p #numpy.linspace(0, 1, num=101)  # 101 values from 0 to 1

# Extract edges and create the desired format
edges = G.edges(keys=True)
edge_dict = {(u, v): None for u, v, _ in edges}

# Example printing the converted edge format
for edge in edge_dict:
    print(edge)


average_failure_rates = []
edge_probabilities = {}
for probability_p in failure_probability_values:
    for edge in edge_dict:
        edge_probabilities[edge] = probability_p
    # edge_probabilities = {
    #     (1, 2): probability_p,
    #     (2, 4): probability_p,
    #     (1, 3): probability_p,
    #     (3, 4): probability_p
    # }
    # if probability_p == 0.45 :
        # print("HERE")
    failure_rates = simulate_packet_arrival(source, destination, pathsets, edge_probabilities, num_packets)
    average_failure_rate = failure_rates[-1]#sum(failure_rates) / len(failure_rates)
    average_failure_rates.append(average_failure_rate)
    print(f"Probability: {probability_p}, Simulated Failure Rate: {average_failure_rate}, Polynomial Value: {probability_p**9 -9*probability_p**8 +32*probability_p**7 -56*probability_p**6 +46*probability_p**5 -6*probability_p**4 -16*probability_p**3 +8*probability_p**2 +probability_p**1 }")

# Calculate the values of the polynomial p^4 - 4p^3 + 4p^2 for the same range of failure probabilities
polynomial_values = failure_probability_values**9 -9*failure_probability_values**8 +32*failure_probability_values**7 -56*failure_probability_values**6 +46*failure_probability_values**5 -6*failure_probability_values**4 -16*failure_probability_values**3 +8*failure_probability_values**2 +failure_probability_values**1 #failure_probability_values**4 -4*failure_probability_values**3 +4*failure_probability_values**2#failure_probability_values**9 -9*failure_probability_values**8 +32*failure_probability_values**7 -56*failure_probability_values**6 +46*failure_probability_values**5 -6*failure_probability_values**4 -16*failure_probability_values**3 +8*failure_probability_values**2 +failure_probability_values**1  #failure_probability_values**5 - 4*failure_probability_values**4 + 4*failure_probability_values**3
# polynomial_values = 4*failure_probability_values**2 -2*failure_probability_values**3 -4*failure_probability_values**4 +4*failure_probability_values**5 -failure_probability_values**6

# # Plotting the failure rates and the polynomial
# plt.plot(failure_probability_values, average_failure_rates, label="Packet Arrival failure Rate")
# plt.plot(failure_probability_values, polynomial_values, label="p^9 - 9p^8 + 32p^7 - 56p^6 + 46p^5 - 6p^4 - 16p^3 + 8p^2 + p", linestyle='dashed')
# plt.xlabel("failure Probability (probability_p)")
# plt.ylabel("Values")
# plt.title(f"Packet Arrival failure Rate vs. Polynomial Comparison for {num_packets} packets")
# plt.yscale('log')  # Set the y-axis to a logarithmic scale
# plt.legend()
# plt.grid(True)
# plt.savefig(f"monte-carlo_failure_{num_packets}packets_TESTGRAPH_REAL.png")


OutagePolynomial.plot(p, average_failure_rates, UpperB2, "Packet Arrival failure Rate", marker='k:')

######################################################### DRAWING THE GRAPH ################################################

draw_graph(G, "path.png")
draw_graph(G1, "path1.png")
draw_graph(G2, "path2.png")
draw_graph(G3, "path3.png")





# # draw and save graph
# subax1 = plt.subplot(121)
# # pos = nx.spring_layout(G, scale=2)
# # edge_labels = nx.get_edge_attributes(G,'capacity')
# nx.draw(G, with_labels=True, font_weight='bold')
# nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))
# # subax2 = plt.subplot(122)
# # nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
# plt.savefig("path.png")
# # plt.show() 