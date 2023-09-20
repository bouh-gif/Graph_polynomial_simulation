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
  def plot(cls, x, y, UpperB, string_expanded_polynomial):
    ### PLOT
    # convert y-axis to Logarithmic scale
    plt.yscale("log")
    # plotting the points
    plt.plot(x, y)
    # plt.plot(x,UpperB)
    # legend
    plt.legend([string_expanded_polynomial])
    # naming the x axis
    plt.xlabel('p')
    # naming the y axis
    plt.ylabel('O(p)')

    # giving a title to my graph
    plt.title('Reliability polynomial O(p)')

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
def display_collection(graph_name, K,L,M) :     
    print("FOR GRAPH : ", graph_name)   
    print("Here is K the collection of all s-t separating cut-sets: a cut-set is a subset of edges whose removal from the network disconnects nodes s and node t  ")
    for cut_set in K:
        # K.append(cut_set)
        print(cut_set)
    print("Here is L the collection of all minimal cut-sets in the graph : a minimal cut-set is a subset of edges whose removal from the network disconnects nodes s and node t and from which no subset can be called a cut-set : ")
    for minimal_cut_set in L:
        # L.append(minimal_cut_set)
        print(minimal_cut_set)
    print("Here is M the collection of all minimum cut-sets in the graph : a minimum cut-set is a subset of edges whose removal from the network disconnects nodes s and node t and from which no subset can be called a cut-set and is of minimum size : ")
    for minimum_cut_set in M:
        # M.append(minimum_cut_set)
        print(minimum_cut_set)

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
def multiply_two_lists(list1, list2) :
    # Multiplying two lists
    # of same length
    res_list = []
    for i in range(0, len(list1)):
        res_list.append(list1[i] * list2[i])
    return res_list
    
        
# CREATE THE GRAPH
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
# G.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (1, 2, 1, {'capacity': "e2"}), (2, 3, 0, {'capacity': "e3"}), (2, 3, 1, {'capacity': "e4"})])
# G.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (2, 3, 0, {'capacity': "e2"}), (2, 3, 1, {'capacity': "e3"})])
# G.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (2, 3, 0, {'capacity': "e2"})])
# G.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (2, 3, 0, {'capacity': "e5"}), (1, 4, 0, {'capacity': "e2"}), (4, 5, 0, {'capacity': "e3"}), (2, 5, 0, {'capacity': "e4"}), (5, 3, 0, {'capacity': "e6"})])
# G.add_edges_from([(1, 2, 0, {'capacity': "a"}), (2, 3, 0, {'capacity': "b"}), (1, 3, 0, {'capacity': "c"}), (1, 4, 0, {'capacity': "d"}), (4, 3, 0, {'capacity': "f"})])
# G.add_edges_from([(2, 1, 0, {'capacity': "a"}), (1, 3, 0, {'capacity': "b"}), (2, 4, 0, {'capacity': "c"}), (4, 3, 0, {'capacity': "d"}), (1, 4, 0, {'capacity': "f"})])
G.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (1, 3, 0, {'capacity': "e2"}), (2, 4, 0, {'capacity': "e3"}), (3, 4, 0, {'capacity': "e4"})])
G.add_edges_from([(4, 5, 0, {'capacity': "e5"}), (5, 6, 0, {'capacity': "e6"}), (5, 7, 0, {'capacity': "e7"}), (6, 8, 0, {'capacity': "e8"}), (7, 8, 0, {'capacity': "e9"})])


G1 = nx.MultiDiGraph(directed=True)
# Nodes 
G1.add_node(1, subset='source')
G1.add_node(4, subset='target')
s_node1 = 1 # source node
t_node1 = 4 # target node
G1.add_node(2, subset='others')
G1.add_node(3, subset='others')
# Edges 
# G.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (1, 2, 1, {'capacity': "e2"}), (2, 3, 0, {'capacity': "e3"}), (2, 3, 1, {'capacity': "e4"})])
# G.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (2, 3, 0, {'capacity': "e2"}), (2, 3, 1, {'capacity': "e3"})])
# G.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (2, 3, 0, {'capacity': "e2"})])
# G.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (2, 3, 0, {'capacity': "e5"}), (1, 4, 0, {'capacity': "e2"}), (4, 5, 0, {'capacity': "e3"}), (2, 5, 0, {'capacity': "e4"}), (5, 3, 0, {'capacity': "e6"})])
# G.add_edges_from([(1, 2, 0, {'capacity': "a"}), (2, 3, 0, {'capacity': "b"}), (1, 3, 0, {'capacity': "c"}), (1, 4, 0, {'capacity': "d"}), (4, 3, 0, {'capacity': "f"})])
# G.add_edges_from([(2, 1, 0, {'capacity': "a"}), (1, 3, 0, {'capacity': "b"}), (2, 4, 0, {'capacity': "c"}), (4, 3, 0, {'capacity': "d"}), (1, 4, 0, {'capacity': "f"})])
G1.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (1, 3, 0, {'capacity': "e2"}), (2, 4, 0, {'capacity': "e3"}), (3, 4, 0, {'capacity': "e4"})])
# G1.add_edges_from([(4, 5, 0, {'capacity': "e5"}), (5, 6, 0, {'capacity': "e6"}), (5, 7, 0, {'capacity': "e7"}), (6, 8, 0, {'capacity': "e8"}), (7, 8, 0, {'capacity': "e9"})])


G2 = nx.MultiDiGraph(directed=True)
# Nodes 
G2.add_node(4, subset='source')
G2.add_node(8, subset='target')
s_node2 = 4 # source node
t_node2 = 8 # target node
G2.add_node(5, subset='others')
G2.add_node(6, subset='others')
G2.add_node(7, subset='others')
# Edges 
# G.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (1, 2, 1, {'capacity': "e2"}), (2, 3, 0, {'capacity': "e3"}), (2, 3, 1, {'capacity': "e4"})])
# G.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (2, 3, 0, {'capacity': "e2"}), (2, 3, 1, {'capacity': "e3"})])
# G.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (2, 3, 0, {'capacity': "e2"})])
# G.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (2, 3, 0, {'capacity': "e5"}), (1, 4, 0, {'capacity': "e2"}), (4, 5, 0, {'capacity': "e3"}), (2, 5, 0, {'capacity': "e4"}), (5, 3, 0, {'capacity': "e6"})])
# G.add_edges_from([(1, 2, 0, {'capacity': "a"}), (2, 3, 0, {'capacity': "b"}), (1, 3, 0, {'capacity': "c"}), (1, 4, 0, {'capacity': "d"}), (4, 3, 0, {'capacity': "f"})])
# G.add_edges_from([(2, 1, 0, {'capacity': "a"}), (1, 3, 0, {'capacity': "b"}), (2, 4, 0, {'capacity': "c"}), (4, 3, 0, {'capacity': "d"}), (1, 4, 0, {'capacity': "f"})])
# G2.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (1, 3, 0, {'capacity': "e2"}), (2, 4, 0, {'capacity': "e3"}), (3, 4, 0, {'capacity': "e4"})])
G2.add_edges_from([(4, 5, 0, {'capacity': "e5"}), (5, 6, 0, {'capacity': "e6"}), (5, 7, 0, {'capacity': "e7"}), (6, 8, 0, {'capacity': "e8"}), (7, 8, 0, {'capacity': "e9"})])

#### GRAPH 0
### CALL TO CUSTOM FLOW COMPUTING FUNCTION
# Find all minimum edge cuts in the graph
all_min_cuts = find_all_minimum_cuts(G, source=s_node, target=t_node, capacity_key='capacity')

### EVALUATING K, L, AND M (collection of cut-sets, minimal cut-sets and minimum cut-sets)
K, L, M = get_collection(all_min_cuts)
display_collection("G", K,L,M)
number_of_edges, minimum_cut_size = get_n_m_values(G, M)
coefs = find_coefs(K)
string_expanded_polynomial = get_polynomial_expression(coefs, minimum_cut_size, number_of_edges)
p = numpy.linspace(0.01, 0.99, num=1000)
O_p, UpperB = OutagePolynomial.eval(p, minimum_cut_size, number_of_edges, coefs)
OutagePolynomial.plot(p, O_p, UpperB, string_expanded_polynomial)

#### GRAPH 1
# Find all minimum edge cuts in the graph
all_min_cuts1 = find_all_minimum_cuts(G1, source=s_node1, target=t_node1, capacity_key='capacity')

### EVALUATING K, L, AND M (collection of cut-sets, minimal cut-sets and minimum cut-sets)
K1, L1, M1 = get_collection(all_min_cuts1)
display_collection("G1", K1, L1, M1)
number_of_edges1, minimum_cut_size1 = get_n_m_values(G1, M1)
coefs1 = find_coefs(K1)
string_expanded_polynomial1 = get_polynomial_expression(coefs1, minimum_cut_size1, number_of_edges1)
p1 = numpy.linspace(0.01, 0.99, num=1000)
O_p1, UpperB1 = OutagePolynomial.eval(p1, minimum_cut_size1, number_of_edges1, coefs1)
OutagePolynomial.plot(p1, O_p1, UpperB1, string_expanded_polynomial1)


#### GRAPH 2
# Find all minimum edge cuts in the graph
all_min_cuts2 = find_all_minimum_cuts(G2, source=s_node2, target=t_node2, capacity_key='capacity')

### EVALUATING K, L, AND M (collection of cut-sets, minimal cut-sets and minimum cut-sets)
K2, L2, M2 = get_collection(all_min_cuts2)
display_collection("G2", K2, L2, M2)
number_of_edges2, minimum_cut_size2 = get_n_m_values(G2, M2)
coefs2 = find_coefs(K2)
string_expanded_polynomial2 = get_polynomial_expression(coefs2, minimum_cut_size2, number_of_edges2)
p2 = numpy.linspace(0.01, 0.99, num=1000)
O_p2, UpperB2 = OutagePolynomial.eval(p2, minimum_cut_size2, number_of_edges2, coefs2)
OutagePolynomial.plot(p2, O_p2, UpperB2, string_expanded_polynomial2)


##################################################


######################################################### DRAWING THE GRAPH ################################################

def draw_graph(G) :
    # Draw and save the graph
    plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference
    
    # pos = nx.multipartite_layout(G, subset_key='subset', align='horizontal')  # Use multipartite_layout for automatic separation of parallel edges
    # offset = 0.05  # Adjust this value to control the separation of parallel edges
    pos = nx.planar_layout(G)

    # # Get the edge labels
    # edge_labels = {(u, v, key): G[u][v][key]['capacity'] for u, v, key in G.edges(keys=True)}

    # Draw edge labels

    # for u, v, key in G.edges(keys=True):
    #     x1, y1 = pos[u]
    #     x2, y2 = pos[v]
    #     capacity = G[u][v][key]['capacity']
    #     plt.text((x1 + x2) / 2, (y1 + y2) / 2, f"{capacity}", horizontalalignment='center', verticalalignment='center', fontsize=10)

    #     x_arr, y_arr = x2, y2
    #     plt.arrow(x1, y1, x_arr - x1, y_arr - y1, color='black', width=0.005, head_width=0.03, head_length=0.03)  # Adjust the width parameter

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

        
    # # Draw nodes and labels
    # nx.draw(G, pos, with_labels=True, font_weight='bold')

    # Draw node labels
    node_labels = {n: n for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels)

    # Turn off axis
    plt.axis('off')

    # Save the file and show the figure 
    plt.savefig("path.png")
    # plt.show()
draw_graph(G)





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