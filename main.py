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
  def plot(cls, x, y, UpperB, string_expanded_polynomial):
    ### PLOT
    # convert y-axis to Logarithmic scale
    plt.yscale("log")

    # plotting the points
    plt.plot(x, y)
    # plt.plot(x,UpperB)
    # legend
    legend.append(string_expanded_polynomial)
    plt.legend(legend)
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
G1.add_node(4, subset='target')
s_node1 = 1 # source node
t_node1 = 4 # target node
G1.add_node(2, subset='others')
G1.add_node(3, subset='others')
# Edges 
G1.add_edges_from([(1, 2, 0, {'capacity': "e1"}), (1, 3, 0, {'capacity': "e2"}), (2, 4, 0, {'capacity': "e3"}), (3, 4, 0, {'capacity': "e4"})])
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
p = numpy.linspace(0.01, 0.99, num=1000)
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
p1 = numpy.linspace(0.01, 0.99, num=1000)
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
p2 = numpy.linspace(0.01, 0.99, num=1000)
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
p3 = numpy.linspace(0.01, 0.99, num=1000)
O_p3, UpperB3 = OutagePolynomial.eval(p3, minimum_cut_size3, number_of_edges3, coefs3)
OutagePolynomial.plot(p3, O_p3, UpperB3, string_expanded_polynomial3)

O_p123 = multiply_two_lists(O_p1, O_p2, O_p3)
UpperB123 = [] # Upper bound for this case is nothing for now, to be changed later on
plt.scatter(p2, O_p123, marker="*") 
OutagePolynomial.plot(p2, O_p123, UpperB2, " Three OUTAGE polynomial multiplication")

##################################################


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