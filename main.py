import networkx as nx
from itertools import combinations
from itertools import chain, combinations
from networkx.algorithms.connectivity import build_auxiliary_node_connectivity
from networkx.algorithms.connectivity import minimum_st_edge_cut
from networkx.algorithms.connectivity import minimum_edge_cut
from networkx.algorithms.flow import build_residual_network
from networkx.algorithms.flow import minimum_cut
import matplotlib.pyplot as plt


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def find_all_minimum_cuts(graph, source, target, capacity_key):
    # min_cut_value, partition = nx.minimum_cut(graph, source, target, capacity=capacity_key)
    # reachable, non_reachable = partition

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


# CREATE THE GRAPH
G = nx.MultiDiGraph(directed=True)
#   Graph 1 :
# G.add_edges_from([(1, 2, {'capacity': 1.0}), (2, 3, {'capacity': 1.0}), (2, 4, {'capacity': 1.0}), (4, 5, {'capacity': 1.0})])
#   Graph 2 :
G.add_node(1, subset='source')
G.add_node(2, subset='others')
G.add_node(3, subset='target')
G.add_edges_from([(1, 2, 0, {'capacity': "e1"}),(2, 3, 0, {'capacity': "e2"}), (2, 3, 1, {'capacity': "e3"})])
# G.add_edge(1, 2, 2, capacity=2.0)  # Add another edge with a different key
# print(list(G.nodes))
# print(list(G.edges))

# G = nx.icosahedral_graph()
# print(list(G.nodes))
# print(list(G.edges))

### CALL TO FLOW COMPUTING FUNCTIONS
# H = build_auxiliary_node_connectivity(G)
# R = build_residual_network(H, "capacity")
s_node = 1
t_node = 3
# cut_value, partition = nx.minimum_cut(G, s_node, t_node, capacity='capacity')
# reachable, non_reachable = partition
# cutset = set()
# for u, nbrs in ((n, G[n]) for n in reachable):
#     cutset.update((u, v) for v in nbrs if v in non_reachable)

# Reuse the auxiliary digraph and the residual network by passing them
# as parameters

# node_combs = list(combinations(G.nodes, 2))

# for s, t in node_combs:
#     print(minimum_st_node_cut(G, s, t, auxiliary=H, residual=R))



# Find all minimum edge cuts in the graph
# min_edge_cuts = list(nx.minimum_edge_cut(G, s=s_node, t=t_node))
all_min_cuts = find_all_minimum_cuts(G, source=s_node, target=t_node, capacity_key='capacity')


### RESULTS PRINT

print("All possible sets of edges that form the minimum cut:")
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
            break
        # # Check if the edge_combination is not a subset of any existing minimal cut sets in L
        # if flag == 0:
        #     tuple_edge_combination = tuple(edge_combination)  # Convert to tuple
        #     is_subset = any(all(e in minimal_set for e in tuple_edge) for minimal_set in L for tuple_edge in tuple_edge_combination)
        #     if not is_subset:
        #         is_minimal = True
        #         for subset in powerset(tuple_edge_combination):
        #             if subset != () and subset != tuple_edge_combination and set(subset).issubset(tuple_edge_combination):
        #                 is_minimal = False
        #                 break
        #         if is_minimal:
        #             L.append(tuple_edge_combination)
    # a=0
    # If edge_combination is not a cut-set for every member of list_edge_combination, add min_cut_set to L :
    if flag_is_present == 0 :
        L.append(min_cut_set)
# Create collection of minimum cut-sets M : a minimum cut-set is a cut-set that is of minimum size
minimum_size = 100000000000000000 #big value, will be replaced when entering below for loop
for cut_set in L:
    temp_size = len(cut_set)
    if temp_size < minimum_size :
        minimum_size = temp_size
for cut_set in L:
    if len(cut_set) == minimum_size:
        M.append(cut_set)
    
print("Here is K the collection of all s-t separating cut-sets: a cut-set is a subset of edges whose removal from the network disconnects nodes s and node t  ")
for cut_set in K:
    # K.append(min_cut_set)
    print(cut_set)
print("Here is L the collection of all minimal cut-sets in the graph : a minimal cut-set is a subset of edges whose removal from the network disconnects nodes s and node t and from which no subset can be called a cut-set : ")
for minimal_cut_set in L:
    # K.append(min_cut_set)
    print(minimal_cut_set)
print("Here is M the collection of all minimum cut-sets in the graph : a minimum cut-set is a subset of edges whose removal from the network disconnects nodes s and node t and from which no subset can be called a cut-set and is of minimum size : ")
for minimum_cut_set in M:
    # K.append(min_cut_set)
    print(minimum_cut_set)

number_of_nodes = G.number_of_nodes() # noted n in Kschischang's paper
minimum_cut_size = len(M[0]) # noted m in Kschischang's paper

# Finding coefficients for A(x) in reliability polynomial : Am = number of cut-sets of size m, Am+1 = number of cut-sets of size m+1 etc...
# Iterating over collection K
temp_list_of_cut_set_sizes = []
for cut_set in K:
    # get size of cut-set and store it if not already stored
    temp_list_of_cut_set_sizes.append(len(cut_set))
# eliminate duplicates and order list in ascending order
list_of_cut_set_sizes = sorted(set(temp_list_of_cut_set_sizes))

    




# print("Set of edges that, if removed from the graph, will disconnect it : ",(minimum_st_edge_cut(G, s_node, t_node)))
# print("The value of the minimum cut : ", cut_value)
# print("Two sets of nodes that define the minimum cut : ", reachable, " and ",  non_reachable)
# print("The cut set of edges that induce the minimum cut : ", sorted(cutset))
# print("List of all the minimum edge cut sets in the graph : ", min_edge_cuts)
### DRAWINGs


# Draw and save the graph
plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference
 
# pos = nx.multipartite_layout(G, subset_key='subset', align='horizontal')  # Use multipartite_layout for automatic separation of parallel edges
# offset = 0.05  # Adjust this value to control the separation of parallel edges
pos = nx.random_layout(G)

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