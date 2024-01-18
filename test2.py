import numpy as np


def verify_number_already_attributed_to_edge(edge_to_test, random_number, edge_number_association, pathsets) :
    # Verify if edge already has a random number associated with it
    if edge_to_test not in edge_number_association:
        # Associate the random number to it
        edge_number_association[edge_to_test] = random_number
        print("Now edge:", edge_to_test, "has been attributed the random number:", random_number)
    else:
        # Edge already has a random number associated with it
        print("Edge ", edge_to_test, "already has a random number:", edge_number_association[edge_to_test])
    return edge_number_association


### FOR EVERY PACKET :
# edge_number_association = {(1, 2): 0.5, (2, 3): 0.25, (3, 5): 0.75, (4, 5): 0.355}
edge_number_association = {}
pathsets = [[(1, 2), (2, 3), (3, 5)], [(1, 2), (2, 4), (4, 5)]] 
# Take edge from pathset to attribute a random number to it
# edge_to_test = pathsets[1][1]
for pathset in pathsets:
    for edge_to_test in pathset:
        # Roll dice 
        # print("Edge being tested is : ", edge_to_test)
        random_number_list = np.random.uniform(0,1,1)
        random_number = random_number_list[-1]
        # Update_edge_number_association
        edge_number_association = verify_number_already_attributed_to_edge(edge_to_test, random_number, edge_number_association, pathsets)
        # Print the updated edge_number_association
        print("Updated edge_number_association:", edge_number_association)
print(edge_number_association.get((1,2)))
