from topspin import TopSpinState
from minPriorityQueue import MinPriorityQueue


def BWAS(start, W, B, heuristic_function, T):

    # ------------------- verifiers ----------------------

    if W < 1:
        raise ValueError('W must be >=1')

    if B < 0:
        raise ValueError("batch size can't be negative!")

    if T < 0 :
        raise ValueError("T can't be negative!")

 
    # ------------------- variabels ----------------------

    open_q = MinPriorityQueue() # a minimum priority queue of all nodes that ere disciverd but not yet expanded.
    closed = {}                 # a dict of nodes thate were already discivered, and their costs from start.
    UB = float('inf')           # cost of best solution so far
    LB = 0                      # mimimal priority of nodes in the current batch
    n_ub = None                 # a node of the solotion state, containing a revese path to the start node
    expentions = 0              # total expentions so far, to prevent algorithen from running to much.      


    # ------------------- init state ----------------------

    open_q.push(Node(start, 0, None), heuristic_function([start])[0])    # push the initial state. the inital state costs 0 (already there), and have no parent.
    closed[start] = 0


    # ------------------- run search algorithem ----------------------

    while (not open_q.is_empty()) and (expentions <= T):
        batch_expentions = 0    # total expentions so far in the current batch.
        generated = []          # list of states generated in this batch

        # expend batch
        while (not open_q.is_empty()) and (expentions <= T) and (batch_expentions <= B):
            expentions+=1
            batch_expentions+=1
            priority, current_node = open_q.pop_min_priority()

            # if current_node in closed and closed[current_node] < current_node.reach_cost: # ignore nodes that already founded with better path
            #     continue

            (state, reach_cost, parent) = current_node

            if len(generated) == 0: # save cost of minimal iteem in batch
                LB = max(priority, LB)

            if state.is_goal():         # check if goal is reached
                if UB > reach_cost:
                    UB = reach_cost     # a less costly pathh to goal was discovered
                    n_ub = current_node # save goal node to recreate path.
                continue                # keep serching the batch.

            for successor, cost2successor in state.get_neighbors(): # go through all the neighbors of current item, and push them to the queue
                successor_cost = reach_cost + cost2successor

                if successor not in closed or successor_cost < closed[successor]: # ethier a path to an new node, or a less costly path to an already discoverd node.
                    closed[successor] = successor_cost
                    generated.append(Node(successor, successor_cost, current_node))

        # generated now contains a list of all noded generated in this batch

        if LB >= UB : # the pest posibale path in this batch was found, simpy returns it
            return n_ub.path_to_goal(), expentions

        heuristics = heuristic_function([node.state for node in generated]) if generated else []

        for node, heuristic in zip(generated, heuristics): # insert all the newly discoverd items to the queue
            open_q.push(node,  node.reach_cost + W*heuristic)


    # ------------------- end ----------------------
    
    return None if n_ub is None else  n_ub.path_to_goal(), expentions  # if no n_ub, then falied to get results (perhaps use a bigger T?).




# -------------------------------------------------------------------------------------------------

# a class that represent a node, 
# with the state of the elemnt, 
# the total cost from initial state,
# and a pointer to it's  parent (to generate the path to get to it from initial state)
class Node:
    def __init__(self, state, reach_cost, parent):
        self.state=state
        self.reach_cost=reach_cost
        self.parent=parent

    def __iter__(self):
        yield self.state
        yield self.reach_cost
        yield self.parent

    def __str__(self):
        return f"{str(self.state)}@{self.reach_cost}"

    def __repr__(self):
        return f"Node({str(self.state)}, {self.reach_cost})"

    # get a list of all states until the goal
    def path_to_goal(self):
        path = []
        current = self
        
        while current is not None:
            path.append(current.state)
            current = current.parent

        return path[::-1] # revese so it will be in the right order