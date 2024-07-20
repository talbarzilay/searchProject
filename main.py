from heuristics import BaseHeuristic
from heuristics import BellmanUpdateHeuristic
from BWAS import BWAS
from topspin import TopSpinState

instance_1 = [1, 7, 10, 3, 6, 9, 5, 8, 2, 4, 11]  # easy instance
instance_2 = [1, 5, 11, 2, 6, 3, 9, 4, 10, 7, 8]  # hard instance

n,k,W,B,T=11,4,4,5,1000000

print('11111111111111111111111111111111111111')

start1 = TopSpinState(instance_1, k)
base_heuristic = BaseHeuristic(n, k)
path, expansions = BWAS(start1, W, B, base_heuristic.get_h_values, T)
if path is not None:
    print(expansions)
    print(len(path))
    # for vertex in path:
    #     print(vertex)
else:
    print("unsolvable")

print('2222222222222222222222222222222222222')
    
start2 = TopSpinState(instance_2, k)
#BU_heuristic = BellmanUpdateHeuristic(11, 4)
BU_heuristic=base_heuristic
path, expansions = BWAS(start2, W, B, BU_heuristic.get_h_values, T)
if path is not None:
    print(expansions)
    print(len(path))
    # for vertex in path:
    #     print(vertex)
else:
    print("unsolvable")
