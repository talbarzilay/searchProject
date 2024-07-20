import random
import csv
from datetime import datetime
from heuristics import BootstrappingHeuristic, BaseHeuristic
from BWAS import BWAS
from topspin import TopSpinState


n,k = 11,4
N=5000


def generate():
    #boot_heuristic = BootstrappingHeuristic(n, k)
    #_bootstrappingTraining(boot_heuristic, n, k, 4, 5, 1000) if isTrain else boot_heuristic.load_model()
    #boot_heuristic.load_model()
    base_heuristic = BaseHeuristic(n,k)

    print(datetime.now())
    save_sample(*create_sample(N, base_heuristic))
    print(datetime.now())




# -----------------------------------------------

def create_sample(ammount, heuristic):
    X,Y=[],[]

    for sample in range(ammount):
        if sample % 100 == 0:
            print(f'at sample #{sample} #current examples: {len(Y)}')

        topspin = TopSpinState(random.sample(range(1, n+1), n), k)
        path_to_goal, expentions = BWAS(topspin, 4, 10, heuristic.get_h_values, 20000)

        if path_to_goal is None:
            continue

        pathLen = len(path_to_goal)

        for x,y in zip(path_to_goal, list(range(pathLen))[::-1]):
            x = x.get_state_as_list()

            if x in X:
                index = X.index(x)
                Y[index] = min(Y[index], y) 
            else:
                X.append(x)
                Y.append(y)

    print(f"generated {len(Y)} examples")
    return X,Y
   

def save_sample(X,Y):
    x_file_name = 'sample.csv'
    y_file_name = 'labels.csv'

    with open(x_file_name, 'w', newline='') as file:
        csv.writer(file).writerows(X)

    with open(y_file_name, 'w', newline='') as file:
        csv.writer(file).writerows([[y] for y in Y])


# sample a randon topspin state
# def getRandomState(n, k):
#     return TopSpinState(random.sample(range(1, n+1), n), k)
def getRandomState(n, k, maxDis=None):
    topspin = TopSpinState(list(range(1,n+1)), k)

    if maxDis is None:
        maxDis = n*10

    for _ in range(maxDis):
        neighbors = topspin.get_neighbors()
        topspin = neighbors[random.randint(0, len(neighbors)-1)][0] # move to a random neighbor

    return topspin



# -----------------------------------------------

if __name__ == '__main__':
    generate()

