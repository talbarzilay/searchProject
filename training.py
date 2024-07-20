import random
from heuristics import BaseHeuristic
from heuristics import BellmanUpdateHeuristic
from BWAS import BWAS
from topspin import TopSpinState


def bellmanUpdateTraining(bellman_update_heuristic):
    _bellmanUpdateTraining(bellman_update_heuristic, 11, 4, 100000)
    
def bootstrappingTraining(bootstrapping_heuristic):
    _bootstrappingTraining(bootstrapping_heuristic, 11, 4, 4, 10, 2000)
    


# ------------------------------------------------------


def _bootstrappingTraining(bootstrapping_heuristic, n, k, W, B, expentionEachRound):

    goal = TopSpinState(list(range(1,n+1)), k)
    
    #train on goal state 
    print('training on goal...')
    bootstrapping_heuristic.train_model([goal], [0], epochs=10)

    #train by expending state an train on new data
    print('training by stepping from goal...')
    _bootstrap([goal], bootstrapping_heuristic, W, B, expentionEachRound, 50, lambda step: 1 if step <=5 else 0.95 if step<=8 else 0.8 if step<=10 else 0.65 if step<= 13 else 0.5 if step<=15  else 0.4 if step<= 18 else 0.35 if step<=20  else 0.3)

    #train by random examples
    print('training by random states...')
    _bootstrap([], bootstrapping_heuristic, W, B, expentionEachRound, 30, lambda _: 0.333, expenstioFunction=lambda _: randomStates(n,k))

    # save the trained model
    bootstrapping_heuristic.save_model()

    


def _bellmanUpdateTraining(bellman_update_heuristic, n, k, expentionEachRound):

    goal = TopSpinState(list(range(1,n+1)), k)
    
    #train on goal state 
    print('training on goal...')
    bellman_update_heuristic.train_model([goal], [0], epochs=10)

    #train by expending state an train on new data
    print('training by stepping from goal...')
    _bellman([goal], bellman_update_heuristic, expentionEachRound, 60)

    #train by random examples
    #print('training by random states...')
    #_bellman([], bellman_update_heuristic, expentionEachRound, 30, expenstioFunction=lambda _: randomStates(n,k))

    # save the trained model
    bellman_update_heuristic.save_model()


# ------------------------------------------------------

def _bootstrap(prev_states, heuristic, W, B, expentionEachRound, numOfRounds, successRatio, expenstioFunction=None, T=128):
    if expenstioFunction is None:
        expenstioFunction = traverseStates
    
    for roundNum in range(numOfRounds):
        print(f'starting iteration {roundNum+1}/{numOfRounds}.\t', end='', flush=True)

        train, labels = [], []
        cur_states = list(set(firstN(expenstioFunction(prev_states), expentionEachRound)))
        
        solved = []

        # try to solve untill enough solved examples were found, if not - double T and try again.
        print(f'solving {len(cur_states)} insances.\t', end='', flush=True)
        to_solve = cur_states
        cur_T = T
        solved = []

        sr = 0
        while sr < successRatio(roundNum+1):
            bwas = [BWAS(state, W, B, heuristic.get_h_values, cur_T) for state in to_solve]
            solved.extend([path for path,_ in bwas if path is not None])
            sr = len(solved)/len(cur_states)

            if sr < successRatio(roundNum+1):
                print(f'T={cur_T} - SR={round(len(solved)/len(cur_states), 2)}. ', end='', flush=True)
                to_solve = [state for (path,_), state in zip(bwas, to_solve) if path is None] # continue with failed states only
                cur_T *=2

        print(f'success solving ratio: {round(len(solved)/len(cur_states), 2)}.\t', end='', flush=True)

        # use solved cases for training
        for path in solved:
            for dis, state in enumerate(path[:-1][::-1], start=1):
                if state not in train:
                    train.append(state)
                    labels.append(dis)

        #train on data
        if len(labels) > 0:
            print(f'training on {len(labels)} entries.\t\t', end='', flush=True)
            heuristic.train_model(train, labels)
        else:
            print('no train data.\t', end='', flush=True)
        
        # setup next loop
        prev_states = cur_states
        heuristic.save_model()
        print('done iteration.')


def _bellman(prev_states, heuristic, expentionEachRound, numOfRounds, expenstioFunction=None):
    if expenstioFunction is None:
        expenstioFunction = traverseStates
    
    for roundNum in range(numOfRounds):
        print(f'starting iteration {roundNum+1}/{numOfRounds}.\t', end='', flush=True)

        train, labels = [], []
        cur_states = list(set(firstN(expenstioFunction(prev_states), expentionEachRound)))

        # use solved cases for training
        for state in cur_states:
            neighbors = state.get_neighbors()
            heuristics = heuristic.get_h_values([successor for successor, _ in neighbors])

            #if any is goal, use 0 as the heuristics
            if any([successor.is_goal() for successor, _ in neighbors]):
                heuristics = [0]

            train.append(state)
            labels.append(min([cost + heuristic for (successor, cost), heuristic in zip(neighbors, heuristics)]))


        #train on data
        if len(labels) > 0:
            print(f'training on {len(labels)} entries.\t\t', end='', flush=True)
            heuristic.train_model(train, labels)
        else:
            print('no train data.\t', end='', flush=True)
        
        # setup next loop
        prev_states = cur_states
        heuristic.save_model()
        print('done iteration.')



# ------------------------------------------------------

# get a list of states, travese trough it
# will not yeild seen states
def traverseStates(states, seen=[]):
    states = states.copy()

    while len(states) > 0: #there are more travese to yield
        current = states.pop(random.randint(0, len(states) - 1)) # select a random state to expend
    
        for neighbor, cost in current.get_neighbors():
            if neighbor not in seen:
                yield neighbor


#genertor for random states
def randomStates(n,k):
    while True: 
        yield getRandomState(n,k)


# get a list of first n elements from a generator
def firstN(gen, n):
    if isinstance(gen, list):
        if len(gen) <= n:
            return gen
        return gen[:n]

    ans = []
    for i in range(n):
        try:
            ans.append(next(gen))
        except StopIteration:
            return ans
    return ans


#transform a list to generetor
def list_to_generator(lst):
    for item in lst:
        yield item


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



