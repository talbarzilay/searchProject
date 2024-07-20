import random
from datetime import datetime, timedelta
from heuristics import BaseHeuristic
from heuristics import BellmanUpdateHeuristic
from ensambel_models import RFHeuristic, AdaBoostHeuristic, StackingHeuristic
from BWAS import BWAS
from topspin import TopSpinState



def full_analysis():
    n,k=11,4
    data = [getRandomState(n,k) for _ in range(1000)]

    base_heuristic = BaseHeuristic(n, k)
    rf_huristic = RFHeuristic(None)
    ada_huristic = AdaBoostHeuristic(None)
    stacking_huristic = StackingHeuristic(None)

    rf_huristic.load_model()
    ada_huristic.load_model()
    stacking_huristic.load_model()

    print('W', '\t', 'B', '\t', 'heuristic', '\t', '#succs', 'avg. time', '\t', 'avglen', 'avg. exps', '\n')

    for W, B in [ (5,10)]:
        for heuristic, heuristic_name in [(rf_huristic.get_h_values, 'rf'), (ada_huristic.get_h_values, 'ada'), (stacking_huristic.get_h_values, 'stk')]:
            analysis(W, B, heuristic, heuristic_name, data, T=15000)




def analysis(W, B, heuristic, heuristic_name, data, T=1000000):
    paths, times, expentions = get_metrics(W, B, heuristic, data, T)
    succesfull_runs_count = len([path for path in paths if path is not None])
    paths, times, expentions = get_succesfull_runs_metrics(paths, times, expentions)

    print(W, '\t', B, '\t',  heuristic_name, '\t\t', succesfull_runs_count, '\t', (sum(times, timedelta(0))/succesfull_runs_count).total_seconds(), '\t', sum(len(path) for path in paths)/succesfull_runs_count, '\t', sum(expentions)/succesfull_runs_count)






def get_metrics(W, B, heuristic, data, T=1000000):
    times = []
    paths = []
    expentions = []

    for state in data:
        start = datetime.now()
        path, expention_count = BWAS(state, W, B, heuristic, T)
        end = datetime.now()
    
        times.append(end-start)
        paths.append(path)
        expentions.append(expention_count)

    return paths,times,expentions


def get_succesfull_runs_metrics(paths, times, expentions):
    def extract(tuplelist, index):
        return [tup[index] for tup in tuplelist]

    data = [(path, time, expention) for path, time, expention in zip(paths, times, expentions) if path is not None]

    return extract(data, 0),extract(data, 1),extract(data, 2)




# sample a random topspin state by random walking.
def getRandomState(n, k, maxDis=None):
    topspin = TopSpinState(list(range(1,n+1)), k)

    if maxDis is None:
        maxDis = n*100

    for _ in range(maxDis):
        neighbors = topspin.get_neighbors()
        topspin = neighbors[random.randint(0, len(neighbors)-1)][0] # move to a random neighbor

    return topspin





if __name__ == '__main__':
    full_analysis()