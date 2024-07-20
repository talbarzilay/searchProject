import heapq

class MinPriorityQueue:
    def __init__(self):
        self.q = []
        self.index = 0

    def __str__(self):
        return str(self.q)

    def __repr__(self):
        return repr(self.q)

    # return true if the que is empty, false otherwise
    def is_empty(self):
        return len(self.q) == 0

    # push a new item with the given priority
    def push(self, item, priority, tiebreaker=float('inf')):
        heapq.heappush(self.q, (priority, tiebreaker, self.index, item)) # index to help with equal priorities
        self.index+=1

    # get the minimal priority item
    # returns a pair: priority, item
    def pop_min_priority(self):
        priority,tiebreaker,index,item = heapq.heappop(self.q)
        return priority,item