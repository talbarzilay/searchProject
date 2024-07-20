
class TopSpinState:

    def __init__(self, state, k=4):
        
        if k < 0:
            raise ValueError('k must be positive')

        if k > len(state):
            raise ValueError('k must be smaller or equal to n')
        
        if sorted(state) !=list(range(1, len(state) + 1)):
            raise ValueError('state must contains all number fron 1 to n, and exactly once.')

        self.state = state.copy() # copy to avid problem when changing list, both points to same list.
        self.k = k
        self.n = len(self.state)

    def __eq__(self, other):
        if isinstance(other, TopSpinState):
            return self.state == other.state and self.k == other.k
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((tuple(self.state), self.k))

    def __str__(self):
        return str(self.state)

    def __repr__(self):
        return f"TopSpinState({str(self.state)}, k={self.k})"




    def is_goal(self):
        return all([disk == shouldBe+1 for shouldBe, disk in enumerate(self.state)])


    def get_state_as_list(self):
        return self.state.copy() # copy to avid problem when changing list, both points to same list.


    def get_neighbors(self):
        #rotate left
        left = self.state[1:] + self.state[:1]

        #rotate right
        right = self.state[-1:] + self.state[:-1]

        # spin first k
        spink = self.state[:self.k][::-1] + self.state[self.k:]

        return [(TopSpinState(left, self.k), 1), (TopSpinState(right,self.k), 1), (TopSpinState(spink,self.k), 1)]
