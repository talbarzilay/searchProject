import unittest
import random
from BWAS import BWAS
from topspin import TopSpinState
from heuristics import BaseHeuristic

class TestBWAS(unittest.TestCase):

    def test_bwas_input_check(self):
        with self.assertRaises(ValueError) as context:
            BWAS(TopSpinState([1,2,3,4,5], 3), 0.5, 32, lambda x: [1,2], 40)

        with self.assertRaises(ValueError) as context:
            BWAS(TopSpinState([1,2,3,4,5], 3), 5, -32, lambda x: [1,2], 40)

        with self.assertRaises(ValueError) as context:
            BWAS(TopSpinState([1,2,3,4,5], 3), 5, 32, lambda x: [1,2], -40)


    # -------------------------------------------------------------

    # test solved case

    def test_bwas_solved(self):
        solved_topspin = TopSpinState([1,2,3,4,5,6,7,8,9,10], 4)
        path, expentions = BWAS(solved_topspin, 5, 10, BaseHeuristic(10, 4).get_h_values , 1)

        self.assertEqual(len(path), 1)
        self.assertTrue(path[-1].is_goal())
        self.assertEqual(path[0].get_state_as_list(), [1,2,3,4,5,6,7,8,9,10])


    # -------------------------------------------------------------

    # test simlpe cases

    # test single shift
    def test_bwas_shift(self):
        topspin = TopSpinState([2,3,1], 2)
        path, expentions = BWAS(topspin, 5, 10, BaseHeuristic(10, 4).get_h_values , 1)

        self.assertEqual(len(path), 2)
        self.assertTrue(path[-1].is_goal())
        self.assertEqual(path[0].get_state_as_list(), topspin.get_state_as_list())

    # test single rotation
    def test_bwas_rotate(self):
        topspin = TopSpinState([2,1,3,4,5], 2)
        path, expentions = BWAS(topspin, 5, 10, BaseHeuristic(10, 4).get_h_values , 1)

        self.assertEqual(len(path), 2)
        self.assertTrue(path[-1].is_goal())
        self.assertEqual(path[0].get_state_as_list(), topspin.get_state_as_list())


    # -------------------------------------------------------------

    # test general cases

    def test_bwas_easyMainExample(self):
        topspin = TopSpinState([1, 7, 10, 3, 6, 9, 5, 8, 2, 4, 11], 4)
        path, expentions = BWAS(topspin, 5, 10, BaseHeuristic(11, 4).get_h_values , 1000000)


        self.assertEqual(path[0].get_state_as_list(), topspin.get_state_as_list())
        self.assertTrue(path[-1].is_goal())

        for i in range(len(path)-1):
            self.assertIn(path[i+1], [neighbor for neighbor,_ in path[i].get_neighbors()])


    def test_bwas_hardMainExample(self):
        topspin = TopSpinState([1, 5, 11, 2, 6, 3, 9, 4, 10, 7, 8], 4)
        path, expentions = BWAS(topspin, 5, 10, BaseHeuristic(11, 4).get_h_values , 1000000)

        self.assertEqual(path[0].get_state_as_list(), topspin.get_state_as_list())
        self.assertTrue(path[-1].is_goal())

        for i in range(len(path)-1):
            self.assertIn(path[i+1], [neighbor for neighbor,_ in path[i].get_neighbors()])


    def test_bwas_random(self):
        n = 7
        k = 3
        state = random.sample(range(1, n+1), n)

        topspin = TopSpinState(state, k)
        path, expentions = BWAS(topspin, 5, 10, BaseHeuristic(11, 4).get_h_values , 1000000)

        if path is None:
            print(f'!!! random test falied to be solved after {expentions} expentions. initial state={state} !!!', end=' ')
            return

        
        print(f'!!! random test was solved after {expentions} expentions with path len {len(path)}. initial state={state} !!!', end=' ')

        self.assertEqual(path[0].get_state_as_list(), state)
        self.assertTrue(path[-1].is_goal())
        

        for i in range(len(path)-1):
            self.assertIn(path[i+1], [neighbor for neighbor,_ in path[i].get_neighbors()])




if __name__ == '__main__':
    unittest.main()