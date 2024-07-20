import unittest
from topspin import TopSpinState

class TestTopSpin(unittest.TestCase):

    def test_constrctor_valid(self):
        try:
            TopSpinState([1,2,3], 1)
            TopSpinState([3,1,2], 2)
            TopSpinState([3,1,2], 3)
            TopSpinState([3,1,4,2], 3)
        except:
            self.fail('constrcto faild in valid inputs')


    def test_constrctor_invalid(self):
        with self.assertRaises(ValueError) as context:
            TopSpinState([1,2,3], -5)

        with self.assertRaises(ValueError) as context:
            TopSpinState([1,2,3], 4)

        with self.assertRaises(ValueError) as context:
            TopSpinState([1,3], 1)

        with self.assertRaises(ValueError) as context:
            TopSpinState([1,2,2], 1)

        with self.assertRaises(ValueError) as context:
            TopSpinState([1,2,2,3], 1)

    # -------------------------------------------------------------

    def test_is_goal(self):
        for n in range(1,6):
            for k in range(1, n+1):
                self.assertTrue(TopSpinState(list(range(1,n+1)), k).is_goal())


    def test_not_is_goal(self):
        self.assertFalse(TopSpinState([2,1], 1).is_goal())
        self.assertFalse(TopSpinState([2,1], 2).is_goal())

        self.assertFalse(TopSpinState([2,1,3], 2).is_goal())

        self.assertFalse(TopSpinState([2,1,5,4,3], 3).is_goal())

    # -------------------------------------------------------------

    def test_get_state_as_list(self):
        state = [1,2,3,4,5]
        self.assertEqual(TopSpinState(state, 1).get_state_as_list(), state)
        self.assertEqual(TopSpinState(state, 3).get_state_as_list(), state)

        state = [3,5,2,1,4]
        self.assertEqual(TopSpinState(state, 1).get_state_as_list(), state)
        self.assertEqual(TopSpinState(state, 3).get_state_as_list(), state)
        self.assertEqual(TopSpinState(state, 4).get_state_as_list(), state)

    
    def test_get_state_as_list_changed(self):
        state = [1,2,3,4,5]
        topspin = TopSpinState(state, 3)
        
        self.assertEqual(topspin.get_state_as_list(), [1,2,3,4,5])
        
        state.pop()
        self.assertEqual(topspin.get_state_as_list(), [1,2,3,4,5])

        state.append(15)
        self.assertEqual(topspin.get_state_as_list(), [1,2,3,4,5])

    # -------------------------------------------------------------

    def test_get_neighbors(self):
        def get_state(item):
            # change to identity function if get_neighbors returns a list insted of TestTopSpinStare items
            return item.get_state_as_list()

        def neighbors(topspin):
            return [get_state(state) for state,cost in topspin.get_neighbors()]
        
        children = neighbors(TopSpinState([1,2,3,4,5], 3))
        self.assertIn([2,3,4,5,1], children)
        self.assertIn([5,1,2,3,4], children)
        self.assertIn([3,2,1,4,5], children)

        children = neighbors(TopSpinState([1,2,3,4,5], 2))
        self.assertIn([2,3,4,5,1], children)
        self.assertIn([5,1,2,3,4], children)
        self.assertIn([2,1,3,4,5], children)

        children = neighbors(TopSpinState([4,1,3,2,5], 3))
        self.assertIn([1,3,2,5,4], children)
        self.assertIn([5,4,1,3,2], children)
        self.assertIn([3,1,4,2,5], children)

        children = neighbors(TopSpinState([4,1,3,2,5], 4))
        self.assertIn([1,3,2,5,4], children)
        self.assertIn([5,4,1,3,2], children)
        self.assertIn([2,3,1,4,5], children)


    # -------------------------------------------------------------

    def test_equlity_same_pointer(self):
        ts = TopSpinState([1,2,3,4,5], 3)
        self.assertEqual(ts, ts)

    def test_equlity(self):
        ts1 = TopSpinState([1,2,3,4,5], 3)
        ts2 = TopSpinState([1,2,3,4,5], 3)
        self.assertEqual(ts1, ts2)

    def test_inequlity(self):
        ts1 = TopSpinState([1,2,3,4,5], 3)
        ts2 = TopSpinState([1,2,3,5,4], 3)
        self.assertNotEqual(ts1, ts2)

        ts2 = TopSpinState([1,2,3,4,5], 4)
        self.assertNotEqual(ts1, ts2)

    

if __name__ == '__main__':
    unittest.main()

