import numpy as np
import pprint

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class State():
    """
    Represents a state or a point in the grid.
    coord: coordinate in grid world
    """
    def __init__(self, coord):
        self.coord = coord
        self.action_state_transitions = self._getActionStateTranstions()
        self.is_terminal = self.isTerminal()
        self.reward = self._getReward()

    def __str__(self):
        return f"({self.coord[0]},{self.coord[1]})"

    # Returns if the current state is a terminal state
    def isTerminal(self):
        if (self.coord == (4,3)) or (self.coord == (4,2)):
            return True
        return False
    
    # Returns the reward of current state
    def _getReward(self):
        if self.coord == (4,3):
            return 1
        elif self.coord == (4,2):
            return -1
        return -0.04
    
    # Returns a dictionary mapping each action to the following state it would put the agent in from the currrent state
    def _getActionStateTranstions(self):
        action_state_transitions = dict()

        x, y = self.coord

        action_state_transitions[UP] = (self._validateCoords(x, y+1), self._validateCoords(x+1, y), self._validateCoords(x-1, y))
        action_state_transitions[RIGHT] = (self._validateCoords(x+1, y), self._validateCoords(x, y+1), self._validateCoords(x, y-1))
        action_state_transitions[DOWN] = (self._validateCoords(x, y-1), self._validateCoords(x+1, y), self._validateCoords(x-1, y))
        action_state_transitions[LEFT] = (self._validateCoords(x-1, y), self._validateCoords(x, y+1), self._validateCoords(x, y-1))

        return action_state_transitions

    # If the action cause agent to bump into walls it stays in the same cell
    def _validateCoords(self, x, y):
        if (x, y) == (2,2) or (x<1) or (x>4) or (y<1) or (y>3):
            return self.coord
        return (x, y)

    # Returns the likelihood of ending up in state s_prime after taking action a from the current state - P(s'|s,a)
    def getNextStateLikelihood(self, a, s_prime):
        p = 0
        if s_prime.coord in self.action_state_transitions[a]:
            if self.action_state_transitions[a][0] == s_prime.coord:
                p += 0.8
            if self.action_state_transitions[a][1] == s_prime.coord:
                p += 0.1
            if self.action_state_transitions[a][2] == s_prime.coord:
                p += 0.1
        return p

    # Return the reward for stepping into this state - R(s)
    def getReward(self):
        return self.reward

##############################################################################

class ValueIterationAgent():
    """
    Base implementation of a Dynamic Programming Agent for the Grid World Problem
    env: Gym env the agent will be trained on
    """
    def __init__(self, gamma):
        self.gamma = gamma

        # of states and actions for the grid world problem
        self.num_states = 12
        self.num_actions = 4

    # Prints the values of each state on the grid
    def _printStateValues(self, V):
        grid = np.zeros([3,4])

        for state, value in V.items():
            i, j = self._coordsToIndex(state.coord)

            grid[i, j] = value

        print("Value Function--------------------------")
        pprint.pprint(grid)
        print('\n')

    # Convert list index to world coordinates
    def _indexToCoords(self, i, j):
        x = j + 1
        y = 3 - i
        return (x, y)
    
    # Convert list index to world coordinates
    def _coordsToIndex(self, coord):
        x, y = coord
        j = x - 1
        i = 3 - y
        return (i,j)
    
    # Prints the policy as a grid of arrows
    def _printPolicy(self, pi):
        grid = np.empty([3,4], dtype=object)

        for state, actions in pi.items():
            i, j = self._coordsToIndex(state.coord)
            actions = np.argwhere(actions == np.max(actions)).flatten().tolist()
            grid[i, j] = actions

        # Convert actions to arrows
        arrow_grid = []
        for row_index, row in enumerate(grid):
            arrow_grid_row = []
            for col_index, actions in enumerate(row):
                arrow_char = ''
                if ((row_index,col_index) == self._coordsToIndex((4,3))) or ((row_index,col_index) == self._coordsToIndex((4,2))) or ((row_index,col_index) == self._coordsToIndex((2,2))):
                    arrow_grid_row.append(arrow_char)
                else:
                    # print(actions)
                    for action in actions:
                        if action == 0:
                            arrow_char += '↑'
                        elif action == 1:
                            arrow_char += '→'
                        elif action == 2:
                            arrow_char += '↓'
                        elif action == 3:
                            arrow_char += '←'
                    arrow_grid_row.append(arrow_char)
            arrow_grid.append(arrow_grid_row)

        print("Policy--------------------------")
        pprint.pprint(arrow_grid, width=50)
        print('\n')

    ## Initialize the states (S), state value function (V), and the policy (pi)
    # Value Function--------------------------
    # array([[0., 0., 0., 0.],
    #     [0., 0., 0., 0.],
    #     [0., 0., 0., 0.]])
    #
    # Policy--------------------------
    # [['', '↑→↓←', '↑→↓←', '↑→↓←'],
    # ['↑→↓←', '↑→↓←', '↑→↓←', '↑→↓←'],
    # ['↑→↓←', '↑→↓←', '↑→↓←', '']]
    def initSVAndPi(self):
        self.S = []
        V = {}
        pi = {}
        for y in range(1, 4):
            for x in range(1, 5):
                # Create the state
                s = State((x,y))
                self.S.append(s)
                # Initialize the value of every state to 0
                V[s] = 0
                # Begin with a policy that selects every  action with equal probability
                pi[s] = [0.25] * self.num_actions
        return V, pi

    # Gets the action values for a state by getting the expected return of taking each action
    def getActionValuesForState(self, s, V):
        action_values = []
        # (2,2) is not a valid state
        if s.coord == (2,2):
            return [0]*self.num_states
        for action in range(self.num_actions):
            action_value = 0
            for s_prime in self.S:
                p = s.getNextStateLikelihood(action, s_prime) # P(s'|s,a)
                action_value += p * (s_prime.getReward() + self.gamma * V[s_prime]) # sum over all s': P(s'|s,a))*(R(s') + gamma*V_s')
            action_values.append(action_value)
        return action_values # (sum over all s': P(s'|s,a))*(R(s') + gamma*V_s')) for all actions

    def valueIterate(self):
        V, pi = self.initSVAndPi()

        # threshold for determing the end of the policy eval loop
        theta = 0.000001

        print("Value Iteration 0")

        i = 1
        while True:
            self._printStateValues(V)
            self._printPolicy(pi)
            print("**************************************************************\n")
            
            print("Value Iteration", i)
            # change in state values
            delta = 0

            for s in self.S:
                # current state value
                v = V[s]

                # set the new state value
                V[s] = 0

                if s.isTerminal():
                    continue

                action_values = np.round(self.getActionValuesForState(s, V), 10) # (sum over all s': P(s'|s,a))*(R(s') + gamma*V_s')) for all actions

                V[s] = max(action_values)
                new_best_actions = np.argwhere(action_values == np.max(action_values)).flatten().tolist()

                # Set the likelihood of selecting the new best action in the policy to 1 for all other actions make it 0
                for action in range(self.num_actions):
                    if action not in new_best_actions:
                        pi[s][action] = 0
                    else:
                        pi[s][action] = 1

                delta = max(delta, abs(v - V[s]))

            

            if delta < theta:
                self._printStateValues(V)
                self._printPolicy(pi)
                break

            i += 1

if __name__ == '__main__':
    agent = ValueIterationAgent(0.9)
    agent.valueIterate()
