from Q2 import NextState
from tqdm import trange
import random

def ADP(num_trials, pi=None, actions=None):
    assert ((pi != None) and (actions == None)) or ((pi == None) and (actions != None))
    # Initialize a dictionary to store the transition counts
    transition_counts = {}

    # Run the trials
    for trial in trange(num_trials):
        # Start at cell (1,1)
        state = (1, 1)
        
        while True:
            # Get the action from the policy
            if pi == None:
                action = random.choice(actions)
            else:
                action = pi[state]
            
            # Get the next state using the NextState function
            next_state = NextState(state, action)
            
            # Update the transition counts
            if (state, action, next_state) not in transition_counts:
                transition_counts[(state, action, next_state)] = 0
            transition_counts[(state, action, next_state)] += 1
            
            # Update the state
            state = next_state
            
            # Check if we've reached the goal state
            if state == (4, 3) or state == (4, 2):
                break

    # Compute the transition probabilities
    transition_probs = {}
    for (state, action, next_state), count in transition_counts.items():
        if (state, action) not in transition_probs:
            transition_probs[(state, action)] = {}
        
        # P(s'|a,s) = No. of times s' has been reached when action 'a' was executed in state 's' / No. of times state 's' was reached and action 'a' was executed
        transition_probs[(state, action)][next_state] = count / sum(transition_counts.get((state, action, (x,y)), 0) for x in range(1, 5) for y in range(1,4))
    
    return transition_probs

def printProbs(transition_probs):
    count = 0
    # Print the estimated transition probabilities
    for key in transition_probs:
        for item in transition_probs[key]:
            print(f"P({item})|{key}:\t{transition_probs[key][item]}")
            count += 1
            # print(f"({key[0]},'{key[1]}',{item}):{transition_probs[key][item]}")
    print(count, "number of probabilities printed")

if __name__ == '__main__':
    # Given policy
    pi={
        (1,1):'Move Up',
        (1,2):'Move Up',
        (1,3):'Move Right',
        (2,1):'Move Left',
        (2,3):'Move Right',
        (3,1):'Move Left',
        (3,2):'Move Up',
        (3,3):'Move Right',
        (4,1):'Move Left'
    }
    transition_probs = ADP(100000, pi=pi)
    printProbs(transition_probs)