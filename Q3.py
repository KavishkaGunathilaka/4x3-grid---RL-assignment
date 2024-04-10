from Q2 import NextState
from tqdm import trange

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

# Initialize a dictionary to store the transition counts
transition_counts = {}

# Run the trials
num_trials = 100000 # or some other reasonable number
for trial in trange(num_trials):
    # Start at cell (1,1)
    state = (1, 1)
    
    while True:
        # Get the action from the policy
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
    transition_probs[(state, action)][next_state] = count / sum(transition_counts.get((state, action, (i,j)), 0) for i in range(1, 4) for j in range(1,5))

# Print the estimated transition probabilities
for key in transition_probs:
    for item in transition_probs[key]:
        print(f"P({item})|{key}:\t{transition_probs[key][item]}", )