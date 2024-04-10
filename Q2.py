import random

# Returns the next state 𝑠' if action 𝑎 is executed in state 𝑠 according to Rule 2.
def NextState(s, a):
    action_state_transitions = dict()

    x, y = s

    action_state_transitions["Move Up"] = (validateCoords(s, x, y+1), validateCoords(s, x+1, y), validateCoords(s, x-1, y))
    action_state_transitions["Move Right"] = (validateCoords(s, x+1, y), validateCoords(s, x, y+1), validateCoords(s, x, y-1))
    action_state_transitions["Move Down"] = (validateCoords(s, x, y-1), validateCoords(s, x+1, y), validateCoords(s, x-1, y))
    action_state_transitions["Move Left"] = (validateCoords(s, x-1, y), validateCoords(s, x, y+1), validateCoords(s, x, y-1))

    return random.choices(action_state_transitions[a], weights=[0.8, 0.1, 0.1], k=1)[0]

# If the action cause agent to bump into walls it stays in the same cell
def validateCoords(s, x, y):
    if (x, y) == (2,2) or (x<1) or (x>4) or (y<1) or (y>3):
        return s
    return (x, y)

def nextStateToLabel(s, next_s):
    x1, y1 = s
    x2, y2 = next_s

    if x2 == x1 + 1:
        return 'right'
    elif x2 == x1 -1:
        return 'left'
    elif y2 == y1 + 1:
        return 'upper'
    elif y2 == y1 - 1:
        return 'lower'
    elif s == next_s:
        return 'itself'
    else:
        raise Exception("Invalid next state")

if __name__ == '__main__':
    test_cases = (
        ((1,1), "Move Right"),
        ((1,1), "Move Up"),
        ((3,2), "Move Down"),
        ((3,2), "Move Left"),
        ((3,3), "Move Left")
    )

    for s,a in test_cases:
        frequency = {'itself':0, 'upper': 0, 'lower':0, 'right':0, 'left':0}
        for i in range(100):
            next_s = NextState(s, a)
            frequency[nextStateToLabel(s, next_s)] += 1
        print((s, a), frequency)