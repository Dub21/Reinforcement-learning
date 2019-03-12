
import numpy as np
import random

GAMMA = 0.9
ALPHA = 0.1
Reward_matrix = ('U', 'D', 'L', 'R')
discount = 0.95
learning_rate = 0.1
#Reward structure: T = 5, R = 3, P = 1, S = 0

environment_matrix = [[3,0],
                      [5,1]] # Player 1 cooperates, player 2 defects

q_matrix =  [[[0, 0],[0, 0]],[[0, 0],[0, 0]]]

sarsa_matrix = [[[0, 0],[0, 0]],[[0, 0],[0, 0]]]

strategy1=[]
strategy2=[]

def getAction(prev_own_action, prev_opponent_action):

 #   print(np.argmax(q_matrix[cur_pos][prev_own_action][prev_opponent_action]))
    if(q_matrix[prev_own_action][prev_opponent_action][0]==q_matrix[prev_own_action][prev_opponent_action][1]):
        return random.choice([0,1])
    else:    
        return np.argmax(q_matrix[prev_own_action][prev_opponent_action])


def getActionOpponent(prev_own_action, prev_opponent_action):
    
  #  print("opponent"+ str(np.argmax(sarsa_matrix[cur_pos][prev_own_action][prev_opponent_action])))
    if(sarsa_matrix[prev_own_action][prev_opponent_action][0]==sarsa_matrix[prev_own_action][prev_opponent_action][1]):
        return random.choice([0,1])
    else:
        return np.argmax(sarsa_matrix[prev_own_action][prev_opponent_action])
    


discount = 0.95
learning_rate = 0.1
epsilon = 1
max_epsilon=1.0
min_epsilon=0.01
decay_rate = 0.99

prev_own_action= random.choice([0,1])
prev_opponent_action= random.choice([0,1])

for _ in range(1000):
    # get starting place-
        # get all possible next states from cur_step
       # possible_actions = getAllPossibleNextAction[state1, state2]
        # select any one action randomly
        # get starting place
        # find the next state corresponding to the action selected
        strategy1.clear()
        strategy2.clear()

       # opponent_future_action =  opponent_action
        
        if (random.uniform(0,1)<epsilon):
                own_action = random.choice([0,1])
                opponent_action = random.choice([0,1])
                opponent_future_action =  random.choice([0,1])
        else:
                own_action = getAction(prev_own_action, prev_opponent_action)
                opponent_action = getActionOpponent(prev_opponent_action, prev_own_action)
                opponent_future_action =  getActionOpponent(opponent_action, own_action)
            
        # update the q_matrix
            
        q_matrix[prev_own_action][prev_opponent_action][own_action] = q_matrix[prev_own_action][prev_opponent_action][own_action] + learning_rate * (environment_matrix[own_action][opponent_action] + 
        discount * max(q_matrix[prev_own_action][prev_opponent_action]) - 
        q_matrix[prev_own_action][prev_opponent_action][own_action])
           
        # update the sarsa_matrix
            
        sarsa_matrix[prev_opponent_action][prev_own_action][opponent_action] = sarsa_matrix[prev_opponent_action][prev_own_action][opponent_action] + learning_rate * (environment_matrix[opponent_action][own_action] + 
        discount * (sarsa_matrix[prev_opponent_action][prev_own_action][opponent_future_action]) - 
        sarsa_matrix[prev_opponent_action][prev_own_action][opponent_action])
            
        # go to next state
        
        prev_own_action = own_action
        prev_opponent_action = opponent_action
        strategy1.append(own_action)
        strategy2.append(opponent_action)
        
        # print status
        epsilon = epsilon*decay_rate
        print(epsilon)
        print("Episode ", _ , " done")
        print(strategy1)
        print(strategy2)
        
print(q_matrix)
print(sarsa_matrix)
print("Training done...")
print("/")


print("let's play")
initial_own_action= random.choice([0,1])
initial_opp_action= random.choice([0,1])

for _ in range(10):
    own_action = getAction(initial_own_action, initial_opp_action)
    opponent_action = getActionOpponent(initial_opp_action, initial_own_action)
    opponent_future_action =  getActionOpponent(opponent_action, own_action)
    initial_own_action = own_action
    initial_opp_action = opponent_action
    strategy1.append(own_action)
    strategy2.append(opponent_action)

print("q-player")
print(strategy1)
print("sarsa-player")
print(strategy2)

