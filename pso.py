import random

# DO NOT import any other modules.
# DO NOT change the prototypes of any of the functions.
# Sample test cases given
# Grading will be based on hidden tests


# Cost function to be optimised
# Takes a list of elements
# Return the total sum of squares of even-indexed elements and inverse squares of odd-indexed elements
def cost_function(X): # 0.25 Marks
    cost=0
    for i in range(0, len(X)):
        if i%2 == 0:
            cost=cost+X[i]**2
        else:
            cost=cost+1/X[i]**2
    return cost

    


# Takes length of vector as input
# Returns 4 values - initial_position, initial_velocity, best_position and best_cost
# Initialises position to a list with random values between [-10, 10] and velocity to a list with random values between [-1, 1]
# best_position is an empty list and best cost is set to -1 initially
def initialise(length): # 0.25 Marks
    best_position=[]
    initial_position=[]
    initial_velocity=[]
    best_cost=-1
    for _ in range(0, length):
        initial_position.append(random.random()*20-10)
        initial_velocity.append(random.random()*2-1)
    #print(initial_position)
    return initial_position, initial_velocity, best_position, best_cost
    


# Evaluates the position vector based on the input func
# On getting a better cost, best_position is updated in-place
# Returns the better cost 
def assess(position, best_position, best_cost, func): # 0.25 Marks
    cost1=func(position)
    cost2=func(best_position)
    if best_cost==-1 and best_position==[]:
        best_cost=cost1
        for i in range(0, len(position)):
            best_position.append(position[i])
    if cost1<cost2:
        best_cost=cost1
        #replacing indivudual elements for in-place update
        for i in range(0, len(best_position)):
            best_position[i]=position[i]
    else:
        best_cost=cost2
    return best_cost


    


# Updates velocity in-place by the given formula for each element:
# vel = w*vel + c1*r1*(best_position-position) + c2*r2*(best_group_position-position)
# where r1 and r2 are random numbers between 0 and 1 (not same for each element of the list)
# No return value
def velocity_update(w, c1, c2, velocity, position, best_position, best_group_position): # 0.5 Marks
    for i in range(0, len(velocity)):
        velocity[i]=w*velocity[i] + c1*random.random()*(best_position[i]-position[i]) + c2*random.random()*(best_group_position[i]-position[i])
    return
    
    


# Input - position, velocity, limits(list of two elements - [min, max])
# Updates position in-place by the given formula for each element:
# pos = pos + vel
# Position element set to limit if it crosses either limit value
# No return value
def position_update(position, velocity, limits): # 0.5 Marks
    for i in range(0, len(position)):
        position[i]=position[i]+velocity[i]
        if position[i]>limits[1]:
            position[i]=limits[1]
        if position[i]<limits[0]:
            position[i]=limits[0]
    return


# swarm is a list of particles each of which is a list containing current_position, current_velocity, best_position and best_cost
# Initialise these using the function written above
# In every iteration for every swarm particle, evaluate the current position using the assess function (use the cost function you have defined) and update the particle's best cost if needed
# Update the best group cost and best group position based on performance of that particle
# Then for every swarm particle, first update its velocity then its position
# Return the best position and cost for the group

def optimise(vector_length, swarm_size, w, c1, c2, limits, max_iterations, initial_best_group_position=[], initial_best_group_cost=-1): # 1.25 Marks
    swarm=[]
    particle=[]
    for _ in range(0, swarm_size):
        particle=list(initialise(vector_length))
        swarm.append(particle)
    for _ in range(0, max_iterations):
        for j in range(0, swarm_size):
            swarm[j][3]=assess(swarm[j][0],swarm[j][2],swarm[j][3],cost_function)
            initial_best_group_cost=assess(swarm[j][0], initial_best_group_position,initial_best_group_cost, cost_function)
        for j in range(0, swarm_size):
            velocity_update(w,c1,c2,swarm[j][1],swarm[j][0],swarm[j][2],initial_best_group_position)
            position_update(swarm[j][0],swarm[j][1],limits)
    #print(initial_best_group_position)
    return initial_best_group_position, initial_best_group_cost

