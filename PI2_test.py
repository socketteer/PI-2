import numpy as np

from numpy import genfromtxt
from pydmp import dmp
import scipy.ndimage
from pi2 import PI2
from rl_cassie import RL_Cassie

# Create input/output buffers
q_pos = genfromtxt('qpos.csv', delimiter=',')
q_vel = genfromtxt('qvel.csv', delimiter=',')
q_acc = genfromtxt('qacc.csv', delimiter=',')

cassie = RL_Cassie(q_pos, q_vel, q_acc)

#taskspace of expert trajectory
taskspace_output = genfromtxt('cycle.csv', delimiter=',').transpose()
y_des = scipy.ndimage.filters.gaussian_filter1d(taskspace_output, 15, axis=1, mode='wrap')

params = {'dt': 0.0005,
          'dims': 12,
          'tau': 1.31492439185,
          'canonicals': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
          'y0': y_des[:, 0],
          'num_basis' : 100}

def dy_mult():
    return 1

def dy_add():
    return 0

def ddy_mult():
    return 1

def ddy_add():
    return 0

def dx_mult():
    return 1

def dx_add():
    return 0

def cost(cassie):
    '''Computes cost-to-go using state information. Returns float.'''
    pos = cassie.pos()
    vel = cassie.vel()
    
    #quadratic distance between center of mass and center of pressure
    #[24] = right foot x pos, [25] = right foot y pos, [55]= left x, [56] = left y
    stability_cost_x = (cassie.state[24] + cassie.state[55])**2
    stability_cost_y = (cassie.state[25] + cassie.state[56])**2
    #height deviation penalty with forgiveness region
    height_cost = 0
    
    forgive_range = 0.3 #TODO no clue what range this should be in
    
    if(abs(pos[2] - 1) > forgive_range):
        height_cost = abs(pos[2] - 1)
    
    #TODO linear velocity deviation penalty with forgiveness region
    
    cost = stability_cost_x + stability_cost_y + height_cost
    
    return cost

def terminal_cost(cassie):
    pos_error, vel_error, acc_error = cassie.diff()
    return np.sum(np.square(pos_error)) + np.sum(np.square(vel_error)) + np.sum(np.square(acc_error))

params.update({'dy_mult':dy_mult, \
               'dy_add':dy_add, \
               'ddy_mult':ddy_mult, \
               'ddy_add':ddy_add, \
               'dx_mult':dx_mult, \
               'dx_add':dx_add})

#imitation of path run
dmp = dmp.RhythmicDMP(**params)
params.update(dmp.imitate_path(y_des=y_des, **params))

#PI2 parameters
num_bestpaths = 3
timesteps = 30
num_traj = 10

#Creating cost buffer
#TODO this is stupid, find better way
cost = np.zeros(timesteps)

pi = PI2(c, dmp, params, timesteps, num_traj, num_bestpaths)

for _ in range(10):
    #update weights of DMP
    params['weights'] = pi.update_params(params['weights'])
    #getting cumulative cost of rollout with new weights to evaluate convergence behavior
    print(pi.get_cumulative_cost(params['weights'], test = 1))

#Clean up
cassie.clean_up()