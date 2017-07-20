import ctypes
import numpy as np

from numpy import genfromtxt
from dmp import DMP, RhythmicDMP
import PI2

# Load cassie library
libcassie = ctypes.CDLL('libcassie.so')
c_double_p = ctypes.POINTER(ctypes.c_double)
libcassie.cassie_init.argtypes = []
libcassie.cassie_init.restype = ctypes.c_void_p
libcassie.cassie_duplicate.argtypes = [ctypes.c_void_p]
libcassie.cassie_duplicate.restype = ctypes.c_void_p
libcassie.cassie_copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libcassie.cassie_copy.restype = None
libcassie.cassie_free.argtypes = [ctypes.c_void_p]
libcassie.cassie_free.restype = None

libcassie.cassie_step1.argtypes = [ctypes.c_void_p, c_double_p]
libcassie.cassie_step1.restype = None
libcassie.cassie_step2.argtypes = [ctypes.c_void_p, c_double_p]
libcassie.cassie_step2.restype = None

libcassie.cassie_vis_init.argtypes = []
libcassie.cassie_vis_init.restype = ctypes.c_void_p
libcassie.cassie_vis_free.argtypes = [ctypes.c_void_p]
libcassie.cassie_vis_free.restype = None
libcassie.cassie_vis_draw.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libcassie.cassie_vis_draw.restype = ctypes.c_bool

libcassie.cassie_time.argtypes = [ctypes.c_void_p]
libcassie.cassie_time.restype = ctypes.c_double
libcassie.cassie_pos.argtypes = [ctypes.c_void_p, c_double_p]
libcassie.cassie_pos.restype = None
libcassie.cassie_vel.argtypes = [ctypes.c_void_p, c_double_p]
libcassie.cassie_vel.restype = None

# Create input/output buffers
q_pos = genfromtxt('qpos.csv', delimiter=',')
q_vel = genfromtxt('qvel.csv', delimiter=',')
q_acc = genfromtxt('qacc.csv', delimiter=',')


# Initialize cassie simulation
c = libcassie.init_from_data(q_pos.ctypes.data_as(c_double_p), \
                             q_vel.ctypes.data_as(c_double_p), \
                             q_acc.ctypes.data_as(c_double_p))

#taskspace of expert trajectory
taskspace_output = genfromtxt('cycle.csv', delimiter=',')

params = {'dt': 0.0005, \
          'dims': 12, \
          'canonicals': [0,0,0,0,0,0,1,1,1,1,1,1,1], \
          'y0':taskspace_output[0,:]}


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

params.update({'dy_mult':dy_mult, \
               'dy_add':dy_add, \
               'ddy_mult':ddy_mult, \
               'ddy_add':ddy_add, \
               'dx_mult':dx_mult, \
               'dx_add':dx_add})

dmp = DMP(params)

#TODO add saved weights, heights, centers from file to params dictionary

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
libcassie.cassie_free(c)    