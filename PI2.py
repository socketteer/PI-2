import ctypes
import numpy as np

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
 
#TODO Pi-2 CMA

class PI2():
    '''
    Policy Improvement with Path Integrals (Pi^2) with elitism
    Model-free RL algorithm to learn motion primitive weights for bipedal locomotion
    '''
    
    def __init__(self, cassie, dmp, params, \
                 timesteps = 1000, \
                 num_traj = 10, \
                 num_best = 3, 
                 variance = 1, \
                 sensitivity = 1, \
                 control_ratio = 0.7):
        '''
        cassie: cassie_t object. Initial Cassie state.
        pydmp: DMP object
        params: dictionary of DMP parameters
        timesteps: int, number of timesteps for rollouts.
        num_traj: int, number of noisy trajectories to create rollouts for. 
        num_best: int. Number of best paths to remember for elitism
        variance: float array(dims) or float. variance of noise. If float, 
            all dimensions receive noise with the same variance
        sensitivity: float. determines probability's sensitivity to cost
        control_ratio: float array(basis_dims) or float. If float, all dimensions have 
            the same control cost

        '''
        self.dmp = dmp
        self.params = params
        self.dims = self.params['dims']
        self.basis_dims = self.params['basis_dims'] 
        self.timesteps = timesteps
        self.num_traj = num_traj
        self.num_best = num_best
        self.sensitivity= sensitivity 
        self.bestpaths = np.zeros((num_best, self.dims, self.basis_dims))
        
        #If variance is a constant, self.variance becomes an array filled with that constant
        #If variance is an array, self.variance remains that array
        #If control_ratio is a constant, R is a diagonal matrix filled with that constant
        #If control_ratio is an array, R is a diagonal matrix with control_ratio as the diagonal
        ones_array = np.ones(self.basis_dims)
        self.variance = variance * ones_array
        self.R = np.diag(control_ratio * ones_array)
        
        self.cassie_reset = cassie
        self.cassie = cassie
    
    def reset_cassie(self):
        '''resets cassie to state from __init__'''
        libcassie.cassie_copy(self.cassie, self.cassie_reset)
    

    def update_params(self, base_trajectory):
        '''        
        base_trajectory: float array((dims, basis dims))
            
        PI^2 Algorithm: 
            Creates num_traj rollouts from same start state with stochastic DMP parameters.
            Best trajectories from last iteration are included. 
            Datermines cost of each trial, probability of each trial, averages over trials,
            averages over timesteps, and then updates parameter with weighted average. 

        Returns array((dims, basis_dims)) of updated DMP weights
        '''    
        
        #creating buffers    
        prob = np.zeros((self.num_traj, self.timesteps))
        cost = np.zeros((self.num_traj, self.timesteps))
        timestep_param_update = np.zeros((self.timesteps, self.dims, self.basis_dims))
        param_update = np.zeros((self.dims, self.basis_dims))
        M = np.zeros((self.num_traj, self.timesteps, self.dims, self.basis_dims, self.basis_dims)) 
        
        #generating noise array
        noise = self.generate_noise()
        
        #if bestpaths not zeros, using num_best paths from previous iteration
        if np.any(self.bestpaths):
            noise[:self.num_best, :, :] = self.bestpaths 

        #creating rollouts and getting probabilities for K noisy trajectories
        for k in range (self.num_traj):
            #compute M and cost-to-go for each trajectory
            M[k,:,:,:,:], cost[k,:] = self.rollout(base_trajectory)
            #appending control cost
            for i in range(1, self.timesteps):
                for d in range (self.dims):
                    #TODO should control cost of each dimension be summed or averaged?
                    cost[k,:i] += np.transpose(base_trajectory + \
                        M[k,i,d,:,:]*noise[k,d,:]) * self.R * \
                        (base_trajectory + M[k,i,d,:,:]*noise[k,d,:])
            prob_sum = 0
            for i in range(self.timesteps):        
                prob[k,i] = self.probability(cost[k,i])
                prob_sum += prob[k,i]                
            prob[k,:] /= prob_sum  
        
        #averaging over trials, getting param update for each timestep    
        for i in range (self.timesteps):
            timestep_param_update[i,:,:] = self.avg_over_trajectories(prob[:,i], noise, M[:,i,:,:,:])
        
        #averaging over timesteps, getting final param update    
        param_update = self.avg_over_timesteps(timestep_param_update) 
       
        #storing num_best of lowest cumulative cost paths
        self.n_best_paths(cost, 0)   
    
        return base_trajectory + param_update 
    
    def generate_noise(self):
        '''returns float array((num_traj, dims, basis_dims))'''

        noise = np.zeros((self.num_traj, self.dims, self.basis_dims))
        for i in range(self.dims):
            noise[:,i,:] = np.random.normal(0,np.sqrt(self.variance[i]),(self.num_traj, self.basis_dims)) 
        return noise 
    
    def compute_M(self, x):
        '''
        x: float. temporal variable of pydmp
        
        M is needed to project exploration noise onto parameter space
        
        Returns float array((basis_dims, basis_dims))
        '''
        #R is a diagonal matrix
        R_inverse = .1/self.R
        #dmp.get_psi returns a float array(basis dims) of activations at x       
        return R_inverse*self.dmp.get_psi(x)*np.transpose(self.dmp.get_psi(x)) \
                / (np.transpose(self.dmp.get_psi(x))*R_inverse*self.dmp.get_psi(x))
        
    def rollout(self, params, test_rollout = 0):
        '''
        params: float array(dims). initial weights. 
        
        Resets mujoco and pydmp state. Creates a rollout of specified trajectory and determines
        cost-to-go at each timestep. Cost here is a one-dimensional slice of the 2 dimensional 
        array defined in update_params. 
        
        This method is called for each trajectory.
        Returns cost (float array(timesteps)) and M (float array((timesteps, dims, basis_dims, basis_dims)'''
        
        #create buffers
        init_pos = np.zeros(35)
        init_vel = np.zeros(32)
        init_acc = np.zeros(32)
        final_pos = np.zeros(35)
        final_vel = np.zeros(32)
        final_acc = np.zeros(32)
        cassie_input = np.zeros(98, dtype=np.double)
        cassie_output = np.zeros(20, dtype=np.double)
        cost = np.zeros(self.timesteps)
        M = np.zeros((self.timesteps, self.dims, self.basis_dims, self.basis_dims))
        
        #resetting cassie and copying initial state
        self.reset_cassie()
        libcassie.copy_data(self.cassie, init_pos, init_vel, init_acc)
        
        #resetting pydmp and setting weights in params dict
        self.dmp.reset()
        self.params['weights'] = params
 
        x = 0
        for i in range(self.timesteps):
            cassie_output[8:], _, _, x = self.dmp.step(self.params) 
            libcassie.cassie_step2(self.cassie, cassie_output.ctypes.data_as(c_double_p))
            libcassie.cassie_step1(self.cassie, cassie_input.ctypes.data_as(c_double_p))
            #computing M for all dimensions corresponding x at timestep i 
            for d in range(self.dims):
                M[i,d,:,:] = self.compute_M(x[d])                  
            #computing cost to go
            #appends instantaneous cost to current and previous timesteps
            cost[:i+1] += self.cost(cassie_input)                  

        #copying final state
        libcassie.copy_data(self.cassie, final_pos, final_vel, final_acc)

        #appending terminal cost to each instantaneous cost
        cost += self.terminal_cost(init_pos, init_vel, init_acc, final_pos, final_vel, final_acc)
    
        if(test_rollout == 1):
            return cost
        else:
            return cost, M
    
    def get_cumulative_cost(self, params, cost):
        '''
        params: array(dims) of floats. initial weights. 
        cost: array(timesteps) of floats
        
        returns scalar cumulative cost for one rollout'''
        
        self.rollout(params, cost)
        return np.sum(cost, axis = 0)
    
    def n_best_paths(self, cost, rank, noise):
        '''
        cost: float array((num_traj, timesteps))
        rank: int. rank 0 is best path, rank 1 is second best, ect
        
        stores best path. If rank of stored path is lower than num_best, removes
        path from cost and recurses with new cost array to find next best path
        '''
        self.bestpaths[rank] = noise[self.get_best_path(cost),:,:]
        if (rank < self.num_best):
            self.n_best_paths(np.delete(cost, self.get_best_path(cost), axis = 0), rank + 1)

    
    def get_best_path(self, cost):
        '''
        cost: float array((num_traj, timesteps))
        
        returns int index of best path 
        '''
        #sums across timesteps for each trajectory and returns trajectory with lowest cost
        return np.argmin(np.sum(cost, axis = 0))
        
    
    def cost(self, cassie_inputs):
        '''Computes cost-to-go using state information. Returns float.'''
        
        # Get pelvis positions and velocities for costs
        pos = np.zeros(7, dtype=np.double)
        vel = np.zeros(6, dtype=np.double)
        libcassie.cassie_pos(self.cassie, pos.ctypes.data_as(c_double_p))
        libcassie.cassie_vel(self.cassie, vel.ctypes.data_as(c_double_p))
        
        #quadratic distance between center of mass and center of pressure
        #[24] = right foot x pos, [25] = right foot y pos, [55]= left x, [56] = left y 
        stability_cost_x = (cassie_inputs[24] + cassie_inputs[55])**2 
        stability_cost_y = (cassie_inputs[25] + cassie_inputs[56])**2
        
        #height deviation penalty with forgiveness region
        height_cost = 0
        
        forgive_range = 0.3 #TODO no clue what range this should be in
        
        if(abs(pos[2] - 1) > forgive_range):
            height_cost = abs(pos[2] - 1)
            
        #TODO linear velocity deviation penalty with forgiveness region
        
        cost = stability_cost_x + stability_cost_y + height_cost
        
        return cost   
        
    def terminal_cost(self, init_pos, init_vel, init_acc, final_pos, final_vel, final_acc):
        '''Computes quadratic cost on error between initial and final mujoco states. Returns float.'''
        cost = 0
        #should these be weighted differently?
        cost += (init_pos - final_pos)**2 
        cost += (init_vel - final_vel)**2 
        cost += (init_acc - final_acc)**2 
        return cost

    def probability(self, cost):
        '''Returns a float probability. Probability decays exponentially relative to cost'''
        return np.exp(-1/self.sensitivity * cost)

    def avg_over_trajectories(self, prob, noise, M):
        '''
        prob: float(num_traj). Probabilities for each trajectory for a single timestep
        noise: float (num_traj, dims, basis_dims)
        
        Computes parameter update for each timestep by averaging noise across trajectories weighted
        by probability. This function is called for each timestep.
        
        returns float array((dims, basis_dims))
        '''
        avg = np.zeros((self.dims, self.basis_dims))
         
        for d in range(self.dims): 
            for k in range(self.num_traj): 
                #incrementing average by noise of parameter weighted by probability of that trajectory
                avg[d,:] += prob[k] * M[k,d,:,:] * noise[k,d,:]        
        return avg    

    def avg_over_timesteps(self, timestep_param_update):                         
        '''timestep_param_update: float array((timesteps, dims, basis_dims))
        
        This function takes in the parameter update for each timestep and averages across timesteps,
        each paramater update eighted by the number of steps left in the trajectory and the 
        activation of the corresponding basis function in the DMP at that timestep
        
        returns float array((dims, basis_dims))                 
        '''
        avg = np.zeros((self.dims, self.basis_dims))
        denominator = 0 #to scale avg
        for i in range(self.timesteps - 1): #for each timestep
            for j in range(self.basis_dims):
                #param update for basis j across all pydmp dimensions incremented by timestep param update
                #weighted by corresponding activation and steps left in trajectory  
                #get_activation is a placeholder
                avg[:,j] += (self.timesteps - i) * self.dmp.get_activations(activation = j, time = i) * \
                    timestep_param_update[i,:,j]     
                denominator += self.dmp.get_activations(activation = j, time = i) * (self.timesteps - i)
                                                                   
        return avg / denominator 
