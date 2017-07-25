import ctypes
import numpy as np

# TODO Shouldn't this be done elsewhere? What happens if two are loaded?
# Load cassie library
libcassie = ctypes.CDLL('libcassie/libcassie.so')
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
                 sensitivity = 0.1, \
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
        control_ratio: float array(num_basis) or float. If float, all dimensions have 
            the same control cost

        '''
        self.dmp = dmp
        self.params = params
        self.dims = self.params['dims']
        self.num_basis = self.params['num_basis'] 
        self.timesteps = timesteps
        self.num_traj = num_traj
        self.num_best = num_best
        self.sensitivity = sensitivity
        self.bestpaths = np.zeros((num_best, self.dims, self.num_basis))
        
        #If variance is a constant, self.variance becomes an array filled with that constant
        #If variance is an array, self.variance remains that array
        #If control_ratio is a constant, R is a diagonal matrix filled with that constant
        #If control_ratio is an array, R is a diagonal matrix with control_ratio as the diagonal
        ones_array = np.ones(self.num_basis)
        self.variance = variance * ones_array
        self.R = np.diag(control_ratio * ones_array)
        
        self.cassie_reset = cassie
        self.cassie = cassie
    
    def reset_cassie(self):
        '''resets cassie to state from __init__'''
        libcassie.cassie_copy(self.cassie, self.cassie_reset)
    
    def update_params(self, base_weights):
        '''        
        base_weights: float array((dims, basis dims))
            
        PI^2 Algorithm: 
            Creates num_traj rollouts from same start state with stochastic DMP parameters.
            Best trajectories from last iteration are included. 
            Datermines cost of each trial, probability of each trial, averages over trials,
            averages over timesteps, and then updates parameter with weighted average. 

        Returns array((dims, num_basis)) of updated DMP weights
        '''    
        
        #creating buffers    
        prob = np.zeros((self.num_traj, self.timesteps))
        cost = np.zeros((self.num_traj, self.timesteps))
        #timestep_param_update = np.zeros((self.timesteps, self.dims, self.num_basis))
        param_update = np.zeros((self.dims, self.num_basis))
        M = np.zeros((self.num_traj, self.timesteps, self.dims, self.num_basis, self.num_basis)) 
        x = np.zeros((self.num_traj, self.timesteps, self.dims))

        #generating noise array
        noise = self.generate_noise()

        # TODO Best paths need to be stored as weights, not as offset noise
        #if bestpaths not zeros, using num_best paths from previous iteration
        if np.any(self.bestpaths):
            noise[:self.num_best, :, :] = self.bestpaths 

        #creating rollouts and getting probabilities for K noisy trajectories
        for k in range (self.num_traj):
            #compute M and cost-to-go for each trajectory
            M[k,:,:,:,:], cost[k,:], x[k,:,:] = self.rollout(base_weights)
            #appending control cost
            for i in range(1, self.timesteps):
                for d in range (self.dims):
                    #TODO should control cost of each dimension be summed or averaged?
                    #TODO what should the ratio of state to control cost be?
                    cost[k,:i] += .5 * \
                        np.matmul(np.matmul(np.transpose(base_weights[d,:] \
                        + np.matmul(M[k,i,d,:,:], noise[k,d,:])), self.R), \
                        (base_weights[d,:] + np.matmul(M[k,i,d,:,:], noise[k,d,:])))
        #TODO append lambda term
            prob_sum = 0
            for i in range(self.timesteps):        
                prob[k,i] = self.probability(cost[k,i])
                prob_sum += prob[k,i]                
            prob[k,:] /= prob_sum  
        
#        #averaging over trials, getting param update for each timestep
#        for i in range (self.timesteps):
#            timestep_param_update[i,:,:] = self.avg_over_trajectories(prob[:,i], noise, M[:,i,:,:,:], x)
#
#        #averaging over timesteps, getting final param update
#        param_update = self.avg_over_timesteps(timestep_param_update)

        #getting parameter update averaging over timesteps, each timestep averaging over trajectories
        for d in range(self.dims):
            param_update[d,:] = self.avg_over_timesteps(x[:,:,d], prob, M[:,:,d,:,:], noise[:,d,:])

        #storing num_best of lowest cumulative cost paths
        self.n_best_paths(cost, 0)   
    
        return base_weights + param_update 

    # TODO What is in the array?  Gaussian noise with mean 0, self.variance variance.
    # TODO We should store and use std instead
    def generate_noise(self):
        '''returns float array((num_traj, dims, num_basis))'''

        noise = np.zeros((self.num_traj, self.dims, self.num_basis))
        for i in range(self.dims):
            noise[:,i,:] = np.random.normal(0,np.sqrt(self.variance[i]),(self.num_traj, self.num_basis)) 
        return noise 
    
    def compute_M(self, x):
        '''
        x: float. temporal variable of pydmp
        
        M is needed to project exploration noise onto parameter space
        
        Returns float array((num_basis, num_basis))
        '''
        #R is a diagonal matrix
        # TODO Precompute this if R never changes
        R_inverse = np.diag(.1/np.diagonal(self.R))
        #dmp.gen_psi returns a float array(basis dims) of activations at x       
        return np.matmul(np.matmul(R_inverse, self.dmp.gen_psi(x, self.params['centers'], self.params['widths'])) \
                , np.transpose(self.dmp.gen_psi(x, self.params['centers'], self.params['widths']))) \
                / np.matmul(np.matmul(np.transpose(self.dmp.gen_psi(x, self.params['centers'], self.params['widths'])) \
                   , R_inverse), self.dmp.gen_psi(x, self.params['centers'], self.params['widths']))
        
    def     rollout(self, params):
        '''
        params: float array(dims). initial weights. 
        
        Resets mujoco and pydmp state. Creates a rollout of specified trajectory and determines
        cost-to-go at each timestep. Cost here is a one-dimensional slice of the 2 dimensional 
        array defined in update_params. 
        
        This method is called for each trajectory.
        Returns cost (float array(timesteps)), M (float array((timesteps, dims, num_basis, num_basis),
        and x float array ((timesteps, dims))'''

        # TODO Make Cassie class?
        #create buffers
        init_pos = np.zeros(35, dtype=np.double)
        init_vel = np.zeros(32, dtype=np.double)
        init_acc = np.zeros(32, dtype=np.double)
        final_pos = np.zeros(35, dtype=np.double)
        final_vel = np.zeros(32, dtype=np.double)
        final_acc = np.zeros(32, dtype=np.double)
        cassie_input = np.zeros(98, dtype=np.double)
        cassie_output = np.zeros(20, dtype=np.double)
        cost = np.zeros(self.timesteps)
        M = np.zeros((self.timesteps, self.dims, self.num_basis, self.num_basis))
        x = np.zeros((self.timesteps, self.dims))

        #resetting cassie and copying initial state
        self.reset_cassie()
        libcassie.copy_data(self.cassie, init_pos.ctypes.data_as(c_double_p), init_vel.ctypes.data_as(c_double_p), init_acc.ctypes.data_as(c_double_p))
        
        #resetting pydmp and setting weights in params dict
        self.dmp.reset_state()
        self.params['weights'] = params

        for i in range(self.timesteps):
            cassie_output[8:], _, _, x[i,:] = self.dmp.step(**self.params)
            libcassie.cassie_step2(self.cassie, cassie_output.ctypes.data_as(c_double_p))
            libcassie.cassie_step1(self.cassie, cassie_input.ctypes.data_as(c_double_p))
            #computing M for all dimensions corresponding x at timestep i 
            for d in range(self.dims):
                M[i,d,:,:] = self.compute_M(x[i, d])
            #computing cost to go
            #appends instantaneous cost to current and previous timesteps
            cost[:i+1] += self.cost(cassie_input)                  

        #copying final state
        libcassie.copy_data(self.cassie, final_pos.ctypes.data_as(c_double_p), final_vel.ctypes.data_as(c_double_p), final_acc.ctypes.data_as(c_double_p))

        #appending terminal cost to each instantaneous cost
        cost += self.terminal_cost(init_pos, init_vel, init_acc, final_pos, final_vel, final_acc)

        return M, cost, x

    # TODO What are the inputs of this function?  Why does it exist?
    def get_cumulative_cost(self, params):
        '''
        params: array(dims) of floats. initial weights. 
        cost: array(timesteps) of floats
        
        returns scalar cumulative cost for one rollout'''

        _, cost, _ = self.rollout(params)
        return cost
    
    def n_best_paths(self, cost, rank, noise):
        '''
        cost: float array((num_traj, timesteps))
        rank: int. rank 0 is best path, rank 1 is second best, ect
        
        stores best path. If rank of stored path is lower than num_best, removes
        path from cost and recurses with new cost array to find next best path
        '''
        # TODO This will be slow!
        self.bestpaths[rank] = noise[self.get_best_path(cost),:,:]
        if (rank < self.num_best):
            self.n_best_paths(np.delete(cost, self.get_best_path(cost), axis = 0), rank + 1)

    # TODO Don't you already have summed cost?
    def get_best_path(self, cost):
        '''
        cost: float array((num_traj, timesteps))
        
        returns int index of best path 
        '''
        #sums across timesteps for each trajectory and returns trajectory with lowest cost
        return np.argmin(np.sum(cost, axis = 0))
        
    # TODO this should be an argument to the class, not a class method
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

    # TODO this should be an argument to the class, not a class method
    def terminal_cost(self, init_pos, init_vel, init_acc, final_pos, final_vel, final_acc):
        '''Computes quadratic cost on error between initial and final mujoco states. Returns float.'''
        cost = 0
        #should these be weighted differently?
        cost += np.sum((init_pos - final_pos)**2)
        cost += np.sum((init_vel - final_vel)**2) 
        cost += np.sum((init_acc - final_acc)**2)
        return cost

    def probability(self, cost):
        '''
        cost: float 
        
        Returns a float probability. Probability decays exponentially relative to cost'''
        return np.exp(-1/self.sensitivity * cost)

#    def avg_over_trajectories(self, prob, noise, M):
#        '''
#        prob: float(num_traj). Probabilities for each trajectory for a single timestep
#        noise: float (num_traj, dims, num_basis)
#
#        Computes parameter update for each timestep by averaging noise across trajectories weighted
#        by probability. This function is called for each timestep.
#
#        returns float array((dims, num_basis))
#        '''
#        avg = np.zeros((self.dims, self.num_basis))
#
#        for d in range(self.dims):
#            for k in range(self.num_traj):
#                #incrementing average by noise of parameter weighted by probability of that trajectory
#                avg[d,:] += np.matmul(prob[k] * M[k,d,:,:], noise[k,d,:])
#        return avg
#
#    def avg_over_timesteps(self, timestep_param_update, x):
#        '''timestep_param_update: float array((timesteps, dims, num_basis))
#        x: float array((num_traj????, timesteps, dims))
#
#        This function takes in the parameter update for each timestep and averages across timesteps,
#        each paramater update weighted by the number of steps left in the trajectory and the
#        activation of the corresponding basis function at that timestep
#
#        returns float array((dims, num_basis))
#        '''
#        avg = np.zeros((self.dims, self.num_basis))
#        denominator = 0 #to scale avg
#        for i in range(self.timesteps - 1): #for each timestep
#            for j in range(self.num_basis):
#                #param update for basis j across all pydmp dimensions incremented by timestep param update
#                #weighted by corresponding activation and steps left in trajectory
#                avg[:,j] += (self.timesteps - i) * self.dmp.gen_psi() * \
#                    timestep_param_update[i,:,j]
#                denominator += self.dmp.gen_psi() * (self.timesteps - i)
#
#        return avg / denominator


    def avg_over_timesteps(self, x, prob, M, noise):
        '''
        x: float array((num_traj, timesteps))
        prob: float array((num traj, timesteps))
        M: float array((num_traj, timesteps, num_basis, num_basis))
        noise: float array((num_traj, num_basis))

        This function calculates the parameter update across trajectories for each timestep and takes
        the average across timesteps weighted by the number of steps left in the trajectory.
        This function is called for each dimension.

        returns float array(num_basis)
        '''
        avg = np.zeros(self.num_basis)
        denominator = np.zeros(self.dims)
        for i in range(self.timesteps):
            trajectories_sum, activations = self.avg_over_trajectories(x[:,i], prob[:,i], M[:,i,:,:], noise)
            avg += (self.timesteps - i) * trajectories_sum
            denominator += np.mean(activations) * (self.timesteps - i)
        return avg / denominator


    def avg_over_trajectories(self, x, prob, M, noise):
        '''
        x: float array(num_traj)
        p: float array(num_traj)
        M: float array((num_traj, num_basis, num_basis))
        noise: float array ((num_traj, num_basis))

        Averages added noise over trajectories to get parameter update at a single timestep.
        Update is weighted by activation of corresponding basis function at timestep and
        probability. This function is called for each timestep for each dimension.

        returns:
        trajectories_sum: float array(num_basis)
        activations: float array(num_basis)
        '''
        trajectories_sum = np.zeros(self.num_basis)
        activations = np.zeros(self.num_basis)
        for k in range(self.num_traj):
            #self.dmp.gen_psi(x[k]) returns float array(num_basis)
            activations[k] = self.dmp.gen_psi(x[k], self.params['centers'], self.params['widths'])
            trajectories_sum += activations[k] * prob[k] * np.matmul(M[k,:,:], noise[k,:])
        return trajectories_sum, activations


# TODO speed up computations