import numpy as np
import math
import scipy.signal
import time
import scipy.optimize as opt
import os

class IslandsEvoAdaptiveStep:
    # Runs evolution simulation for the island ecology model.
    # Adaptive step size: The ecological dynamics use an adaptive step-size. Each step, the timestep is computed so that the fractional change in abundances is at most +- max_frac_change. 
    # Mutants: 
        ## Each "epoch" new mutants are allowed to invade and the ecological dynamics are run to determine the surviving strains. 
        ## The invasion eigenvalue for each new mutant is estimated before invasion and only those with eigenvalues similar to previously successful invasions are allowed to invade. This avoids wasting time on unsuccessful invasions. 
        ## New mutants are added at frequency invasion_freq_factor/K. The default value inserts new mutants at relatively large size so that they have the best chance of successfully invading.
    # Saving: 
        ## Each epoch the results are various average quantities are saved using an instance of the save object, EvoSavedQuantitiesAdaptiveStep. 
        ## Every new_save_epoch_num epochs new save file is created and a new instance of the save object is created. This keeps a low memory profile even for very long runs.
    # Normalization: abundances are normalized to be frequencies and interactions/selective differences are assumed to be order one.
    # Timescales: each epoch lasts for a time epoch_timescale*K*M, corresponding to the long timescale for marginal types with biases of order 1/K.

    def __init__(self, file_name, D, K, m, gamma, thresh, mu, seed, epoch_timescale, epoch_num=np.inf, 
                 corr_mut=0, sig_S=0, max_frac_change=0.75, invasion_freq_factor = 1,
                 first_epoch_timescale=4,
                 invasion_criteria_memory=100, invasion_eig_buffer=0.2, new_save_epoch_num = 50,
                 S_distribution='gaussian', S_tail_power = 1,
                 long_epochs=None,long_factor=5,
                 epochs_to_save_traj = None, sample_num = 1, save_interactions = False,
                 n_init=None, V_init = None, S_init = None, invasion_eigs_init = None, 
                 init_species_idx = None, new_species_idx = None, start_epoch_num = 0):
        
        # General simulation parameters 
            # input file_name: prefix for save files.
            # input D: number of islands
            # input K: initial number of strains
            # input m: migration rate
            # input gamma: symmetry parameter
            # input thresh: extinction threshold for log-frequency equal to log(1/N) for total population N.
            # input mu: number of strains invading per epoch
            # input seed: random seed
            # input epoch_timescale: time for each epoch, units of K*M
            # input epoch_num: number of epoch. Default of np.inf runs until cluster job times out.
            # input corr_mut: correlation of new types with previous ones
            # input sig_S: standard dev. of selective differences
            # input max_frac_change: the maximum fractional change in frequency allowed across islands. Sets adaptive step size.
            # input invasion_freq_factor: new invasions come in at freq invasion_freq_factor/K
            # input first_epoch_timescale: timescale for first epoch, units of sqrt(K)*M

        # Parameters for invading new types
            # input invasion_criteria_memory: number of most recent successful invasions kept
            # input invasion_eig_buffer: sets threshold below the minimum successful invasion eigenvalues for new invasions.

        # Parameters for distribution of new types
            # input S_distribution:
                # 'gaussian' for gaussian distribution.
                # 'exponential_tail' for tail of the form exp(-(x/sig_S)**p) where p = S_tail_power. Distribution is for S>0.

        # Debugging with long epochs, used to see if additional strains go extinct when epochs are longer.
            # input long_epochs: list of epochs that are long
            # input long_factor: long epochs are factor long_factor longer.

        # Parameters for saving 
            # input new_save_epoch_num: number of epochs after which a new save file is created.
            # input epochs_to_save_traj: list of epochs to save trajectories for. Use -1 to save most recent epoch. 
                ## Saves trajectories for all strains on a single island. May take up a lot of memory.
            # input sample_num: for trajectories, saves one step for every sample_num steps.
            # input save_interactions: save interaction matrix in a dictionary. Makes save files larger, partly because
                # dictionary is saved in npz file as array, which is somewhat unnatural.

        # Parameters for extending simulations for additional epochs
            # input n_init: initial frequencies to start with
            # input V_init: interaction matrix to start with
            # input S_init: selective differences to start with
            # input invasion_eigs_init: list of successful invasion eigs to start with
            # input init_species_idx: list of species indices to start with
            # input new_species_idx: species index to start with next
            # input start_epoch_num: epoch number to start from


        self.file_name = file_name
        self.D = D
        self.K0 = K
        self.m = m
        self.gamma = gamma
        self.N = 1  #frequency normalization
        self.thresh = thresh
        self.invasion_freq_factor = invasion_freq_factor
        self.first_epoch_timescale = first_epoch_timescale
        self.epoch_timescale = epoch_timescale
        self.epoch_num = epoch_num
        self.mu = mu
        self.seed = seed
        self.sample_num = sample_num
        self.max_frac_change = max_frac_change
        self.corr_mut = corr_mut
        self.sig_S = sig_S
        self.S_distribution = S_distribution
        self.S_tail_power = S_tail_power
        if epochs_to_save_traj is None:
            self.epochs_to_save_traj = []
        else:
            self.epochs_to_save_traj = epochs_to_save_traj
        self.save_interactions = save_interactions
        self.n_init = n_init
        self.V_init = V_init
        self.S_init = S_init
        self.invasion_eigs_init = invasion_eigs_init
        self.init_species_idx = init_species_idx
        self.new_species_idx = new_species_idx
        if long_epochs is None:
            self.long_epochs = []
        else:
            self.long_epochs = long_epochs
        self.long_factor = long_factor
        self.invasion_criteria_memory = invasion_criteria_memory
        self.invasion_eig_buffer = invasion_eig_buffer
        self.new_save_epoch_num = new_save_epoch_num
        self.start_epoch_num = start_epoch_num

        if V_init is not None:
            self.K0 = V_init.shape[0]

        if n_init is not None:
            self.K0 = n_init.shape[1]
            self.D = n_init.shape[0]

        self.sim_start_time = time.time()
        self.sim_start_process_time = time.process_time()

    def start_simulation(self):
        self.save_parameters()
        self.evo_sim()

    def extend_simulation(self):
        self.evo_sim()

    def save_parameters(self):
        # Creates separate save file with parameters.
        file_name = self.file_name+'.params'
        data = vars(self)
        np.savez(file_name,data = data)

    def evo_sim(self):
        # Run evolutionary dynamics and saves data

        np.random.seed(seed=self.seed)  # Set random seed
        self.initializations()  #Sets V, S, and n0.

        # evolution steps
        cur_epoch = self.start_epoch_num
        while cur_epoch < self.start_epoch_num + self.epoch_num:
            self.setup_save_object(cur_epoch) #creates save object. Uses new file_name after new_file_epoch_num
            self.evo_step(cur_epoch) # dynamics for one epoch. V, n0 are for non-extinct types
            self.mut_step(cur_epoch)  # add new types

            self.SavedQuants.save_data() # Save data in case job terminates on cluster

            num_types = self.n0.shape[1]
            if num_types <= 10+self.mu:
                break
            print(cur_epoch)
            cur_epoch += 1

    def initializations(self):
        # interaction matrix
        if self.V_init is None:
            self.V = generate_interactions_with_diagonal(self.K0, self.gamma)
        else:
            self.V = self.V_init

        # initial frequencies
        if self.n_init is None:
            self.n0 = initialization_many_islands(self.D, self.K0, self.N, self.m, 'flat')
        else:
            self.n0 = self.n_init

        # initial selective differences
        if self.S_init is None:
            S = self.draw_S_distribution(self.K0)
            self.S = np.reshape(S,(1,self.K0))
        else:
            self.S = self.S_init

        #species index for next type
        if self.init_species_idx is not None:
            self.current_species_idx = self.init_species_idx  
        else:
            self.current_species_idx = np.arange(self.K0)

        # list of successful invasion eigenvalues
        if self.invasion_eigs_init is not None:
            self.invasion_success_eigs = self.invasion_eigs_init
        else:
            self.invasion_success_eigs = []

         # Define idx for next species, if not already defined
        if self.new_species_idx is None:
            self.new_species_idx = max(self.current_species_idx)+1

    def draw_S_distribution(self,num):
        # Draw from distribution of S
        # input num: number of draws to make
        # return S: (num,) vec of S values.

        if self.S_distribution is 'gaussian':
            S = self.sig_S * np.random.normal(size=(num))

        elif self.S_distribution is 'exponential_tail':
            p = self.S_tail_power

            min_prob = 1e-12
            s_max = (np.log(1/min_prob))**(1./p)
            if s_max > 8:
                ds = s_max/3e4
            else:
                s_max = 8
                ds = 0.0003

            s_vec = np.arange(0, s_max, ds) #units of sig_S
            prob_density = np.exp(-s_vec**p)*ds
            prob_density *= 1/np.sum(prob_density)
            cum_prob = np.cumsum(prob_density)
            rand_nums = np.random.rand(num)
            indices = np.digitize(rand_nums, cum_prob)
            S = self.sig_S * s_vec[indices]

        return S

    def evo_step(self, cur_epoch):
        # Runs dynamics for one epoch. 

        # Basic variables. 
        # n: (D,K) array of frequencies
        # xbar0: (1,K) array of log of island averaged frequencies
        # y: (D,K) array of n/xbar0 values.
            ## Use y and xbar0 so that arbitrarily small frequencies can be considered.
            ## If n was only used then under flow when n< 1e-300 or so. 
            ## If only log-frequencies were used, issues with migration term.
        
        V = self.V
        S = self.S
        n0 = self.n0
        SavedQuants = self.SavedQuants

        K = np.shape(n0)[1]
        D = self.D
        m = self.m
        N = self.N
        thresh = self.thresh

        M = self.define_M(K,m)
        epoch_time = self.setup_epoch(cur_epoch,K,M)

        # Define functions as local variables. May improve speed.
        Normalize = self.Normalize
        Extinction = self.Extinction
        step_forward = self.define_step_forward()
        check_new_types_extinct = self.check_new_types_extinct

        nbar = np.mean(n0, axis=0, keepdims=True)
        xbar0 = np.log(nbar) # Assumes all types have nbar>0
        y0 = n0 / nbar
        n0 = y0 * np.exp(xbar0)

        # initialize for epoch
        SavedQuants.initialize_epoch(D,K,V,S,epoch_time,n0,self.current_species_idx)

        # dynamics
        time = 0
        while time < epoch_time:

            # Step abundances forward
            y1, xbar1, dt, stabilizing_term = step_forward(y0,xbar0)
            time += dt
            y1, xbar1 = Extinction(y1, xbar1, thresh)
            y1, xbar1 = Normalize(y1, xbar1, N)

            # Save values based on initial abundances.
            SavedQuants.save_sample(dt, time, y0, xbar0, stabilizing_term, cur_epoch)

            # Prep for next time step
            y0 = y1
            xbar0 = xbar1

            # if all new types extinct, move to next epoch
            # new_type_extinct_bool = check_new_types_extinct(xbar0,cur_epoch)
            # if new_type_extinct_bool and cur_epoch not in self.long_epochs:
            #     self.epoch_time_list[cur_epoch] = step*dt
            #     break

            ######### end epoch time steps

        ####### Compute averages and save
        SavedQuants.compute_averages(time)
        SavedQuants.save_to_lists(cur_epoch,xbar0,self.mu,self.current_species_idx)

        ####### Change number of surviving species.
        surviving_bool = self.subset_to_surviving_species(y0,xbar0,cur_epoch,SavedQuants)
        self.save_invasion_results(SavedQuants,surviving_bool,cur_epoch)


    def define_M(self,K,m):
        #Define log-migration parameter
        M = np.log(1/(m*np.sqrt(K)))
        return M

    def setup_epoch(self,cur_epoch,K,M):
        # Set epoch_time

        m = self.m
        N = self.N
        max_frac_change = self.max_frac_change

        # epoch time
        if cur_epoch == 0:
            epoch_timescale = self.first_epoch_timescale*K**(-0.5)  # First epoch only for 4 * K**(0.5) * M
        elif cur_epoch in self.long_epochs:
            epoch_timescale = self.epoch_timescale * self.long_factor
        else:
            epoch_timescale = self.epoch_timescale
        epoch_time = epoch_timescale*(K**1)*M  # epoch_timescale*(long timescale for marginal types (bias ~ 1/sqrt(K)) )
        return epoch_time

    def setup_save_object(self,cur_epoch):
        # Determine file name from cur_epoch and new_save_epoch_num
        # If new save file needed, starts new save object from EvoSavedQuantitiesAdaptiveStep

        epoch_num = self.new_save_epoch_num
        if cur_epoch % epoch_num == 0:
            file_name = self.file_name+'.{}'.format(cur_epoch//epoch_num)
            SavedQuants = EvoSavedQuantitiesAdaptiveStep(file_name, self.sim_start_time,
                                                         self.sim_start_process_time, self.sample_num,
                                                         self.epochs_to_save_traj,self.save_interactions)
            SavedQuants.initialize_dicts(self.V, self.current_species_idx)  # store interactions and selective diffs in dictionaries
            self.SavedQuants = SavedQuants

    def define_step_forward(self):
        # Define the step_forward function with the V and S for the current epoch. 
        # May save time to define this way.

        N = self.N
        m = self.m
        max_frac_change = self.max_frac_change
        V = self.V
        S = self.S

        def step_forward(y,xbar):
            # Returns y after one step forward. Assumes xbar is a fixed constant.
            # Step size, dt, is computed by making dt*log_deriv is at most max_frac_change in magnitude.
                # log_deriv is the log derivative of y. 
                # Typically set by strains at very low frequency since strains at large abundance have log_deriv reduced by the response term.

            n = np.exp(xbar) * y
            D = n.shape[0]
            growth_rate = S + np.einsum('ij,dj', V, n)

            stabilizing_term = np.einsum('di,di->d', n, growth_rate) / N  #stabilizing term (Upsilon)
            stabilizing_term = np.reshape(stabilizing_term, (D, 1))

            y_dot = y * (growth_rate - stabilizing_term)

            if m == 0:
                y_dot = y_dot
            else:
                y_dot = y_dot + m * (1 - y)

            log_deriv = y_dot[y > 0] / y[y > 0]
            if np.max(log_deriv) ==0:
                print('fixed pt reached')
                dt = 1
            else:
                dt_pos = max_frac_change / np.max(log_deriv)  # from max of positive log_deriv
                dt_neg = -max_frac_change / np.min(log_deriv)
                dt = np.min([dt_pos, dt_neg])
            y1 = y + y_dot * dt

            return y1, xbar, dt, stabilizing_term

        return step_forward


    def subset_to_surviving_species(self,y0,xbar0,cur_epoch,SavedQuants):
        surviving_bool = xbar0[0, :] > -np.inf  # surviving species out of K1 current species.
        SavedQuants.surviving_bool_list.append(surviving_bool)
        self.current_species_idx = self.current_species_idx[surviving_bool]

        # only pass surviving types to next timestep
        n0 = np.exp(xbar0) * y0
        self.n0 = n0[:, surviving_bool]
        self.V = self.V[np.ix_(surviving_bool, surviving_bool)]
        self.S = self.S[:,surviving_bool]

        return surviving_bool

    def mut_step(self,cur_epoch):
        #Generate new mutants. Updates list of current species indices.

        V = self.V
        S = self.S
        n0 = self.n0
        SavedQuants = self.SavedQuants

        mu = self.mu
        K = np.shape(V)[0]
        V_new, S_new, parent_idx_list = self.gen_new_invasions(V,S,SavedQuants,cur_epoch) #generated interactions correlated with parent

        # set new invasion types
        n0_new = self.next_epoch_abundances(mu, n0)

        new_species_idx = self.new_species_idx
        species_idx = np.append(self.current_species_idx,np.arange(new_species_idx,new_species_idx+mu))
        self.current_species_idx = species_idx
        self.new_species_idx = new_species_idx + mu

        self.V = V_new
        self.S = S_new
        self.n0 = n0_new

        SavedQuants.store_new_interactions(V_new,mu,species_idx,parent_idx_list)

    def gen_new_invasions(self,V,S,SavedQuants,cur_epoch):
        #Generates possible invasions. Estimates invasion eig. 
        # And evaluates whether new mutants passes invasion criteria.
        # If it passes, its interactions and selected differences are added to V and S
        # Keeps a list of parent indices for each new mutant.

        mu = self.mu
        K = np.shape(V)[0]
        V_new = np.zeros((K+self.mu,K+self.mu))
        V_new[0:K, 0:K] = V
        S_new = np.zeros((1,K+self.mu))
        S_new[0,0:K] = S

        parent_idx_list = []
        invasion_eigs = []
        invasion_counts = 0
        for idx in range(K,K+mu):

            invade_bool = False
            while not invade_bool:
                par_idx = np.random.choice(K)
                V_row, V_col, V_diag, s = self.gen_related_interactions(V_new,S,par_idx,idx)
                invasion_eig = self.compute_invasion_eig(SavedQuants,V_row[:K],s)
                invade_bool = self.invasion_criteria(invasion_eig,K,cur_epoch)
                invasion_counts += 1

            invasion_eigs.append(invasion_eig)
            parent_idx_list.append(int(self.current_species_idx[par_idx]))
            V_new[idx, 0:idx] = V_row
            V_new[0:idx, idx] = V_col
            V_new[idx, idx] = V_diag
            S_new[0, idx] = s

        self.invasion_eigs = invasion_eigs
        self.invasion_rejects = invasion_counts-mu

        return V_new, S_new, parent_idx_list

    def gen_related_interactions(self,V_new,S_new,par_idx,idx):
        # Generates new types from random parents with interaction matrix correlated with parent.

        corr_mut = self.corr_mut
        cov_mat = [[1., self.gamma], [self.gamma, 1.]]
        if self.gamma == -1:
            z_vec = np.random.randn(idx)
            V_row = corr_mut * V_new[par_idx, 0:idx] + np.sqrt(1 - corr_mut ** 2) * z_vec  # row
            V_col = corr_mut * V_new[0:idx, par_idx] - np.sqrt(1 - corr_mut ** 2) * z_vec  # column
            V_diag = 0  # diagonal
        else:
            z_mat = np.random.multivariate_normal([0, 0], cov=cov_mat, size=idx)  # 2 x (K+k) matrix of differences from parent
            V_row = corr_mut * V_new[par_idx, 0:idx] + np.sqrt(1 - corr_mut ** 2) * z_mat[:, 0]  # row
            V_col = corr_mut * V_new[0:idx, par_idx] + np.sqrt(1 - corr_mut ** 2) * z_mat[:, 1]  # column
            V_diag = np.sqrt(1 + self.gamma) * np.random.normal()  # diagonal

        if self.S_distribution is 'gaussian':
            s = corr_mut * S_new[0, par_idx] + np.sqrt(1 - corr_mut ** 2) * self.draw_S_distribution(1)
        elif self.S_distribution is 'exponential_tail':
            #TODO implement correlated mutation for different S distributions.
            # Need to use random walk process with correct equilibrium distribution.
            s = self.draw_S_distribution(1)

        return V_row, V_col, V_diag, s

    def compute_invasion_eig(self,SavedQuants,row,s):
        # Computes invasion eigenvalue based on previous epochs mean abundances and stabilizing_term

        surviving_bool = SavedQuants.surviving_bool_list[-1]
        n_mean = SavedQuants.n_mean_ave_list[-1]
        stabilizing_term = SavedQuants.stabilizing_term_mean_ave_list[-1]  #includes S_mean
        invasion_eig = s + np.dot(row,n_mean[surviving_bool]) - stabilizing_term
        return invasion_eig

    def invasion_criteria(self,invasion_eig,K,cur_epoch):
        # find the minimum invasion eigenvalue of the last x successful invasions
        # Where x is self.invasion_criteria_memory
        # Return invade_bool as True if invasion_eig is within invasion_eig_buffer of the minimum.

        success_eigs = np.array(self.invasion_success_eigs)
        if len(success_eigs) >= self.invasion_criteria_memory:

            # Twice per invasion_criteria_memory invade types with eig below the minimum threshold to see.
            min_eig = np.min(success_eigs)
            eig_scale = np.sqrt(1 / K + self.sig_S ** 2)  # estimate of eig standard dev

            half_memory = int(self.invasion_criteria_memory/(2*self.mu))
            if cur_epoch % half_memory == 0:
                invade_bool = invasion_eig > min_eig-5*self.invasion_eig_buffer*eig_scale
            else:
                invade_bool = invasion_eig > min_eig - self.invasion_eig_buffer*eig_scale
        else:
            invade_bool = True
        return invade_bool

    def next_epoch_abundances(self,mu,n0):
        # Sets frequency of new mutant to invasion_freq_factor/K where K is number of strains.
        # Starting at large frequency allows best chance for strain to invade. Depends on invasion eigenvalue.
        # Starting at very low frequency makes extinction sensitive to short time dynamics.

        D = self.D
        K = len(self.current_species_idx)
        n0_new = np.zeros((D,K+mu))
        n0_new[:,:K] = n0
        n0_new[:,K:] = self.invasion_freq_factor/K
        n0_new = n0_new/np.sum(n0_new,axis=1,keepdims=True)
        return n0_new

    def save_invasion_results(self,SavedQuants, surviving_bool,cur_epoch):

        if cur_epoch==self.start_epoch_num:
            SavedQuants.invasion_rejects_list.append([])
            SavedQuants.invasion_eigs_list.append([])
            SavedQuants.invasion_success_list.append([])
        else:
            if self.mu>0:
                invasion_success_bool = surviving_bool[-self.mu:]
            else:
                invasion_success_bool = []
            SavedQuants.invasion_rejects_list.append(self.invasion_rejects)
            SavedQuants.invasion_eigs_list.append(self.invasion_eigs)
            SavedQuants.invasion_success_list.append(invasion_success_bool)

            eigs = np.array(self.invasion_eigs)
            success_eigs = self.invasion_success_eigs
            success_eigs.extend(eigs[invasion_success_bool])
            if len(success_eigs) > self.invasion_criteria_memory:
                success_eigs = success_eigs[-self.invasion_criteria_memory:]
            self.invasion_success_eigs = success_eigs


    def Normalize(self, y, xbar, N):
        # Normalize frequencies and redefine xbar
        n = np.exp(xbar) * y
        Nhat = np.sum(n, axis=1, keepdims=True)
        Yhat = np.mean(y * N / Nhat, axis=0, keepdims=True)

        Yhat[Yhat == 0] = 1

        y1 = (y * N / Nhat) / Yhat
        xbar1 = xbar + np.log(Yhat)

        return y1, xbar1

    def Extinction(self, y, xbar, thresh):
        # Evaluates extinction
        y[y<0] = 0
        local_ext_ind = xbar + np.log(y) < thresh
        y[local_ext_ind] = 0

        global_ext_ind = np.all(y == 0, axis=0)
        xbar[:, global_ext_ind] = -np.inf
        y[:, global_ext_ind] = 0

        return y, xbar

    def check_new_types_extinct(self,xbar,cur_epoch):
        if cur_epoch>0:
            new_type_extinct_bool = np.all(xbar[0,-self.mu:]==-np.inf)
        else:
            new_type_extinct_bool = False
        return new_type_extinct_bool

class EvoSavedQuantitiesAdaptiveStep:
    # Object that saves quantities.

    def __init__(self,file_name,sim_start_time,sim_start_process_time,sample_num,epochs_to_save_traj,save_interactions):
        self.file_name = file_name
        self.sim_start_time = sim_start_time
        self.sim_start_process_time = sim_start_process_time
        self.sample_num = sample_num
        self.epochs_to_save_traj = epochs_to_save_traj
        self.save_interactions = save_interactions

        # intialize lists to store values for each epoch
        self.epoch_time_list = []  # list of total times for each epoch
        self.n_init_list = []  # initial distribution of frequencies
        self.surviving_bool_list = [] # list of boolean vecs for whether strain survived
        self.starting_species_idx_list = [] #list of starting species indices

        self.invasion_eigs_list = [] #list of invasion eigenvalues
        self.invasion_rejects_list = [] #list of number of possible invasions rejected by criteria
        self.invasion_success_list = [] #list of booleans for whether invasions were successful.

        self.n_mean_ave_list = []  # mean (over time) frequency averaged over islands.
        self.n2_mean_ave_list = []  # mean freq**2 averaged over islands.
        self.stabilizing_term_mean_ave_list = []  # mean stabilizing term averaged over islands.
        self.n_mean_std_list = [] # std of mean frequency over islands
        self.n2_mean_std_list = [] # std of mean freq**2 over islands
        self.stabilizing_term_mean_std_list = []  # std of mean stabilizing term over islands
        self.force_mean_ave_list = []  # mean V*n averaged over islands.
        self.force_mean_std_list = [] # std of mean V*n over islands.
        self.S_list = [] #list of S vecs.
        self.S_mean_ave_list = []  # mean S averaged over islands
        self.S_mean_std_list = []  # std of mean S over islands
        self.dt_mean_list = []  # mean dt over epoch
        self.dt2_mean_list = []  # mean dt**2 over epoch

        self.n_traj_dict = {}
        self.time_vec_dict = {}

    def initialize_dicts(self,V,current_species_idx):
        if self.save_interactions:
            V_dict = {}
            K = V.shape[0]
            for ii in range(K):
                idx0 = current_species_idx[ii]
                for jj in range(K):
                    idx1 = current_species_idx[jj]
                    V_dict[idx0,idx1] = V[ii, jj]
            self.V_dict = V_dict

        self.parent_idx_dict = {}

    def initialize_epoch(self,D,K,V,S,epoch_time,n_init,current_species_idx):
        self.epoch_time_list.append(epoch_time)
        self.n_init_list.append(n_init)
        self.starting_species_idx_list.append(current_species_idx)
        self.V = V
        self.S = S
        self.count = 0
        self.n_mean_array = np.zeros((D, K))
        self.n2_mean_array = np.zeros((D, K))
        self.stabilizing_term_mean_array = np.zeros((D))
        self.steps = 0
        self.dt_mean = 0
        self.dt2_mean = 0
        self.n_traj = []
        self.time_vec = []

    def save_sample(self,dt,time,y0,xbar0,stabilizing_term,cur_epoch):

        n0 = np.exp(xbar0) * y0
        self.n_mean_array += dt * n0
        self.n2_mean_array += dt * n0 ** 2
        self.stabilizing_term_mean_array += dt * stabilizing_term.flatten()
        self.steps += 1
        self.dt_mean += dt
        self.dt2_mean += dt**2

        if cur_epoch in self.epochs_to_save_traj or -1 in self.epochs_to_save_traj:
            if self.count % self.sample_num == 0:
                self.count += 1
                self.n_traj.append(n0[0, :])
                self.time_vec.append(time)

    def compute_averages(self,time):
        self.n_mean_array *= 1 / time
        self.n2_mean_array *= 1 / time
        self.stabilizing_term_mean_array *= 1 / time
        self.dt_mean *= 1/self.steps
        self.dt2_mean *= 1/self.steps

    def save_to_lists(self,cur_epoch,xbar,mu,species_idx):
        # input mu: number of mutants
        # input species_idx: current species idx

        n_mean_ave = np.mean(self.n_mean_array, axis=0)
        n2_mean_ave = np.mean(self.n2_mean_array, axis=0)
        stabilizing_term_mean_ave = np.mean(self.stabilizing_term_mean_array, axis=0)

        n_mean_std = np.std(self.n_mean_array, axis=0)
        n2_mean_std = np.std(self.n2_mean_array, axis=0)
        stabilizing_term_mean_std = np.std(self.stabilizing_term_mean_array, axis=0)

        force_mean = np.einsum('ij,dj',self.V,self.n_mean_array)
        force_mean_ave = np.mean(force_mean,axis=0)
        force_mean_std = np.std(force_mean, axis=0)

        S_mean = np.sum(self.S*self.n_mean_array,axis=1)
        S_mean_ave = np.mean(S_mean)
        S_mean_std = np.std(S_mean)

        self.n_mean_ave_list.append(n_mean_ave)
        self.n2_mean_ave_list.append(n2_mean_ave)
        self.stabilizing_term_mean_ave_list.append(stabilizing_term_mean_ave)
        self.force_mean_ave_list.append(force_mean_ave)

        self.S_list.append(self.S)
        self.S_mean_ave_list.append(S_mean_ave)

        self.n_mean_std_list.append(n_mean_std)
        self.n2_mean_std_list.append(n2_mean_std)
        self.stabilizing_term_mean_std_list.append(stabilizing_term_mean_std)
        self.force_mean_std_list.append(force_mean_std)
        self.S_mean_std_list.append(S_mean_std)

        self.dt_mean_list.append(self.dt_mean)
        self.dt2_mean_list.append(self.dt2_mean)

        self.save_traj(cur_epoch)

    def save_traj(self,cur_epoch):
        if cur_epoch in self.epochs_to_save_traj:
            self.n_traj_dict[cur_epoch] = np.array(self.n_traj).T
            self.time_vec_dict[cur_epoch] = np.array(self.time_vec)

        if -1 in self.epochs_to_save_traj:
            self.n_traj_dict[-1] = np.array(self.n_traj).T
            self.time_vec_dict[-1] = np.array(self.time_vec)

    def store_new_interactions(self,V,mu,current_species_idx,parent_idx):
        K = V.shape[0]
        parent_idx_iter = iter(parent_idx)
        for ii in range(K-mu,K):
            new_idx = current_species_idx[ii]
            self.parent_idx_dict[new_idx] = next(parent_idx_iter)
            if self.save_interactions:
                for jj in range(K):
                    idx = current_species_idx[jj]
                    self.V_dict[idx,new_idx] = V[jj,ii]
                    self.V_dict[new_idx, idx] = V[ii,jj]

    def save_data(self):
        self.sim_end_time = time.time()
        self.sim_end_process_time = time.process_time()
        self.V_end = self.V  #save final interaction matrix.
        data = vars(self)

        # Remove some variables
        data.pop('n_traj',None)
        data.pop('V', None)
        data.pop('S', None)

        np.savez(self.file_name,data = data)

def extend_adaptive_step_sim(file_prefix,epochs = None):
    # Takes save file generated from IslandEvoAdaptiveStep and runs additional epochs
    # input file_prefix: prefix of save file to start from
    # input epochs: number of epochs to run. If None, will run until cluster terminates job.
    with np.load(file_prefix+'.params.npz') as sim_data:
        params_dict = sim_data['data'].item()

    #remove non-parameter keys
    params_dict['K'] = params_dict['K0']
    nonparams = ['sim_start_time','sim_start_process_time','K0','N']
    for nonparam in nonparams:
        params_dict.pop(nonparam)

    #load last file
    ind = 0
    while True:
        file = file_prefix+'.{}.npz'.format(ind)
        if os.path.isfile(file):
            ind+=1
        else:
            last_ind = ind-1
            break
    file = file_prefix + '.{}.npz'.format(last_ind)
    data = np.load(file)['data'].item()
    species_idx = data['starting_species_idx_list'][-1]
    surviving_bool = data['surviving_bool_list'][-1]
    starting_species_idx = species_idx[surviving_bool]
    params_dict['init_species_idx'] = starting_species_idx

    # get epoch num
    epoch_num = int((last_ind+1) * params_dict['new_save_epoch_num'])
    params_dict['start_epoch_num'] = epoch_num

    K = len(starting_species_idx)
    V = data['V_end'][surviving_bool,:][:,surviving_bool]
    params_dict['V_init'] = V
    params_dict['S_init'] = data['S_list'][-1][:,surviving_bool]

    new_species_idx = max(starting_species_idx)+1
    params_dict['new_species_idx'] = new_species_idx

    #get list of successful invasion eigs
    invasion_success_eigs = []
    ind = last_ind
    while len(invasion_success_eigs)<params_dict['invasion_criteria_memory']:
        file = file_prefix + '.{}.npz'.format(last_ind)
        with np.load(file) as sim_data:
            data = sim_data['data'].item()
            invasion_eigs = np.array(data['invasion_eigs_list']).flatten()
            success_bool = np.array(data['invasion_success_list']).flatten()
            invasion_success_eigs.insert(0,invasion_eigs[success_bool])
        ind -= 1
        if ind<0:
            break
    params_dict['invasion_eigs_init'] = invasion_success_eigs

    if epochs is None:
        params_dict['epoch_num'] = np.inf
    else:
        params_dict['epoch_num'] = int(epochs)
    params_dict['file_name'] = file_prefix
    Evo = IslandsEvoAdaptiveStep(**params_dict)
    Evo.extend_simulation()


def combine_save_files(file_prefix):
    # Combines data from all save files with the same file_prefix.

    file_name = file_prefix + '.params.npz'
    with np.load(file_name) as sim_data:
        data = sim_data['data'].item()
        data_dict = data

    file_exists = True
    ind = 0
    while file_exists:
        file_name = file_prefix + '.{}.npz'.format(ind)
        if not os.path.isfile(file_name):
            break

        with np.load(file_name) as sim_data:
            data = sim_data['data'].item()

        if ind == 0:
            for key in data.keys():
                data_dict[key] = data[key]
        else:
            for key in data.keys():
                # if key is 'V_dict':
                #     break
                value = data[key]
                if type(value) is list:
                    data_dict[key].extend(value)
                elif type(value) is dict:
                    data_dict[key].update(value)
        ind += 1

    data_dict.pop('V_dict',None)
    data_dict.pop('n_init_list',None)
    file_name = file_prefix
    np.savez(file_name, data = data_dict)


#OLDER CODE FOR BACTERIA PHAGE simulations
class bp_evo:
    # Simulation of bacteria-phage evolution with many islands with random invasion of new types.
    # Mutations come in slowly, spaced out in epochs of O((K**(0.5))*M) in natural time units
    # Corresponds to long time between mutations - bursts likely to happen sooner than new mutants coming in
    # Pre-computes interaction matrix and works with appropriate slices.
    # Saves data after each epoch: island-averages, extinction times, correlation function, eta over each epoch

    # Dynamical equations: db/dt = b(Hp-\stabilizing_term_b)b
    #                      dp/dt = p(Gb-stabilizing_term_p)p

    # Frequency normalization
    # Time normalization: simulation is in "natural normalization". Short timescale is dt = K**(1/2)*M**(-1/2).
    # Long timescale is t = K**(1/2)*(temperature)

    def __init__(self, file_name, D, B, P, F, M_b, M_p, omega_b, omega_p, thresh_b, thresh_p, inv_fac_b, inv_fac_p, dt, mu_b, mu_p,
                 D_b, D_p, S, seed, epoch_timescale, epoch_num, sample_num):

        # input file_name: name of save file
        # input D: number of islands
        # input B: initial bacterial types
        # input P: initial phage types
        # input F: phenotype space dimension
        # input M_b: log(1/m_b) where m_b = migration rate
        # input M_p: log(1/m_p) where m_p = migration rate
        # input omega_b: timescale for bacterial dynamics
        # input omega_p: timescale for phage dynamics
        # input thresh_b: extinction threshold for log-frequency equal to log(1/N_b) for total population N_b.
        # input thresh_p: extinction threshold for log-frequency equal to log(1/N_p) for total population N_p.
        # input inv_fac_b: invasion factor, bacteria. Invading types start at number inv_fac*exp(thresh)
        # input inv_fac_p: invasion factor, phage. Invading types start at number inv_fac*exp(thresh)
        # input dt: rescaled step size, typically choose 0.1
        # input mu_b: number of bacteria invading per epoch
        # input mu_p: number of phage invading per epoch
        # input D_b: diffusion constant/matrix for mutations in phenotype space
        # input D_p: diffusion constant/matrix for mutations in phenotype space
        # input S: function that takes two phenotype vectors and gives interaction pair G, H. G is P x B matrix, \
        # H is B x P matrix

        # input seed: random seed
        # input epoch_timescale: time for each epoch, units of M*K**0.5
        # input epoch_num: number of epochs
        # input sample_num: sampling period

        self.file_name = file_name
        self.D = D
        self.B = B
        self.P = P
        self.F = F
        self.M_b = M_b
        self.M_p = M_p
        self.omega_b = omega_b
        self.omega_p = omega_p
        self.thresh_b = thresh_b
        self.thresh_p = thresh_p
        self.inv_fac_b = inv_fac_b
        self.inv_fac_p = inv_fac_p
        self.epoch_timescale = epoch_timescale
        self.epoch_num = epoch_num
        self.dt = dt
        self.mu_b = mu_b
        self.mu_p = mu_p
        self.D_b = D_b
        self.D_p = D_p
        self.S = S
        self.seed = seed
        self.sample_num = sample_num

        self.SetParams()

        self.EvoSim()

    def SetParams(self):


        self.N = 1
        self.m_b = np.exp(-self.M_b)
        self.m_p = np.exp(-self.M_p)

        self.B_tot = self.B + self.epoch_num * self.mu_b  # total number of types through simulation
        self.P_tot = self.P + self.epoch_num * self.mu_p  # total number of types through simulation


        self.vs = np.zeros((self.F,self.B_tot))
        self.ws = np.zeros((self.F,self.P_tot))
        np.random.seed(seed=self.seed)


        self.increment = 0.01 * min(self.M_b,self.M_p)  # increment for histogram

    def EvoSim(self):

        # Run evolutionary dynamics and save data


        self.dt_list = [] # list of dt for each epoch
        self.epoch_time_list = [] # list of total times for each epoch

        self.stabilizing_term_b_mean_ave_list = []  # <\stabilizing_term> over epoch
        self.stabilizing_term_b_mean_std_list = []

        self.stabilizing_term_p_mean_ave_list = []  # <\stabilizing_term> over epoch
        self.stabilizing_term_p_mean_std_list = []

        self.b_init_list = [] # initial distribution of n
        self.b_mean_ave_list = [] # <n> over epoch
        self.b2_mean_ave_list = [] # <n^2> over epoch
        self.b_mig_mean_list = []  # ratio nbar/n
        self.eta_b_list = []  # eta computed from V, <n>

        self.b_mean_std_list = []
        self.b2_mean_std_list = []

        self.b_extinct_time_array = np.inf * np.ones((self.B_tot))  # extinction times
        self.b_alive = np.zeros((self.B_tot, self.epoch_num + 2), dtype=bool)  # K x (epoch_num+2) matrix of which \
        # types are alive at a given time


        self.p_init_list = []  # initial distribution of n
        self.p_mean_ave_list = []  # <n> over epoch
        self.p2_mean_ave_list = []  # <n^2> over epoch
        self.p_mig_mean_list = []  # ratio nbar/n
        self.eta_p_list = []  # eta computed from V, <n>

        self.p_mean_std_list = []
        self.p2_mean_std_list = []

        self.p_extinct_time_array = np.inf * np.ones((self.P_tot))  # extinction times
        self.p_alive = np.zeros((self.P_tot, self.epoch_num + 2), dtype=bool)  # K x (epoch_num+2) matrix of which \
        # types are alive at a given time

        # initial setup

        # Generate phenotypes
        self.vs[:,0:self.B], self.ws[:,0:self.P] = self.gen_random_phenos(self.B,self.P)
        # generate interactions
        G,H = self.int_from_pheno(self.vs[:,0:self.B],self.ws[:,0:self.P])
        self.b_alive[0:self.B,0] = True
        self.p_alive[0:self.P,0] = True

        # index of next new type
        b_idx = self.B
        p_idx = self.P

        # Initialization of bp frequencies
        b0, p0 = initialization_many_islands_bp(self.D, self.B, self.P, self.m_b, self.m_p, 'flat')


        # run first epoch to equilibrate
        b0, p0, b_traj_eq, p_traj_eq,G,H = self.evo_step(G,H,b0,p0,1) # equilibration dynamics
        self.b_traj_eq = b_traj_eq  # save first trajectory for debugging purposes
        self.p_traj_eq = p_traj_eq  # save first trajectory for debugging purposes


        # evolution
        for i in range(2,self.epoch_num+2):
            G,H,b0,p0,b_idx,p_idx = self.mut_step(G,H,b_idx,p_idx,b0,p0,i) # add new types
            b0,p0,b_traj_f,p_traj_f,G,H = self.evo_step(G,H,b0,p0,i) # dynamics

            # save data

            if i % 10 == 0:  #save data every 10 epochs
                class_dict = vars(self)

                np.savez(self.file_name, data = class_dict)

        # save last trajectory for debugging purposes
        self.b_traj_f = b_traj_f
        self.p_traj_f = p_traj_f

        # save data
        class_dict = vars(self)

        np.savez(self.file_name, data = class_dict)
        

    def evo_step(self,G,H,b0,p0,cur_epoch):
        # Runs one epoch for the linear abundances. Returns bf and pf (abundances for next timestep) and b_traj and p_traj (sampled
        # trajectories for a single island)
        # b: (D,B) abundances
        # p: (D,P) abundances

        B = np.shape(b0)[1]
        P = np.shape(p0)[1]
        D = self.D
        m_b = self.m_b
        m_p = self.m_p

        thresh_b = self.thresh_b
        thresh_p = self.thresh_p
        M_b = self.M_b
        M_p = self.M_p


        # rescale time to appropriate fraction of short timescale
        short_timescale_b = M_p**(-1/2)*P**(1/2)*self.omega_b**(-1) 
        short_timescale_p = M_b**(-1/2)*B**(1/2)*self.omega_p**(-1) 
        dt = self.dt * min(short_timescale_b, short_timescale_p)
        self.dt_list.append(dt)

        # epoch time
        long_timescale_b = M_b**(3/2)*M_p**(-1/2)*P**(1/2)*self.omega_b**(-1) 
        long_timescale_p = M_p**(3/2)*M_b**(-1/2)*B**(1/2)*self.omega_p**(-1) 
        epoch_time = self.epoch_timescale*max(long_timescale_b,long_timescale_p)  # epoch_timescale*(long timescale)
        epoch_steps = int(epoch_time / dt)
        epoch_time = epoch_steps * dt
        self.epoch_time_list.append(epoch_time)  # save amount of time for current epoch
        t0 = np.sum(self.epoch_time_list[0:(cur_epoch-1)])  # initial time, used for calculation of extinction time

        # self.sample_time_short = 1
        sample_time = dt * self.sample_num

        Normalize = self.Normalize
        Extinction = self.Extinction
        increment = self.increment


        # set up for dynamics

        deriv = define_deriv_many_islands_bp(G,H)
        step_forward = step_rk4_many_islands_bp

        b_alive = self.b_alive[:,cur_epoch-1]
        p_alive = self.p_alive[:,cur_epoch-1]
        b_indices = np.arange(self.B_tot)[b_alive]
        p_indices = np.arange(self.P_tot)[p_alive]  # list of current species indices

        current_time = 0


        #### initialize for epoch
        count_short = 0


        b_mean_array = np.zeros((D, B))
        b2_mean_array = np.zeros((D, B))
        b_stabilizing_term_mean_array = np.zeros((D))
        b_mig_mean_array = np.zeros((B))

        p_mean_array = np.zeros((D, P))
        p2_mean_array = np.zeros((D, P))
        p_stabilizing_term_mean_array = np.zeros((D))
        p_mig_mean_array = np.zeros((P))


        self.b_init_list.append(b0)
        self.p_init_list.append(p0)

        sample_steps = int(epoch_steps / self.sample_num) + 1
        b_traj = np.zeros((B, sample_steps))
        p_traj = np.zeros((P, sample_steps))

        # dynamics
        for step in range(epoch_steps):

            ##### Save values each dt = sample_time
            if step % self.sample_num == 0:
                ind = int(step // self.sample_num)
                count_short += 1

                b_traj[:, ind] = b0[0, :]
                p_traj[:, ind] = p0[0, :]

                b_mean_array += b0
                b2_mean_array += b0 ** 2
                bbar = np.mean(b0, axis=0).reshape((1, -1))  # island averaged abundance
                temp_rats = np.divide(bbar, b0)  # remove infinities
                temp_rats[~(np.isfinite(temp_rats))] = 0
                b_mig_mean_array += np.mean(temp_rats, axis=0)
                b_stabilizing_term_mean_array += np.einsum('di,ij,dj->d', b0, H, p0)

                p_mean_array += p0
                p2_mean_array += p0 ** 2
                pbar = np.mean(p0, axis=0).reshape((1, -1))  # island averaged abundance
                temp_rats = np.divide(pbar, p0)  # remove infinities
                temp_rats[~(np.isfinite(temp_rats))] = 0
                p_mig_mean_array += np.mean(temp_rats, axis=0)
                p_stabilizing_term_mean_array += np.einsum('di,ij,dj->d', p0, G, b0)

                
            ######### Step abundances forward
            b1, p1 = step_forward(b0,p0,m_b,m_p,dt,deriv)
            b1 = Extinction(b1,thresh_b)
            p1 = Extinction(p1,thresh_p)
            b1 = Normalize(b1)
            p1 = Normalize(p1)

            ######### If extinctions occur, record ext time.
            b_extinct_bool = np.all(b1==0,axis=0)
            b_new_extinct = np.logical_and(b_extinct_bool, self.b_extinct_time_array[b_indices] == np.inf)
            if np.any(b_new_extinct):
                b_new_extinct_indices = b_indices[b_new_extinct]
                self.b_extinct_time_array[b_new_extinct_indices] = current_time+t0
                b_new_extinct_bool = True

            p_extinct_bool = np.all(p1==0,axis=0)
            p_new_extinct = np.logical_and(p_extinct_bool, self.p_extinct_time_array[p_indices] == np.inf)
            if np.any(p_new_extinct):
                p_new_extinct_indices = p_indices[p_new_extinct]
                self.p_extinct_time_array[p_new_extinct_indices] = current_time+t0
                p_new_extinct_bool = True

            ######### Prep for next time step
            b0 = b1
            p0 = p1

            current_time += dt
            ######### end epoch time steps

        ####### Compute averages
        b_mean_array *= 1 / count_short
        b2_mean_array *= 1 / count_short
        b_mig_mean_array *= 1 / count_short
        b_stabilizing_term_mean_array *= 1 / count_short

        p_mean_array *= 1 / count_short
        p2_mean_array *= 1 / count_short
        p_mig_mean_array *= 1 / count_short
        p_stabilizing_term_mean_array *= 1 / count_short


        # Average and standard dev across islands.
        b_mean_ave = np.mean(b_mean_array, axis=0)
        b2_mean_ave = np.mean(b_mean_array, axis=0)
        b_mig_mean_ave = b_mig_mean_array
        b_stabilizing_term_mean_ave = np.mean(b_stabilizing_term_mean_array, axis=0)

        p_mean_ave = np.mean(p_mean_array, axis=0)
        p2_mean_ave = np.mean(p_mean_array, axis=0)
        p_mig_mean_ave = p_mig_mean_array
        p_stabilizing_term_mean_ave = np.mean(p_stabilizing_term_mean_array, axis=0)

        b_mean_std = np.std(b_mean_array, axis=0)
        b2_mean_std = np.std(b_mean_array, axis=0)
        b_stabilizing_term_mean_std = np.std(b_stabilizing_term_mean_array, axis=0)

        p_mean_std = np.std(p_mean_array, axis=0)
        p2_mean_std = np.std(p_mean_array, axis=0)
        p_stabilizing_term_mean_std = np.std(p_stabilizing_term_mean_array, axis=0)

        # compute estimate of etas, from mean field calculations. Assuming close to antisymmetric.

        # sig_V = np.sqrt(np.var(V))  # standard deviation of interaction matrix
        # K_surv = np.sum(surviving_bool)
        # chi = sig_V * np.sqrt(K_surv)  # numerical estimate of chi
        # eta_mean_ave = -self.gamma * chi * n_mean_ave + np.dot(V, n_mean_ave) - m * (mig_mean_ave - 1)


        # Compute estimate of eta.

        # numerical estimate of gamma
        sig_mat = np.cov(G.flatten(),H.T.flatten())
        gamma = sig_mat[0,1]/np.sqrt(sig_mat[0,0]*sig_mat[1,1])

        interactions = np.dot(H,p_mean_ave)
        n_mean = b_mean_ave
        chi0 = np.mean(interactions)/(gamma*np.mean(n_mean))
        chi = chi0 
        diff = 10
        while diff>0.001/np.sqrt(P):   # loop that adds correction to chi to get mean bias for bias>0 and bias<0 to be same magnitude
            bias = interactions - gamma*chi*n_mean
            
            bias_plus = np.mean(bias[bias>0])
            bias_minus = np.mean(bias[bias<0])
            n_plus = np.mean(n_mean[bias>0])
            n_minus = np.mean(n_mean[bias<0])
            chi = chi + (bias_plus + bias_minus)/(gamma*(n_plus + n_minus))  #correction
            diff = np.abs(bias_plus+bias_minus)

        eta_b = bias

        interactions = np.dot(G,b_mean_ave)
        n_mean = p_mean_ave
        chi0 = np.mean(interactions)/(gamma*np.mean(n_mean))
        chi = chi0 
        diff = 10
        while diff>0.001/np.sqrt(B):  
            bias = interactions - gamma*chi*n_mean - np.mean(G)
            
            bias_plus = np.mean(bias[bias>0])
            bias_minus = np.mean(bias[bias<0])
            n_plus = np.mean(n_mean[bias>0])
            n_minus = np.mean(n_mean[bias<0])
            chi = chi + (bias_plus + bias_minus)/(gamma*(n_plus + n_minus))
            diff = np.abs(bias_plus+bias_minus)

        eta_p = bias


        #self.Calculate(n_traj)

        ######## Save data

        self.b_mean_ave_list.append(b_mean_ave)
        self.b2_mean_ave_list.append(b2_mean_ave)
        self.b_mig_mean_list.append(b_mig_mean_ave)
        self.eta_b_list.append(eta_b)
        self.stabilizing_term_b_mean_ave_list.append(b_stabilizing_term_mean_ave)

        self.b_mean_std_list.append(b_mean_std)
        self.b2_mean_std_list.append(b2_mean_std)
        self.stabilizing_term_b_mean_std_list.append(b_stabilizing_term_mean_std)

        self.p_mean_ave_list.append(p_mean_ave)
        self.p2_mean_ave_list.append(p2_mean_ave)
        self.p_mig_mean_list.append(p_mig_mean_ave)
        self.eta_p_list.append(eta_p)
        self.stabilizing_term_p_mean_ave_list.append(p_stabilizing_term_mean_ave)

        self.p_mean_std_list.append(p_mean_std)
        self.p2_mean_std_list.append(p2_mean_std)
        self.stabilizing_term_p_mean_std_list.append(p_stabilizing_term_mean_std)



        # starting frequencies for next step
        bf = b0
        pf = p0

        ####### Change number of surviving species.
        surviving_b = np.all(bf>0,axis=0)  # surviving species out of K1 current species.
        surviving_p = np.all(pf>0,axis=0)
        b_indices = b_indices[surviving_b]
        p_indices = p_indices[surviving_p]


        self.b_alive[b_indices,cur_epoch] = True
        self.p_alive[p_indices,cur_epoch] = True

        # only pass surviving types to next timestep
        bf = bf[:,surviving_b]
        pf = pf[:,surviving_p]

        print(cur_epoch)
        ######## end of current epoch
        return bf, pf, b_traj, p_traj,G[np.ix_(surviving_p,surviving_b)],H[np.ix_(surviving_b,surviving_p)]

    def mut_step(self,G,H,b_idx,p_idx,b0,p0,cur_epoch):
        """
        Generate new mutants and update list of alive types accordingly
        :param n0: Current distribution of types (DxK matrix)
        :param cur_epoch: Current epoch
        :return: V - current interaction matrix, n0_new - distribution of types with new mutants
        """
        D = self.D
        B = np.sum(self.b_alive[:,cur_epoch-1])
        P = np.sum(self.p_alive[:,cur_epoch-1])

        # generate new interaction matrix
        G_new,H_new = self.mut_int_mat(G,H,b_idx,p_idx,cur_epoch,b0,p0)
        # initialize invading types

        # bacteria
        b0_new = np.zeros((D, B + self.mu_b))
        b0_new[:, 0:B] = b0  # old types
        b0_new[:, B:] = self.inv_fac_b * np.exp(self.thresh_b)  # new types
        for i in range(D): # normalize
            b0_new[i,:] = b0_new[i,:]/np.sum(b0_new[i,:])
        # phage
        p0_new = np.zeros((D, P + self.mu_p))
        p0_new[:, 0:P] = p0  # old types
        p0_new[:, P:] = self.inv_fac_p * np.exp(self.thresh_p)  # new types
        for i in range(D):  # normalize
            p0_new[i, :] = p0_new[i, :] / np.sum(p0_new[i, :])

        # update alive types
        self.b_alive[b_idx:(b_idx+self.mu_b),cur_epoch-1] = True
        self.p_alive[p_idx:(p_idx + self.mu_p), cur_epoch - 1] = True

        # update index for new mutation
        b_idx_new = b_idx+self.mu_b
        p_idx_new = p_idx+self.mu_p

        return G_new,H_new,b0_new,p0_new,b_idx_new,p_idx_new

    def mut_int_mat(self,G,H,b_idx,p_idx,cur_epoch,b0,p0):
        # get dimensions
        P = np.shape(G)[0]
        B = np.shape(G)[1]
        P_new = P+self.mu_p
        B_new = B+self.mu_b

        # allocate new interaction matrix
        G_new = np.zeros((P_new,B_new))
        G_new[0:P,0:B] = G

        H_new = np.zeros((B_new,P_new))
        H_new[0:B, 0:P] = H

        # new phenotypes
        self.gen_new_phenos(b_idx,p_idx,cur_epoch,b0,p0)

        # new interaction matrices
        p_alive = self.p_alive[:, cur_epoch-1]
        b_alive = self.b_alive[:, cur_epoch-1]


        new_vs = self.vs[:, b_idx:(b_idx + self.mu_b)]
        old_vs = self.vs[:, b_alive]

        new_ws = self.ws[:,p_idx:(p_idx+self.mu_p)]
        old_ws = self.ws[:,p_alive]

        G_new[P:,0:B],H_new[0:B,P:] = self.int_from_pheno(old_vs,new_ws) # new phage old bact
        G_new[0:P,B:],H_new[B:,0:P] = self.int_from_pheno(new_vs,old_ws) # new bact old phage
        G_new[P:,B:],H_new[B:,P:] = self.int_from_pheno(new_vs,new_ws) # new bact new phage

        return G_new,H_new

    def gen_random_phenos(self,B,P):
        F = self.F
        vs = np.random.randn(F,B)
        ws = np.random.randn(F,P)

        for i in range(B):
            vs[:,i] = vs[:,i]/np.linalg.norm(vs[:,i])

        for j in range(P):
            ws[:, j] = ws[:, j] / np.linalg.norm(ws[:, j])

        return vs, ws

    def gen_new_phenos(self,b_idx,p_idx,cur_epoch,b0,p0):

        F = self.F
        D_b = self.D_b
        D_p = self.D_p

        b_bar = np.mean(b0, axis=0)
        p_bar = np.mean(p0, axis=0)

        b_alive = self.b_alive[:,cur_epoch-1]
        b_types = np.arange(len(b_alive))[b_alive] # currently surviving bacterial indices

        p_alive = self.p_alive[:, cur_epoch - 1]
        p_types = np.arange(len(p_alive))[p_alive]  # currently surviving phage indices

        # bacterial mutation
        if self.D_b==None:
            # invasion by random types
            self.vs[:,b_idx:(b_idx+self.mu_b)], _ = self.gen_random_phenos(self.mu_b,0)
        else:
            for i in range(self.mu_b):
                # pick a bacteria to mutate
                b_mut = np.random.multinomial(1, b_bar)
                # generate mutants
                for k,I in enumerate(b_mut):
                    if I!=0:
                        # initialize with parent's phenotype vector
                        self.vs[:,b_idx+i] = self.vs[:,b_types[k]]
                        # add perturbation
                        if np.shape(D_b)==():
                            self.vs[:,b_idx+i] += np.dot(np.sqrt(D_b), np.random.normal(F))/np.sqrt(F)
                        else:
                            self.vs[:,b_idx+i] += np.real(np.dot(np.linalg.sqrtm(D_b),
                                                                 np.random.normal(F)))/np.sqrt(F)
                        # normalize
                        self.vs[:,b_idx+i] = self.vs[:,b_idx+i]/np.linalg.norm(self.vs[:,b_idx+i])
                        break

        # phage mutation
        if self.D_p == None:
            # invasion by random types
            _, self.ws[:, p_idx:(p_idx + self.mu_p)] = self.gen_random_phenos(0,self.mu_p)
        else:
            for i in range(self.mu_p):
                # pick a bacteria to mutate
                p_mut = np.random.multinomial(1, p_bar)
                # generate mutants
                for k, I in enumerate(p_mut):
                    if I != 0:
                        # initialize with parent's phenotype vector
                        self.ws[:, p_idx + i] = self.ws[:, p_types[k]]
                        # add perturbation
                        if np.shape(D_p) == ():
                            self.ws[:, p_idx + i] += np.dot(np.sqrt(D_p),
                                                            np.random.normal(F)) / np.sqrt(F)
                        else:
                            self.ps[:, p_idx + i] += np.real(np.dot(np.linalg.sqrtm(D_p),
                                                                    np.random.normal(F))) / np.sqrt(F)
                        # normalize
                        self.ws[:, p_idx + i] = self.ws[:, p_idx + i] / np.linalg.norm(self.ws[:, p_idx + i])
                        break

    def int_from_pheno(self,vs,ws):
        B = np.shape(vs)[1]
        P = np.shape(ws)[1]
        G = np.zeros((P,B))
        H = np.zeros((B,P))

        for i in range(B):
            for j in range(P):
                G[j,i], H[i,j] = self.S(vs[:,i],ws[:,j])

        return G,H




    def Calculate(self, n_traj):
        # Compute correlation function and add to autocorr_list

        rows, cols = np.shape(n_traj)
        # autocorr_list = np.zeros((rows,cols))
        autocorr_sum = np.zeros((cols))
        nonzero_num = 0

        for ii in range(rows):
            n = self.K * n_traj[ii, :]  # convert to conventional normalization
            length = len(n)
            timepoint_num_vec = length - np.abs(
                np.arange(-length // 2, length // 2))  # number of time points used to evaluate autocorrelation
            autocorr = scipy.signal.fftconvolve(n, n[::-1], mode='same') / timepoint_num_vec
            autocorr_sum = autocorr_sum + autocorr
            nonzero_num = nonzero_num + 1
            # autocorr_list[step,:] = autocorr

        corr_time_vec = self.sample_time * np.arange(-length // 2, length // 2)
        corr_window = 2 * self.K ** (3 / 2)
        window = np.logical_and(corr_time_vec > -corr_window, corr_time_vec < corr_window)

        # autocorr_list = autocorr_list[:,window]
        autocorr_sum = autocorr_sum[window]
        corr_tvec = corr_time_vec[window]

        self.corr_tvec = np.float32(corr_tvec)
        self.autocorr_list.append(autocorr_sum)  # 1/K comes from normalization of V ~ 1/sqrt(K)

    def Normalize(self, n):
        Nhat = np.sum(n, axis=1, keepdims=True)
        n1 = n/Nhat

        return n1

    def Extinction(self, n, thresh):
        x = np.log(n)

        local_ext_ind = x < thresh
        n[local_ext_ind] = 0

        global_ext_ind = np.all(n == 0, axis=0)
        n[:, global_ext_ind] = 0

        return n


#######################################################


def initialization_many_islands(D,K,N,m,mode):
    #:input K: number of species
    #:input N: normalization of abundances
    #:input D: number of demes
    #:input mode: type of initialization
    #:output x: drawn from normal distribution with mean N/K and var 
    
    if mode=='equal':
        nbar_temp=N/K
        nhat = np.random.exponential(nbar_temp,(D,K))
        Nhat = np.sum(nhat,axis=1,keepdims=1)
        n = N*nhat/Nhat
    elif mode=='flat':
        nbar = N/K
        random_vars = np.random.rand(D,K)
        xhat = (np.log(N) - np.log(m*nbar))*random_vars + np.log(m*nbar)
        nhat = np.exp(xhat)
        Nhat = np.sum(nhat,axis=1,keepdims=1)
        n = nhat/Nhat

    return n

def initialization_many_islands_bp(D, B, P, m_b, m_p, mode='flat'):
    #:input D: number of islands
    # input B: number of bacteria types
    # input P: number of phage types
    # input m_b: migration rate for bacteria
    # input m_p: migration rate for phage
    # input mode: type of initialization
    # output b, p: bacteria and phage starting frequencies
    
    if mode=='equal':
        b = np.random.exponential(1/B,(D,B))
        
        p = np.random.exponential(1/P,(D,P))
        
    elif mode=='flat':
        x_min = np.log(m_b/B)
        x_max = 0
        random_vars = np.random.rand(D,B)
        x = x_min + (x_max- x_min)*random_vars
        b = np.exp(x)

        x_min = np.log(m_p/P)
        x_max = 0
        random_vars = np.random.rand(D,P)
        x = x_min + (x_max- x_min)*random_vars
        p = np.exp(x)
        
    b = b / np.sum(b,axis=1,keepdims=1)
    p = p / np.sum(p,axis=1,keepdims=1)

    return b, p

def generate_interactions_with_diagonal(K,gamma):
    # :param K: number of types
    # :param gamma: correlation parameter. Need gamma in [-1,1]
    # :output V: interaction matrix. Entries have correlation E[V_{i,j} V_{m,n}] = \delta_{i,m}\delta_{j,n} + \gamma \delta_{i,n} \delta_{j,m}

    V = np.zeros((K,K))
    upperInd = np.triu_indices(K,1)
    
    if gamma < 1 and gamma > -1:
        #generate list of correlated pairs.
        mean = [0,0]
        cov = [[1, gamma], [gamma, 1]]
        x = np.random.multivariate_normal(mean, cov, (K*(K-1)//2))

        #put correlated pairs into matrix
        V[upperInd] = x[:,0]
        V = np.transpose(V)
        V[upperInd] = x[:,1]

        diag = np.random.normal(scale=np.sqrt(1+gamma),size=K)
        V = V + np.diag(diag)
    
    elif gamma == 1 or gamma == -1:
        x = np.random.normal(0,1,size=(K*(K-1)//2))

        V[np.triu_indices(K,1)] = x
        V = np.transpose(V)
        V[np.triu_indices(K,1)] = gamma*x

        diag = np.random.normal(scale=np.sqrt(1+gamma),size=K)
        V = V + np.diag(diag)

    else:
        print('Correlation parameter invalid')
    
    return V

def define_deriv_many_islands_bp(G,H):
    # :input G, H: interaction matrices. G is P x B matrix, H is B x P matrix
    # :input m_b, m_p: migration rate for bacteria and phage
    # :output deriv: function of x that computes derivative

    def deriv(b,p,m_b,m_p):
        # input b,p: frequencies of bacteria and phage
        D = np.shape(b)[0]
        growth_rate_b = np.einsum('ij,dj',H,p)
        growth_rate_p = np.einsum('ij,dj',G,b)
        
        norm_factor_b = np.einsum('di,di->d',b,growth_rate_b) #normalization factor
        norm_factor_b = np.reshape(norm_factor_b,(D,1))

        norm_factor_p = np.einsum('di,di->d',p,growth_rate_p) #normalization factor
        norm_factor_p = np.reshape(norm_factor_p,(D,1))
        
        b_island_ave = np.mean(b,axis=0,keepdims=True)
        p_island_ave = np.mean(p,axis=0,keepdims=True)

        if m_b==0:
            b_dot = b*(growth_rate_b - norm_factor_b)
        else:
            b_dot = b*(growth_rate_b - norm_factor_b) + m_b*(b_island_ave - b)
        if m_p==0:
            p_dot = p * (growth_rate_p - norm_factor_p)
        else:
            p_dot = p * (growth_rate_p - norm_factor_p) + m_p * (p_island_ave - p)
        return b_dot, p_dot

    return deriv


def step_rk4_many_islands_bp(b0,p0,m_b,m_p,dt,deriv):
    # :input x0: (K,) vec of current log variables
    # :param dt: step size
    # :function deriv: gives derivative of x. Function of x.
    # :output x1: (K,) vec of log variables after one time step.
    

    k1_b, k1_p = deriv(b0,p0,m_b,m_p)
    k2_b, k2_p = deriv(b0+(dt/2)*k1_b,p0+(dt/2)*k1_p,m_b,m_p)
    k3_b, k3_p = deriv(b0+(dt/2)*k2_b,p0+(dt/2)*k2_p,m_b,m_p)
    k4_b, k4_p = deriv(b0+dt*k3_b,p0+dt*k3_p,m_b,m_p)
    
    b1 = b0 + (dt/6)*(k1_b + 2*k2_b + 2*k3_b+k4_b)
    p1 = p0 + (dt/6)*(k1_p + 2*k2_p + 2*k3_p+k4_p)
    
    return b1, p1

def lv_to_lp(A):
    """
    Generate parameters for lp to find fixed points
    :param A: antisymmetric matrix
    :return c, A_ub, b_ub, A_eq, b_eq: linear program parameters for linprog from scipy.optimize.
    """
    N = np.shape(A)[0]
    # augment value v of "game" to end of variables. Trying to minimize v.
    c = np.zeros(N + 1)
    c[-1] = 1

    # inequality constraints

    A_ub = np.zeros((N, N + 1))
    b_ub = np.zeros(N)

    # max fitness bounded by v

    A_ub[:, :-1] = A
    A_ub[:, -1] = -1

    # equality constraint: probabilities sum to 1
    A_eq = np.ones((1, N + 1))
    A_eq[0, -1] = 0
    b_eq = np.ones(1)

    return c, A_ub, b_ub, A_eq, b_eq

def find_fixed_pt(A):
    """
    Find saturated fixed point of antisymmetric LV model using linear programming
    :param A: Interaction matrix
    :return: saturated fixed point
    """
    N = np.shape(A)[0]
    maxiter = int(np.max(((N**2)/25,1000)))
    c, A_ub, b_ub, A_eq, b_eq = lv_to_lp(A)
    result = opt.linprog(c,A_ub,b_ub,A_eq,b_eq,options={'maxiter':maxiter},method='interior-point')
    return result.x[:-1]

