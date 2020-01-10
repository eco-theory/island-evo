import numpy as np
import math
import scipy.signal
import time
import scipy.optimize as opt


class IslandsEvoAdaptiveStep:
    # Simulation with many islands with random invasion of new types.
    # Mutations come in slowly, spaced out in epochs of O((K**(0.5))*M) in natural time units
    # Corresponds to long time between mutations - bursts likely to happen sooner than new mutants coming in
    # Pre-computes interaction matrix and works with appropriate slices.
    # Saves data after each epoch: island-averages, extinction times, correlation function, eta over each epoch

    # Frequency normalization
    # Time normalization: simulation is in "natural normalization". Short timescale is dt = K**(1/2)*M**(-1/2).
    # Long timescale is t = K**(1/2)*(temperature)

    def __init__(self, file_name, D, K, m, gamma, thresh, mu, seed, epoch_timescale, epoch_num,
                 sample_num = 1, max_frac_change=0.75, invasion_freq_factor = 1, corr_mut=0, sig_S=0, n_init=None,
                 V_init = None, S_init = None, epochs_to_save_traj = None,
                 long_epochs=None,long_factor=1,invasion_criteria_memory=100,invasion_eig_buffer=0.1):

        # input file_name: name of save fil
        # input D: number of islands
        # input K: number of types
        # input m: migration rate
        # input gamma: symmetry parameter
        # input thresh: extinction threshold for log-frequency equal to log(1/N) for total population N.
        # input invasion_freq_factor: new invasions come in at freq invasion_freq_factor/K
        # input mu: number of types invading per epoch
        # input seed: random seed
        # input epoch_timescale: time for each epoch, units of M*K**0.5
        # input epoch_num: number of epochs
        # input sample_num: sampling period
        # input max_frac_change: the maximum fractional change in frequency allowed across islands. Sets adaptive step size.
        # input corr_mut: correlation of new types with previous ones
        # input sig_S: standard dev. of selective differences
        # input n_init: (D,K) array of frequencies to start from.
        # input V_init: (K,K) array of interactions to start with.
        # input S_init: (1,K) array of selective differences to start with.
        # input epochs_to_save_traj: list of ints with index of epoch to save trajectories.
        #       If -1 is included then the most current epoch is also saved.
        # input long_epochs: which epochs to make longer, does not stop early.
        # input long_factor: long epochs are longer by this factor.

        self.file_name = file_name
        self.D = D
        self.K0 = K
        self.m = m
        self.gamma = gamma
        self.N = 1  #frequency normalization
        self.thresh = thresh
        self.invasion_freq_factor = invasion_freq_factor
        self.epoch_timescale = epoch_timescale
        self.epoch_num = epoch_num
        self.mu = mu
        self.seed = seed
        self.sample_num = sample_num
        self.max_frac_change = max_frac_change
        self.sim_start_time = time.time()
        self.sim_start_process_time = time.process_time()
        self.corr_mut = corr_mut
        self.sig_S = sig_S
        if epochs_to_save_traj is None:
            self.epochs_to_save_traj = []
        else:
            self.epochs_to_save_traj = epochs_to_save_traj
        self.n_init = n_init
        self.V_init = V_init
        self.S_init = S_init
        if long_epochs is None:
            self.long_epochs = []
        else:
            self.long_epochs = long_epochs
        self.long_factor = long_factor
        self.invasion_criteria_memory = invasion_criteria_memory
        self.invasion_eig_buffer = invasion_eig_buffer

        if n_init is not None:
            self.K0 = n_init.shape[1]
            self.D = n_init.shape[0]

        self.EvoSim()

    def EvoSim(self):
        # Run evolutionary dynamics and save data

        self.epoch_time_list = [] # list of total times for each epoch
        self.n_init_list = [] # initial distribution of n
        SavedQuants = EvoSavedQuantitiesAdaptiveStep(self.sample_num,self.epochs_to_save_traj)

        self.surviving_bool_list = []
        self.starting_species_idx_list = []
        self.current_species_idx = np.arange(self.K0)
        self.parent_idx_dict = {}
        self.K_tot = self.K0  #current total number of species

        self.invasion_eigs_list = []
        self.invasion_rejects_list = []
        self.invasion_success_list = []

        np.random.seed(seed=self.seed)  # Set random seed
        self.initialize_interactions_abundances()  #Sets V, S, and n0. Define V_dict and S_full

        # evolution
        for cur_epoch in range(self.epoch_num+1):
            self.evo_step(cur_epoch, SavedQuants) # dynamics for one epoch. V, n0 are for non-extinct types
            self.mut_step(SavedQuants)  # add new types
            # Save data in case job terminates on cluster
            self.save_data(SavedQuants)

            num_types = self.n0.shape[1]
            if num_types <= 10+self.mu:
                break

            print(cur_epoch)


    def evo_step(self, cur_epoch, SavedQuants):
        # Runs one epoch for the linear abundances. Returns nf (abundances for next timestep) and n_traj (sampled
        # trajectories for a single island)
        # n: (D,K) abundances
        # xbar0: (1,K) log of island averaged abundances.

        V = self.V
        S = self.S
        n0 = self.n0

        K = np.shape(n0)[1]
        D = self.D
        m = self.m
        N = self.N
        thresh = self.thresh

        M = self.define_M(K,m)

        epoch_time, step_forward = self.setup_epoch(cur_epoch,K,M,V,S)

        Normalize = self.Normalize
        Extinction = self.Extinction
        check_new_types_extinct = self.check_new_types_extinct

        nbar = np.mean(n0, axis=0, keepdims=True)
        xbar0 = np.log(nbar) # Assumes all types have nbar>0
        y0 = n0 / nbar
        n0 = y0 * np.exp(xbar0)
        self.n_init_list.append(n0)
        self.starting_species_idx_list.append(self.current_species_idx)  #Add current species indices to list.

        # initialize for epoch
        SavedQuants.initialize_epoch(D,K,V,S)

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
        self.subset_to_surviving_species(y0,xbar0,cur_epoch)


    def initialize_interactions_abundances(self):
        if self.V_init is None:
            V = generate_interactions_with_diagonal(self.K0, self.gamma)
        else:
            V = self.V_init

        #Add interactions to dictionary
        V_dict = {}
        for ii in range(self.K0):
            for jj in range(self.K0):
                V_dict[ii,jj] = V[ii,jj]
        self.V_dict = V_dict

        if self.n_init is None:
            n0 = initialization_many_islands(self.D, self.K0, self.N, self.m, 'flat')
        else:
            n0 = self.n_init

        if self.S_init is None:
            S = self.sig_S*np.random.normal(size=(1,self.K0))
        else:
            S = self.S_init
        self.S_full = S

        self.n0 = n0
        self.V = V
        self.S = S

    def define_M(self,K,m):
        #Define log-migration parameter
        M = np.log(1/(m*np.sqrt(K)))
        return M

    def setup_epoch(self,cur_epoch,K,M,V,S):
        # rescale time to appropriate fraction of short timescale
        m = self.m
        N = self.N
        max_frac_change = self.max_frac_change

        # epoch time
        if cur_epoch == 0:
            epoch_timescale = 4*K**(-0.5)  # First epoch only for 4 * K**(0.5) * M
        elif cur_epoch in self.long_epochs:
            epoch_timescale = self.epoch_timescale * self.long_factor
        else:
            epoch_timescale = self.epoch_timescale
        epoch_time = epoch_timescale*(K**1)*M  # epoch_timescale*(long timescale for marginal types (bias ~ 1/sqrt(K)) )
        self.epoch_time_list.append(epoch_time)  # save amount of time for current epoch

        u = 0
        normed = True
        deriv = define_deriv_many_islands_selective_diffs(V, S, N, u, m, normed)
        def step_forward(y0,xbar0):
            y1, xbar1, dt, stabilizing_term = adaptive_step(y0, xbar0, m, deriv, max_frac_change)
            return y1, xbar1, dt, stabilizing_term

        return epoch_time, step_forward

    def subset_to_surviving_species(self,y0,xbar0,cur_epoch):
        surviving_bool = xbar0[0, :] > -np.inf  # surviving species out of K1 current species.
        self.surviving_bool_list.append(surviving_bool)
        self.current_species_idx = self.current_species_idx[surviving_bool]

        # only pass surviving types to next timestep
        n0 = np.exp(xbar0) * y0
        self.n0 = n0[:, surviving_bool]
        self.V = self.V[np.ix_(surviving_bool, surviving_bool)]
        self.S = self.S[:,surviving_bool]

        if cur_epoch > 0:
            invasion_success_bool = surviving_bool[-self.mu:]
            self.invasion_success_list.extend(invasion_success_bool)

    def mut_step(self,SavedQuants):
        """
        Generate new mutants and update list of alive types accordingly
        :param n0: Current distribution of types (DxK matrix)
        :return: V - current interaction matrix, n0_new - distribution of types with new mutants
        """
        V = self.V
        S = self.S
        n0 = self.n0

        mu = self.mu
        K = np.shape(V)[0]
        V_new, S_new, parent_idx_list = self.gen_new_invasions(V,S,SavedQuants) #generated interactions correlated with parent

        # set new invasion types
        n0_new = self.next_epoch_abundances(mu, n0)

        K_tot = self.K_tot
        species_idx = np.append(self.current_species_idx,np.arange(K_tot,K_tot+mu))
        self.current_species_idx = species_idx
        for idx in range(mu):
            self.parent_idx_dict[K_tot+idx] = parent_idx_list[idx]
        self.K_tot = K_tot + mu

        self.V = V_new
        self.S = S_new
        self.n0 = n0_new

        self.store_new_interactions()

    def next_epoch_abundances(self,mu,n0):
        D = self.D
        K = len(self.current_species_idx)
        n0_new = np.zeros((D,K+mu))
        n0_new[:,:K] = n0
        n0_new[:,K:] = self.invasion_freq_factor/K
        n0_new = n0_new/np.sum(n0_new,axis=1,keepdims=True)
        return n0_new

    def gen_new_invasions(self,V,S,SavedQuants):
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
                invade_bool = self.invasion_criteria(invasion_eig,K)
                invasion_counts += 1

            invasion_eigs.append(invasion_eig)
            parent_idx_list.append(int(self.current_species_idx[par_idx]))
            V_new[idx, 0:idx] = V_row
            V_new[0:idx, idx] = V_col
            V_new[idx, idx] = V_diag
            S_new[0, idx] = s

        self.invasion_rejects_list.append(invasion_counts - mu)
        self.invasion_eigs_list.extend(invasion_eigs)


        return V_new, S_new, parent_idx_list

    def gen_related_interactions(self,V_new,S_new,par_idx,idx):
        # generate new types from random parents
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

        s = corr_mut * S_new[0, par_idx] + np.sqrt(1 - corr_mut ** 2) * self.sig_S * np.random.normal()

        return V_row, V_col, V_diag, s

    def compute_invasion_eig(self,SavedQuants,row,s):
        # Computes invasion eigenvalue based on previous epochs mean abundances and lambda
        surviving_bool = self.surviving_bool_list[-1]
        n_mean = SavedQuants.n_mean_ave_list[-1]
        lambd = SavedQuants.lambda_mean_ave_list[-1]
        S_mean = SavedQuants.S_mean_ave_list[-1]
        invasion_eig = s + np.dot(row,n_mean[surviving_bool]) - S_mean - lambd
        return invasion_eig

    def invasion_criteria(self,invasion_eig,K):
        # find the minimum invasion eigenvalue of the last x successful invasions.
        # x is self.invasion_criteria_memory
        # set invade_bool to True if invasion_eig is within invasion_eig_buffer of the minimum.

        past_eigs = np.array(self.invasion_eigs_list)
        success_bool = np.array(self.invasion_success_list)==True
        success_eigs = past_eigs[success_bool]
        if len(success_eigs) > self.invasion_criteria_memory:
            min_eig = np.min(success_eigs[-self.invasion_criteria_memory:])
            invade_bool = invasion_eig > min_eig - self.invasion_eig_buffer*1/np.sqrt(K)
        else:
            invade_bool = True
        return invade_bool

    def store_new_interactions(self):
        V = self.V
        S = self.S
        mu = self.mu
        species_idx = self.current_species_idx
        K = V.shape[0]

        self.S_full = np.append(self.S_full,S[:,-mu:])

        V_dict = self.V_dict
        for new_idx in range(K-mu,K):
            for idx in range(K):
                V_dict[species_idx[idx],species_idx[new_idx]] = V[idx,new_idx]
                V_dict[species_idx[new_idx], species_idx[idx]] = V[new_idx,idx]
        self.V_dict = V_dict

    def save_data(self,SavedQuants):
        self.sim_end_time = time.time()
        self.sim_end_process_time = time.process_time()
        data = vars(self)
        data = SavedQuants.add_data_to_dict(data)
        np.savez(self.file_name, data = data)

    def Normalize(self, y, xbar, N):
        n = np.exp(xbar) * y
        Nhat = np.sum(n, axis=1, keepdims=True)
        Yhat = np.mean(y * N / Nhat, axis=0, keepdims=True)

        Yhat[Yhat == 0] = 1

        y1 = (y * N / Nhat) / Yhat
        xbar1 = xbar + np.log(Yhat)

        return y1, xbar1

    def Extinction(self, y, xbar, thresh):
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

    def __init__(self,sample_num,epochs_to_save_traj):
        self.sample_num = sample_num
        self.epochs_to_save_traj = epochs_to_save_traj

        # intialize lists to store values for each epoch
        self.n_mean_ave_list = []  # <n> over epoch
        self.n2_mean_ave_list = []  # <n^2> over epoch
        self.lambda_mean_ave_list = []  # <\lambda> over epoch
        self.n_mean_std_list = []
        self.n2_mean_std_list = []
        self.lambda_mean_std_list = []
        self.force_mean_ave_list = []
        self.force_mean_std_list = []
        self.invasion_species_idx = []
        self.invasion_eigs = []
        self.invasion_success = []
        self.S_mean_ave_list = []
        self.S_mean_std_list = []
        self.dt_mean_list = []
        self.dt2_mean_list = []

        self.n_traj_dict = {}
        self.time_vec_dict = {}

    def initialize_epoch(self,D,K,V,S):
        self.V = V
        self.S = S
        self.count = 0
        self.n_mean_array = np.zeros((D, K))
        self.n2_mean_array = np.zeros((D, K))
        self.lambda_mean_array = np.zeros((D))
        self.steps = 0
        self.dt_mean = 0
        self.dt2_mean = 0
        self.n_traj = []
        self.time_vec = []

    def save_sample(self,dt,time,y0,xbar0,stabilizing_term,cur_epoch):

        n0 = np.exp(xbar0) * y0
        self.n_mean_array += dt * n0
        self.n2_mean_array += dt * n0 ** 2
        self.lambda_mean_array += dt * stabilizing_term.flatten()
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
        self.lambda_mean_array *= 1 / time
        self.dt_mean *= 1/self.steps
        self.dt2_mean *= 1/self.steps

    def save_to_lists(self,cur_epoch,xbar,mu,species_idx):
        # input mu: number of mutants
        # input species_idx: current species idx

        n_mean_ave = np.mean(self.n_mean_array, axis=0)
        n2_mean_ave = np.mean(self.n2_mean_array, axis=0)
        lambda_mean_ave = np.mean(self.lambda_mean_array, axis=0)

        n_mean_std = np.std(self.n_mean_array, axis=0)
        n2_mean_std = np.std(self.n2_mean_array, axis=0)
        lambda_mean_std = np.std(self.lambda_mean_array, axis=0)

        force_mean = np.einsum('ij,dj',self.V,self.n_mean_array)
        force_mean_ave = np.mean(force_mean,axis=0)
        force_mean_std = np.std(force_mean, axis=0)

        S_mean = np.sum(self.S*self.n_mean_array,axis=1)
        S_mean_ave = np.mean(S_mean)
        S_mean_std = np.std(S_mean)

        self.n_mean_ave_list.append(n_mean_ave)
        self.n2_mean_ave_list.append(n2_mean_ave)
        self.lambda_mean_ave_list.append(lambda_mean_ave)
        self.force_mean_ave_list.append(force_mean_ave)
        self.S_mean_ave_list.append(S_mean_ave)

        self.n_mean_std_list.append(n_mean_std)
        self.n2_mean_std_list.append(n2_mean_std)
        self.lambda_mean_std_list.append(lambda_mean_std)
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


    def add_data_to_dict(self,data):
        var_dict = vars(self)
        keys = list(var_dict.keys())
        keys.remove('n_traj')
        keys.remove('V')
        keys.remove('S')

        for key in keys:
            data[key] = var_dict[key]

        return data

class IslandsEvo:
    # Simulation with many islands with random invasion of new types.
    # Mutations come in slowly, spaced out in epochs of O((K**(0.5))*M) in natural time units
    # Corresponds to long time between mutations - bursts likely to happen sooner than new mutants coming in
    # Pre-computes interaction matrix and works with appropriate slices.
    # Saves data after each epoch: island-averages, extinction times, correlation function, eta over each epoch

    # Frequency normalization
    # Time normalization: simulation is in "natural normalization". Short timescale is dt = K**(1/2)*M**(-1/2).
    # Long timescale is t = K**(1/2)*(temperature)

    def __init__(self, file_name, D, K, m, gamma, thresh, dt, mu, seed, epoch_timescale, epoch_num,
                 sample_num, invasion_freq_factor = 1, corr_mut=0, sig_S=0, n_init=None, V_init = None, S_init = None, epochs_to_save_traj = None,
                 long_epochs=None,long_factor=1,invasion_criteria_memory=100,invasion_eig_buffer=0.1):

        # input file_name: name of save file
        # input D: number of islands
        # input K: number of types
        # input m: migration rate
        # input gamma: symmetry parameter
        # input thresh: extinction threshold for log-frequency equal to log(1/N) for total population N.
        # input invasion_freq_factor: new invasions come in at freq invasion_freq_factor/K
        # input dt: rescaled step size, typically choose 0.1
        # input mu: number of types invading per epoch
        # input seed: random seed
        # input epoch_timescale: time for each epoch, units of M*K**0.5
        # input epoch_num: number of epochs
        # input sample_num: sampling period
        # input corr_mut: correlation of new types with previous ones
        # input sig_S: standard dev. of selective differences
        # input n_init: (D,K) array of frequencies to start from.
        # input V_init: (K,K) array of interactions to start with.
        # input S_init: (1,K) array of selective differences to start with.
        # input epochs_to_save_traj: list of ints with index of epoch to save trajectories.
        #       If -1 is included then the most current epoch is also saved.
        # input long_epochs: which epochs to make longer, does not stop early.
        # input long_factor: long epochs are longer by this factor.

        self.file_name = file_name
        self.D = D
        self.K0 = K
        self.m = m
        self.gamma = gamma
        self.N = 1  #frequency normalization
        self.thresh = thresh
        self.invasion_freq_factor = invasion_freq_factor
        self.epoch_timescale = epoch_timescale
        self.epoch_num = epoch_num
        self.dt = dt
        self.mu = mu
        self.seed = seed
        self.sample_num = sample_num
        self.sim_start_time = time.time()
        self.sim_start_process_time = time.process_time()
        self.corr_mut = corr_mut
        self.sig_S = sig_S
        if epochs_to_save_traj is None:
            self.epochs_to_save_traj = []
        else:
            self.epochs_to_save_traj = epochs_to_save_traj
        self.n_init = n_init
        self.V_init = V_init
        self.S_init = S_init
        if long_epochs is None:
            self.long_epochs = []
        else:
            self.long_epochs = long_epochs
        self.long_factor = long_factor
        self.invasion_criteria_memory = invasion_criteria_memory
        self.invasion_eig_buffer = invasion_eig_buffer

        if n_init is not None:
            self.K0 = n_init.shape[1]
            self.D = n_init.shape[0]

        self.EvoSim()

    def EvoSim(self):
        # Run evolutionary dynamics and save data

        self.dt_list = [] # list of dt for each epoch
        self.epoch_time_list = [] # list of total times for each epoch
        self.n_init_list = [] # initial distribution of n
        SavedQuants = EvoSavedQuantities(self.sample_num,self.epochs_to_save_traj)

        self.surviving_bool_list = []
        self.starting_species_idx_list = []
        self.current_species_idx = np.arange(self.K0)
        self.parent_idx_dict = {}
        self.K_tot = self.K0  #current total number of species

        self.invasion_eigs_list = []
        self.invasion_rejects_list = []
        self.invasion_success_list = []

        np.random.seed(seed=self.seed)  # Set random seed
        self.initialize_interactions_abundances()  #Sets V, S, and n0. Define V_dict and S_full

        # evolution
        for cur_epoch in range(self.epoch_num+1):
            self.evo_step(cur_epoch, SavedQuants) # dynamics for one epoch. V, n0 are for non-extinct types
            self.mut_step(SavedQuants)  # add new types
            # Save data in case job terminates on cluster
            self.save_data(SavedQuants)

            num_types = self.n0.shape[1]
            if num_types <= 10+self.mu:
                break

            print(cur_epoch)


    def evo_step(self, cur_epoch, SavedQuants):
        # Runs one epoch for the linear abundances. Returns nf (abundances for next timestep) and n_traj (sampled
        # trajectories for a single island)
        # n: (D,K) abundances
        # xbar0: (1,K) log of island averaged abundances.

        V = self.V
        S = self.S
        n0 = self.n0

        K = np.shape(n0)[1]
        D = self.D
        m = self.m
        N = self.N
        thresh = self.thresh

        M = self.define_M(K,m)

        epoch_steps, sample_steps, step_forward = self.setup_epoch(cur_epoch,K,M,V,S)

        Normalize = self.Normalize
        Extinction = self.Extinction
        check_new_types_extinct = self.check_new_types_extinct

        nbar = np.mean(n0, axis=0, keepdims=True)
        xbar0 = np.log(nbar) # Assumes all types have nbar>0
        y0 = n0 / nbar
        n0 = y0 * np.exp(xbar0)
        self.n_init_list.append(n0)
        self.starting_species_idx_list.append(self.current_species_idx)  #Add current species indices to list.

        # initialize for epoch
        SavedQuants.initialize_epoch(D,K,V,S,sample_steps)

        # dynamics
        for step in range(epoch_steps):

            # Save values each dt = sample_time
            SavedQuants.save_sample(step,y0,xbar0,cur_epoch)

            # Step abundances forward
            y1, xbar1 = step_forward(y0,xbar0)
            y1, xbar1 = Extinction(y1, xbar0, thresh)
            y1, xbar1 = Normalize(y1, xbar1, N)

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
        SavedQuants.compute_averages()
        SavedQuants.save_to_lists(cur_epoch,xbar0,self.mu,self.current_species_idx)

        ####### Change number of surviving species.
        self.subset_to_surviving_species(y0,xbar0,cur_epoch)


    def initialize_interactions_abundances(self):
        if self.V_init is None:
            V = generate_interactions_with_diagonal(self.K0, self.gamma)
        else:
            V = self.V_init

        #Add interactions to dictionary
        V_dict = {}
        for ii in range(self.K0):
            for jj in range(self.K0):
                V_dict[ii,jj] = V[ii,jj]
        self.V_dict = V_dict

        if self.n_init is None:
            n0 = initialization_many_islands(self.D, self.K0, self.N, self.m, 'flat')
        else:
            n0 = self.n_init

        if self.S_init is None:
            S = self.sig_S*np.random.normal(size=(1,self.K0))
        else:
            S = self.S_init
        self.S_full = S

        self.n0 = n0
        self.V = V
        self.S = S

    def define_M(self,K,m):
        #Define log-migration parameter
        M = np.log(1/(m*np.sqrt(K)))
        return M

    def setup_epoch(self,cur_epoch,K,M,V,S):
        # rescale time to appropriate fraction of short timescale
        m = self.m
        N = self.N

        dt = self.dt * (K ** 0.5) * (M ** -0.5)
        self.dt_list.append(dt)

        # epoch time
        if cur_epoch == 0:
            epoch_timescale = 4*K**(-0.5)  # First epoch only for 4 * K**(0.5) * M
        elif cur_epoch in self.long_epochs:
            epoch_timescale = self.epoch_timescale * self.long_factor
        else:
            epoch_timescale = self.epoch_timescale
        epoch_time = epoch_timescale*(K**1)*M  # epoch_timescale*(long timescale)
        epoch_steps = int(epoch_time / dt)
        epoch_time = epoch_steps * dt
        self.epoch_time_list.append(epoch_time)  # save amount of time for current epoch
        sample_steps = int(epoch_steps / self.sample_num) + 1

        u = 0
        normed = True
        deriv = define_deriv_many_islands_selective_diffs(V, S, N, u, m, normed)
        def step_forward(y0,xbar0):
            y1, xbar1 = step_rk4_many_islands(y0,xbar0,m,dt,deriv)
            return y1, xbar1

        return epoch_steps, sample_steps, step_forward

    def subset_to_surviving_species(self,y0,xbar0,cur_epoch):
        surviving_bool = xbar0[0, :] > -np.inf  # surviving species out of K1 current species.
        self.surviving_bool_list.append(surviving_bool)
        self.current_species_idx = self.current_species_idx[surviving_bool]

        # only pass surviving types to next timestep
        n0 = np.exp(xbar0) * y0
        self.n0 = n0[:, surviving_bool]
        self.V = self.V[np.ix_(surviving_bool, surviving_bool)]
        self.S = self.S[:,surviving_bool]

        if cur_epoch > 0:
            invasion_success_bool = surviving_bool[-self.mu:]
            self.invasion_success_list.extend(invasion_success_bool)

    def mut_step(self,SavedQuants):
        """
        Generate new mutants and update list of alive types accordingly
        :param n0: Current distribution of types (DxK matrix)
        :return: V - current interaction matrix, n0_new - distribution of types with new mutants
        """
        V = self.V
        S = self.S
        n0 = self.n0

        mu = self.mu
        K = np.shape(V)[0]
        V_new, S_new, parent_idx_list = self.gen_new_invasions(V,S,SavedQuants) #generated interactions correlated with parent

        # set new invasion types
        n0_new = self.next_epoch_abundances(mu, n0)

        K_tot = self.K_tot
        species_idx = np.append(self.current_species_idx,np.arange(K_tot,K_tot+mu))
        self.current_species_idx = species_idx
        for idx in range(mu):
            self.parent_idx_dict[K_tot+idx] = parent_idx_list[idx]
        self.K_tot = K_tot + mu

        self.V = V_new
        self.S = S_new
        self.n0 = n0_new

        self.store_new_interactions()

    def next_epoch_abundances(self,mu,n0):
        D = self.D
        K = len(self.current_species_idx)
        n0_new = np.zeros((D,K+mu))
        n0_new[:,:K] = n0
        n0_new[:,K:] = self.invasion_freq_factor/K
        n0_new = n0_new/np.sum(n0_new,axis=1,keepdims=True)
        return n0_new

    def gen_new_invasions(self,V,S,SavedQuants):
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
                invade_bool = self.invasion_criteria(invasion_eig,K)
                invasion_counts += 1

            invasion_eigs.append(invasion_eig)
            parent_idx_list.append(int(self.current_species_idx[par_idx]))
            V_new[idx, 0:idx] = V_row
            V_new[0:idx, idx] = V_col
            V_new[idx, idx] = V_diag
            S_new[0, idx] = s

        self.invasion_rejects_list.append(invasion_counts - mu)
        self.invasion_eigs_list.extend(invasion_eigs)


        return V_new, S_new, parent_idx_list

    def gen_related_interactions(self,V_new,S_new,par_idx,idx):
        # generate new types from random parents
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

        s = corr_mut * S_new[0, par_idx] + np.sqrt(1 - corr_mut ** 2) * self.sig_S * np.random.normal()

        return V_row, V_col, V_diag, s

    def compute_invasion_eig(self,SavedQuants,row,s):
        # Computes invasion eigenvalue based on previous epochs mean abundances and lambda
        surviving_bool = self.surviving_bool_list[-1]
        n_mean = SavedQuants.n_mean_ave_list[-1]
        lambd = SavedQuants.lambda_mean_ave_list[-1]
        S_mean = SavedQuants.S_mean_ave_list[-1]
        invasion_eig = s + np.dot(row,n_mean[surviving_bool]) - S_mean - lambd
        return invasion_eig

    def invasion_criteria(self,invasion_eig,K):
        # find the minimum invasion eigenvalue of the last x successful invasions.
        # x is self.invasion_criteria_memory
        # set invade_bool to True if invasion_eig is within invasion_eig_buffer of the minimum.

        past_eigs = np.array(self.invasion_eigs_list)
        success_bool = np.array(self.invasion_success_list)==True
        success_eigs = past_eigs[success_bool]
        if len(success_eigs) > self.invasion_criteria_memory:
            min_eig = np.min(success_eigs[-self.invasion_criteria_memory:])
            invade_bool = invasion_eig > min_eig - self.invasion_eig_buffer*1/np.sqrt(K)
        else:
            invade_bool = True
        return invade_bool

    def store_new_interactions(self):
        V = self.V
        S = self.S
        mu = self.mu
        species_idx = self.current_species_idx
        K = V.shape[0]

        self.S_full = np.append(self.S_full,S[:,-mu:])

        V_dict = self.V_dict
        for new_idx in range(K-mu,K):
            for idx in range(K):
                V_dict[species_idx[idx],species_idx[new_idx]] = V[idx,new_idx]
                V_dict[species_idx[new_idx], species_idx[idx]] = V[new_idx,idx]
        self.V_dict = V_dict

    def save_data(self,SavedQuants):
        self.sim_end_time = time.time()
        self.sim_end_process_time = time.process_time()
        data = vars(self)
        data = SavedQuants.add_data_to_dict(data)
        np.savez(self.file_name, data = data)

    def Normalize(self, y, xbar, N):
        n = np.exp(xbar) * y
        Nhat = np.sum(n, axis=1, keepdims=True)
        Yhat = np.mean(y * N / Nhat, axis=0, keepdims=True)

        Yhat[Yhat == 0] = 1

        y1 = (y * N / Nhat) / Yhat
        xbar1 = xbar + np.log(Yhat)

        return y1, xbar1

    def Extinction(self, y, xbar, thresh):
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


class EvoSavedQuantities:

    def __init__(self,sample_num,epochs_to_save_traj):
        self.sample_num = sample_num
        self.epochs_to_save_traj = epochs_to_save_traj

        # intialize lists to store values for each epoch
        self.n_mean_ave_list = []  # <n> over epoch
        self.n2_mean_ave_list = []  # <n^2> over epoch
        self.lambda_mean_ave_list = []  # <\lambda> over epoch
        self.n_mean_std_list = []
        self.n2_mean_std_list = []
        self.lambda_mean_std_list = []
        self.force_mean_ave_list = []
        self.force_mean_std_list = []
        self.invasion_species_idx = []
        self.invasion_eigs = []
        self.invasion_success = []
        self.S_mean_ave_list = []
        self.S_mean_std_list = []

        self.n_traj_dict = {}

    def initialize_epoch(self,D,K,V,S,sample_steps):
        self.V = V
        self.S = S
        self.count_short = 0
        self.n_mean_array = np.zeros((D, K))
        self.n2_mean_array = np.zeros((D, K))
        self.lambda_mean_array = np.zeros((D))
        self.n_traj = np.zeros((K, sample_steps))

    def save_sample(self,step,y0,xbar0,cur_epoch):

        if step % self.sample_num == 0:
            ind = int(step // self.sample_num)
            self.count_short += 1

            n0 = np.exp(xbar0) * y0
            if cur_epoch in self.epochs_to_save_traj or -1 in self.epochs_to_save_traj:
                self.n_traj[:, ind] = n0[0, :]
            self.n_mean_array += n0
            self.n2_mean_array += n0 ** 2
            self.lambda_mean_array += np.einsum('di,ij,dj->d', n0, self.V, n0)

    def compute_averages(self):
        self.n_mean_array *= 1 / self.count_short
        self.n2_mean_array *= 1 / self.count_short
        self.lambda_mean_array *= 1 / self.count_short

    def save_to_lists(self,cur_epoch,xbar,mu,species_idx):
        # input mu: number of mutants
        # input species_idx: current species idx.

        n_mean_ave = np.mean(self.n_mean_array, axis=0)
        n2_mean_ave = np.mean(self.n2_mean_array, axis=0)
        lambda_mean_ave = np.mean(self.lambda_mean_array, axis=0)

        n_mean_std = np.std(self.n_mean_array, axis=0)
        n2_mean_std = np.std(self.n2_mean_array, axis=0)
        lambda_mean_std = np.std(self.lambda_mean_array, axis=0)

        force_mean = np.einsum('ij,dj',self.V,self.n_mean_array)
        force_mean_ave = np.mean(force_mean,axis=0)
        force_mean_std = np.std(force_mean, axis=0)

        S_mean = np.sum(self.S*self.n_mean_array,axis=1)
        S_mean_ave = np.mean(S_mean)
        S_mean_std = np.std(S_mean)

        self.n_mean_ave_list.append(n_mean_ave)
        self.n2_mean_ave_list.append(n2_mean_ave)
        self.lambda_mean_ave_list.append(lambda_mean_ave)
        self.force_mean_ave_list.append(force_mean_ave)
        self.S_mean_ave_list.append(S_mean_ave)

        self.n_mean_std_list.append(n_mean_std)
        self.n2_mean_std_list.append(n2_mean_std)
        self.lambda_mean_std_list.append(lambda_mean_std)
        self.force_mean_std_list.append(force_mean_std)
        self.S_mean_std_list.append(S_mean_std)

        self.save_traj(cur_epoch)


    def save_traj(self,cur_epoch):
        if cur_epoch in self.epochs_to_save_traj:
            self.n_traj_dict[cur_epoch] = self.n_traj

        if -1 in self.epochs_to_save_traj:
            self.n_traj_dict[-1] = self.n_traj


    def add_data_to_dict(self,data):
        var_dict = vars(self)
        keys = list(var_dict.keys())
        keys.remove('n_traj')
        keys.remove('V')
        keys.remove('S')

        for key in keys:
            data[key] = var_dict[key]

        return data

class ManyIslandsSim:
    # Many islands simulation that saves island-average over all epochs >= sample_epoch_start. This saving happens after each epoch.
    # Quantities saved: histogram of log n for each type, island-averages, correlation function averaged over epochs

    def __init__(self,file_name,D,K,M,gamma,thresh,dt,seed,epoch_time,epoch_num,sample_time,sample_epoch_start):

        # Frequency normalization
        # Time normalization: simulation is in "natural normalization". Intermediate timescale is t = K**(1/2). Long timescale is t = K**(1/2)*(temperature)

        # input file_name: name of save file
        # input D: number of islands
        # input K: number of types
        # input M: log(1/m) where m = migration rate
        # input gamma: symmetry parameter
        # input thresh: extinction threshold for log-frequency equal to log(1/N) for total population N. 
        # input dt: step size, typically choose 0.1
        # input seed: random seed
        # input epoch_time: time for each epoch
        # input epoch_num: number of epochs
        # input sample_time: time over which samples are taken for averaging and computing correlation function.
        # input sample_epoch_start: number of epochs after which averaging starts.

        
        self.start_time = time.time()
        self.file_name = file_name
        self.D = D
        self.K = K
        self.M = M
        self.gamma = gamma
        self.thresh = thresh
        self.epoch_time = epoch_time 
        self.epoch_num = epoch_num
        self.dt = dt
        self.seed = seed
        self.sample_time = sample_time
        self.sample_epoch_start = sample_epoch_start

        self.SetParams()

        self.ManyIslandsSim()


    def SetParams(self):

        D = self.D
        K = self.K
        dt = self.dt

        self.N = 1
        self.m = np.exp(-self.M)  #for normalization with N = 1, V = O(1)
    

        # epoch time
        self.epoch_steps = int(self.epoch_time/dt)
        self.epoch_time = self.epoch_steps*dt

        # self.sample_time_short = 1
        self.sample_num = int(self.sample_time/dt)
        self.sample_time = dt*self.sample_num


        np.random.seed(seed=self.seed)

        self.V = generate_interactions_with_diagonal(K,self.gamma)    
        self.n0 = initialization_many_islands(D,K,self.N,self.m,'flat')
        self.x0 = np.log(self.n0[0,:])

        self.increment = 0.01*self.M  #increment for histogram


    def ManyIslandsSim(self):

        # Runs simulation for the linear abundances.
        # n: (D,K) abundances
        # xbar0: (1,K) log of island averaged abundances.

        file_name = self.file_name
        K = self.K
        D = self.D
        dt = self.dt
        m = self.m
        N = self.N
        n0 = self.n0
        thresh = self.thresh
        M = self.M
        V = self.V
        Normalize = self.Normalize
        Extinction = self.Extinction
        increment = self.increment
        
        sample_time = self.sample_time
        sample_num = self.sample_num
        sample_epoch_start = self.sample_epoch_start

        u = 0
        normed = True
        deriv = define_deriv_many_islands(V,N,u,m,normed)
        step_forward = step_rk4_many_islands

        nbar = np.mean(n0,axis=0,keepdims=True)
        xbar0 = np.log(nbar)
        y0 = n0/nbar

        
        V1 = V
        K1 = K
        self.V1 = V1 #current V during simulation
        self.K1 = K1 #current K during simulation

        species_indices = np.arange(K)  #list of species original indicies
        surviving_bool = xbar0[0,:]>-np.inf

        ########### Initialize variables.
        

        n_mean_ave_list = []
        n2_mean_ave_list = []
        n_cross_mean_list = []
        lambda_mean_ave_list = []

        n_mean_std_list = []
        n2_mean_std_list = []
        lambda_mean_std_list = []

        autocorr_list = []
        self.extinct_time_array = np.inf*np.ones((K))

        xbar_dict = [{} for step in range(K)]
        x_dict = [{} for step in range(K)]

        
        epoch = 0
        current_time = 0

        ########### Run dynamics
        while epoch<self.epoch_num:
            epoch += 1

            #### initialize for epoch
            count_short = 0

            n_mean_array = np.zeros((D,K1))
            n2_mean_array = np.zeros((D,K1))
            n_cross_mean_array = np.zeros((K1))
            lambda_mean_array = np.zeros((D))

            new_extinct_bool = False  #boolean for whether new extinctions have occured during epoch.


            n0 = y0*np.exp(xbar0)

            epoch_steps = self.epoch_steps

            sample_steps = int(epoch_steps/sample_num)+1
            n_traj = np.zeros((K1,sample_steps))


            for step in range(epoch_steps):

                ##### Save values each dt = sample_time
                if step % sample_num == 0 and epoch >= sample_epoch_start:
                    ind = int(step//sample_num)
                    count_short += 1

                    n_traj[:,ind] = n0[0,:]
                    
                    n0 = np.exp(xbar0)*y0
                    n_mean_array += n0
                    n2_mean_array +=  n0**2
                    n_cross_mean_array += (np.sum(n0,axis=0)**2 - np.sum(n0**2,axis=0))/(D*(D-1))
                    
                    lambda_mean_array += np.einsum('di,ij,dj->d',n0,V1,n0)

                    ## Save data for Histograms
                    hist_thresh = -20*M
                    xbar_rounded = np.round(xbar0/increment)*increment
                    index = np.logical_and(xbar_rounded<hist_thresh, xbar_rounded > -np.inf)
                    xbar_rounded[index] = hist_thresh
                    for ii in range(K1):
                        if xbar_rounded[0,ii] > -np.inf:
                            ind = species_indices[ii]
                            xbar_dict[ind][xbar_rounded[0,ii]] = xbar_dict[ind].get(xbar_rounded[0,ii],0) + 1
                            x_dict[ind][xbar_rounded[0,ii]] = xbar_dict[ind].get(xbar_rounded[0,ii],0) + 1

    
                ######### Step abundances forward
                y1, xbar1 = step_forward(y0,xbar0,m,dt,deriv)
                y1, xbar1 = Extinction(y1,xbar1,thresh)
                y1, xbar1 = Normalize(y1,xbar1,N)

                ######### If extinctions occur, record ext time.
                new_extinct = np.logical_and(xbar1[0,:]==-np.inf,self.extinct_time_array[species_indices]==np.inf)                
                if np.any(new_extinct):
                    new_extinct_indices = species_indices[new_extinct]
                    self.extinct_time_array[new_extinct_indices] = (current_time)
                    new_extinct_bool = True
                    

                ######### Prep for next time step
                y0 = y1
                xbar0 = xbar1

                current_time += dt
                ######### end epoch time steps

            if epoch >= sample_epoch_start:
                ####### Compute averages
                n_mean_array *= 1/count_short
                n2_mean_array *= 1/count_short
                n_cross_mean_array *= 1/count_short
                lambda_mean_array *= 1/count_short

                n_mean_ave = np.zeros((K))
                n2_mean_ave = np.zeros((K))
                n_cross_mean = np.zeros((K))

                n_mean_std = np.zeros((K))
                n2_mean_std = np.zeros((K))

                # Average and standard dev across islands.
                n_mean_ave[species_indices] = np.mean(n_mean_array,axis=0)
                n2_mean_ave[species_indices] = np.mean(n2_mean_array,axis=0)
                n_cross_mean[species_indices] = n_cross_mean_array
                lambda_mean_ave = np.mean(lambda_mean_array,axis=0)

                n_mean_std[species_indices] = np.std(n_mean_array,axis=0)
                n2_mean_std[species_indices]= np.std(n2_mean_array,axis=0)
                lambda_mean_std = np.std(lambda_mean_array,axis=0)

                corr_tvec, autocorr = self.Calculate(n_traj)

                self.corr_tvec = corr_tvec
                autocorr_list.append(autocorr)


                ######## Save data

                n_mean_ave_list.append(n_mean_ave)
                n2_mean_ave_list.append(n2_mean_ave)
                n_cross_mean_list.append(n_cross_mean)
                lambda_mean_ave_list.append(lambda_mean_ave)

                n_mean_std_list.append(n_mean_std)
                n2_mean_std_list.append(n2_mean_std)
                lambda_mean_std_list.append(lambda_mean_std)

                self.n_mean_ave = np.mean(np.array(n_mean_ave_list),axis=0)
                self.n2_mean_ave = np.mean(np.array(n2_mean_ave_list),axis=0)
                self.n_cross_mean_ave = np.mean(np.array(n_cross_mean_list),axis=0)
                self.lambda_mean_ave = np.mean(np.array(lambda_mean_ave_list),axis=0)

                self.n_mean_std = np.mean(n_mean_std_list,axis=0)
                self.n2_mean_std = np.mean(n2_mean_std_list,axis=0)
                self.lambda_mean_std = np.mean(lambda_mean_std_list,axis=0)

                self.autocorr = np.mean(np.array(autocorr_list),axis=0)

                self.xbar_dict = xbar_dict
                self.x_dict = x_dict

                self.run_time = time.time() - self.start_time

                class_dict = vars(self)

                np.savez(file_name, class_obj = class_dict)


            ####### Change number of surviving species. 
            if new_extinct_bool is True:

                surviving_bool = xbar0[0,:]>-np.inf  #surviving species out of K1 current species.
                species_indices = species_indices[surviving_bool]

                K1 = np.sum(surviving_bool)
                V1_new = V1[surviving_bool,:][:,surviving_bool]
                V1 = V1_new

                self.K1 = K1
                self.V1 = V1

                y0 = y0[:,surviving_bool]
                xbar0 = xbar0[:,surviving_bool]

                deriv = define_deriv_many_islands(V1,N,u,m,normed)  #redefine deriv with new V1. 
                step_forward = step_rk4_many_islands

            
            ###### End simulation after too many extinctions
            
            # if K1 <= 4:
            #     break

            
            print(epoch)
            ######## end of current epoch


    def Calculate(self,n_traj):
        # Compute correlation function and add to autocorr_list
        
        rows, cols = np.shape(n_traj)
        # autocorr_list = np.zeros((rows,cols))
        autocorr_sum = np.zeros((cols))
        nonzero_num = 0

        for ii in range(rows):
            n = self.K*n_traj[ii,:]  #convert to conventional normalization
            length = len(n)
            timepoint_num_vec = length - np.abs(np.arange(-length//2,length//2))  #number of time points used to evaluate autocorrelation
            autocorr = scipy.signal.fftconvolve(n,n[::-1],mode='same')/timepoint_num_vec
            autocorr_sum = autocorr_sum+autocorr
            nonzero_num = nonzero_num+1
            # autocorr_list[step,:] = autocorr

        corr_time_vec = self.sample_time*np.arange(-length//2,length//2)
        corr_window = 2*self.K**(3/2)
        window = np.logical_and(corr_time_vec>-corr_window,corr_time_vec<corr_window)

        # autocorr_list = autocorr_list[:,window]
        autocorr_sum = autocorr_sum[window] #compute average
        corr_tvec = corr_time_vec[window]

        return corr_tvec, autocorr_sum
        

    def Normalize(self,y,xbar,N):
        n = np.exp(xbar)*y
        Nhat = np.sum(n,axis=1,keepdims=True)
        Yhat = np.mean(y*N/Nhat,axis=0,keepdims=True)

        Yhat[Yhat==0] = 1

        y1 = (y*N/Nhat)/Yhat
        xbar1 = xbar + np.log(Yhat)

        return y1, xbar1


    def Extinction(self,y,xbar,thresh):
        
        local_ext_ind = xbar+np.log(y)<thresh
        y[local_ext_ind] = 0

        global_ext_ind = np.all(y==0,axis=0)
        xbar[:,global_ext_ind] = -np.inf
        y[:,global_ext_ind] = 0
        
        return y, xbar

#################################################

class InfiniteIslandsSim:
    # Infinite island sim. For each epoch island average is set to the previous average. Epochs should increase in length over time.
    # Quantities saved: island average trajectories, correlation function over each epoch

    # Frequency normalization
    # Time normalization: simulation is in "natural normalization". Intermediate timescale is t = K**(1/2). Long timescale is t = K**(1/2)*(temperature)

    def __init__(self,file_name,K,M,gamma,dt,seed,epoch_times,sample_time):

        # input file_name: name of save file
        # input K: number of types
        # input M: log(1/m) where m = migration rate
        # input gamma: symmetry parameter
        # input dt: step size, typically choose 0.1
        # input seed: random seed
        # input epoch_times: list of epoch times
        # input sample_time: time over which samples are taken for averaging and computing correlation function.
        
        self.start_time = time.time()
        self.file_name = file_name
        self.D = 1
        self.K = K
        self.M = M
        self.gamma = gamma
        # self.thresh = thresh
        self.epoch_times = epoch_times 
        self.dt = dt
        self.seed = seed
        self.sample_time = sample_time

        self.SetParams()

        self.InfiniteIslandSim()


    def SetParams(self):

        D = self.D
        K = self.K
        dt = self.dt

        self.N = 1
        self.m = np.exp(-self.M)  #for normalization with N = 1, V = O(1)
    
        t0 = 1

        # epoch time
        self.epoch_steps = (self.epoch_times*t0/dt).astype('int')
        self.epoch_times = self.epoch_steps*dt

        # self.sample_time_short = 1
        self.sample_num = int(self.sample_time/dt)
        self.sample_time = dt*self.sample_num


        np.random.seed(seed=self.seed)

        self.V = generate_interactions_with_diagonal(K,self.gamma)    
        self.n0 = initialization_many_islands(D,K,self.N,self.m,'flat')
        self.x0 = np.log(self.n0[0,:])

        self.increment = 0.01*self.M  #increment for histogram


    def InfiniteIslandSim(self):

        # Runs simulation for the linear abundances.
        # n: (D,K) abundances
        # xbar0: (1,K) log of island averaged abundances.

        file_name = self.file_name
        D = self.D
        K = self.K
        dt = self.dt
        m = self.m
        N = self.N
        n0 = self.n0
        M = self.M
        V = self.V
        Normalize = self.Normalize
        Extinction = self.Extinction
        increment = self.increment
        
        sample_time = self.sample_time
        sample_num = self.sample_num
        

        u = 0
        normed = True
        deriv = define_deriv_infinite_islands(V,N,u,m,normed)
        step_forward = step_rk4_infinite_islands

        nbar = np.mean(n0,axis=0,keepdims=True)
        xbar0 = np.log(nbar)
        y0 = n0/nbar
        xbar_inf = np.log(N/K)*np.ones((D,K))
        
        V1 = V
        K1 = K
        self.V1 = V1 #current V during simulation
        self.K1 = K1 #current K during simulation

        species_indices = np.arange(K)  #list of species original indicies
        surviving_bool = xbar0[0,:]>-np.inf

        ########### Initialize variables.
        

        n_mean_ave_list = []
        n2_mean_ave_list = []
        lambda_mean_ave_list = []
        nbar_inf_list = []

        n_mean_std_list = []
        n2_mean_std_list = []
        lambda_mean_std_list = []

        autocorr_list = []
        
        epoch = 0
        current_time = 0

        ########### Run dynamics
        while epoch<len(self.epoch_times):

            #### initialize for epoch
            count_short = 0

            n_mean_array = np.zeros((D,K1))
            n2_mean_array = np.zeros((D,K1))
            lambda_mean_array= np.zeros((D))

            n0 = y0*np.exp(xbar0)

            epoch_steps = self.epoch_steps[epoch]

            sample_steps = int(epoch_steps/sample_num)+1
            n_traj = np.zeros((K1,sample_steps))


            for step in range(epoch_steps):

                ##### Save values each dt = sample_time
                if step % sample_num == 0:
                    ind = int(step//sample_num)
                    count_short += 1

                    n_traj[:,ind] = n0[0,:]
                    
                    n0 = np.exp(xbar0)*y0
                    n_mean_array += n0
                    n2_mean_array +=  n0**2
                    
                    lambda_mean_array += np.einsum('di,ij,dj->d',n0,V1,n0)

    
                ######### Step abundances forward
                y1, xbar1 = step_forward(y0,xbar0,xbar_inf,m,dt,deriv)
                # y1, xbar1 = Extinction(y1,xbar1,thresh)
                y1, xbar1 = Normalize(y1,xbar1,N)
                    

                ######### Prep for next time step
                y0 = y1
                xbar0 = xbar1

                current_time += dt
                ######### end epoch time steps

            ###### Update epoch number
            epoch += 1

            ####### Compute averages
            n_mean_array *= 1/count_short
            n2_mean_array *= 1/count_short
            lambda_mean_array *= 1/count_short

            n_mean_ave = np.zeros((K))
            n2_mean_ave = np.zeros((K))
            n_cross_mean = np.zeros((K))

            n_mean_std = np.zeros((K))
            n2_mean_std = np.zeros((K))

            # Average and standard dev across islands.
            n_mean_ave[species_indices] = np.mean(n_mean_array,axis=0)
            n2_mean_ave[species_indices] = np.mean(n2_mean_array,axis=0)
            lambda_mean_ave = np.mean(lambda_mean_array,axis=0)

            n_mean_std[species_indices] = np.std(n_mean_array,axis=0)
            n2_mean_std[species_indices]= np.std(n2_mean_array,axis=0)
            lambda_mean_std = np.std(lambda_mean_array,axis=0)

            corr_tvec, autocorr = self.Calculate(n_traj)

            self.corr_tvec = corr_tvec
            autocorr_list.append(autocorr)

            ######## Save data

            n_mean_ave_list.append(n_mean_ave)
            n2_mean_ave_list.append(n2_mean_ave)
            lambda_mean_ave_list.append(lambda_mean_ave)
            nbar_inf_list.append(K*np.exp(xbar_inf))

            n_mean_std_list.append(n_mean_std)
            n2_mean_std_list.append(n2_mean_std)
            lambda_mean_std_list.append(lambda_mean_std)

            self.n_mean_ave = np.array(n_mean_ave_list)           
            self.n2_mean_ave = np.array(n2_mean_ave_list)
            self.lambda_mean_ave = np.array(lambda_mean_ave_list)

            self.n_mean_std = np.array(n_mean_std_list)
            self.n2_mean_std = np.array(n2_mean_std_list)
            self.lambda_mean_std = np.array(lambda_mean_std_list)

            self.autocorr_list = autocorr_list
            self.nbar_inf_list = nbar_inf_list


            self.run_time = time.time() - self.start_time

            class_dict = vars(self)

            np.savez(file_name, class_obj = class_dict)


            ###### Update infinite island average (xbar_inf)

            xbar_inf = np.log(n_mean_array/K)  # factor of 1/K to fix normalization

            
            print(epoch)
            ######## end of current epoch


    def Calculate(self,n_traj):
        # Compute correlation function and add to autocorr_list
        
        rows, cols = np.shape(n_traj)
        # autocorr_list = np.zeros((rows,cols))
        autocorr_sum = np.zeros((cols))
        nonzero_num = 0

        for ii in range(rows):
            n = self.K*n_traj[ii,:]  #convert to conventional normalization
            length = len(n)
            timepoint_num_vec = length - np.abs(np.arange(-length//2,length//2))  #number of time points used to evaluate autocorrelation
            autocorr = scipy.signal.fftconvolve(n,n[::-1],mode='same')/timepoint_num_vec
            autocorr_sum = autocorr_sum+autocorr
            nonzero_num = nonzero_num+1
            # autocorr_list[step,:] = autocorr

        corr_time_vec = self.sample_time*np.arange(-length//2,length//2)
        corr_window = 2*self.K**(3/2)
        window = np.logical_and(corr_time_vec>-corr_window,corr_time_vec<corr_window)

        # autocorr_list = autocorr_list[:,window]
        autocorr_sum = autocorr_sum[window] # 1/K comes from normalization of V ~ 1/sqrt(K)
        corr_tvec = corr_time_vec[window]

        return corr_tvec, autocorr_sum
        

    def Normalize(self,y,xbar,N):
        n = np.exp(xbar)*y
        Nhat = np.sum(n,axis=1,keepdims=True)
        Yhat = np.mean(y*N/Nhat,axis=0,keepdims=True)

        Yhat[Yhat==0] = 1

        y1 = (y*N/Nhat)/Yhat
        xbar1 = xbar + np.log(Yhat)

        return y1, xbar1


    def Extinction(self,y,xbar,thresh):
        
        local_ext_ind = xbar+np.log(y)<thresh
        y[local_ext_ind] = 0

        global_ext_ind = np.all(y==0,axis=0)
        xbar[:,global_ext_ind] = -np.inf
        y[:,global_ext_ind] = 0
        
        return y, xbar


###############################

class ExtinctionTimeLong:
    # Simulation with many islands that allows extinctions. 
    # Removes types that are extinct to speed up simulation.
    # Saves data after each epoch: island-averages, extinction times, correlation function, migration fitness over each epoch

    # Frequency normalization
    # Time normalization: simulation is in "natural normalization". Intermediate timescale is t = K**(1/2). Long timescale is t = K**(1/2)*(temperature)

    def __init__(self,file_name,D,K,M,gamma,thresh,dt,seed,epoch_time,epoch_num,sample_time):

        # input file_name: name of save file
        # input D: number of islands
        # input K: number of types
        # input M: log(1/m) where m = migration rate
        # input gamma: symmetry parameter
        # input thresh: extinction threshold for log-frequency equal to log(1/N) for total population N. 
        # input dt: step size, typically choose 0.1
        # input seed: random seed
        # input epoch_time: time for each epoch
        # input epoch_num: number of epochs
        # input sample_time: time over which samples are taken for averaging and computing correlation function.


        
        self.file_name = file_name
        self.D = D
        self.K = K
        self.M = M
        self.gamma = gamma
        self.thresh = thresh
        self.epoch_time = epoch_time
        self.epoch_num = epoch_num
        self.dt = dt
        self.seed = seed
        self.sample_time = sample_time

        self.SetParams()

        self.ManyIslandsSim()


    def SetParams(self):

        D = self.D
        K = self.K
        dt = self.dt

        self.N = 1
        self.m = np.exp(-self.M)
    
        t0 = 1

        # epoch time
        self.epoch_steps = int(self.epoch_time/dt)
        self.epoch_time = self.epoch_steps*dt

        # self.sample_time_short = 1
        self.sample_num = int(self.sample_time/dt)
        self.sample_time = dt*self.sample_num


        np.random.seed(seed=self.seed)

        self.V = generate_interactions_with_diagonal(K,self.gamma)    
        self.n0 = initialization_many_islands(D,K,self.N,self.m,'flat')
        self.x0 = np.log(self.n0[0,:])

        self.increment = 0.01*self.M  #increment for histogram


    def ManyIslandsSim(self):

        # Runs simulation for the linear abundances.
        # n: (D,K) abundances
        # xbar0: (1,K) log of island averaged abundances.

        file_name = self.file_name
        K = self.K
        D = self.D
        dt = self.dt
        m = self.m
        N = self.N
        n0 = self.n0
        thresh = self.thresh
        M = self.M
        V = self.V
        Normalize = self.Normalize
        Extinction = self.Extinction
        increment = self.increment
        
        sample_time = self.sample_time
        sample_num = self.sample_num


        u = 0
        normed = True
        deriv = define_deriv_many_islands(V,N,u,m,normed)
        step_forward = step_rk4_many_islands

        nbar = np.mean(n0,axis=0,keepdims=True)
        xbar0 = np.log(nbar)
        y0 = n0/nbar

        
        V1 = V
        K1 = K
        self.V1 = V1 #current V during simulation
        self.K1 = K1 #current K during simulation

        species_indices = np.arange(K)  #list of species original indices
        surviving_bool = xbar0[0,:]>-np.inf

        ########### Initialize variables.
        

        self.n_mean_ave_list = []
        self.n2_mean_ave_list = []
        self.n_cross_mean_list = []
        self.mig_mean_list = [] # ratio nbar/n
        self.eta_mean_list = [] # eta computed from V, <n>
        self.eta_from_antisymmetric = np.zeros((K)) # eta computed from (V-V.T)/2
        self.lambda_mean_ave_list = []

        self.n_mean_std_list = []
        self.n2_mean_std_list = []
        self.lambda_mean_std_list = []

        self.autocorr_list = []
        self.n_init_list = []

        self.extinct_time_array = np.inf*np.ones((K))

        
        epoch = 0
        current_time = 0

        self.eta_from_antisymmetric = antisym_etas(V) # eta calculated using antisymmetric fixed point

        ########### Run dynamics
        while epoch<self.epoch_num:
            epoch += 1

            #### initialize for epoch
            count_short = 0

            n_mean_array = np.zeros((D,K1))
            n2_mean_array = np.zeros((D,K1))
            n_cross_mean_array = np.zeros((K1))
            lambda_mean_array = np.zeros((D))
            mig_mean_array = np.zeros((K1))

            new_extinct_bool = False  #boolean for whether new extinctions have occured during epoch.


            n_init = np.zeros((D,K))
            n0 = y0*np.exp(xbar0)

            n_init[:,species_indices] = n0
            self.n_init_list.append(n_init)

            if epoch <= 2:
                epoch_time = 50*M*K**(1/2)
                epoch_steps = int(epoch_time/self.dt)
            else:
                epoch_steps = self.epoch_steps

            sample_steps = int(epoch_steps/sample_num)+1
            n_traj = np.zeros((K1,sample_steps))


            for step in range(epoch_steps):

                ##### Save values each dt = sample_time
                if step % sample_num == 0:
                    ind = int(step//sample_num)
                    count_short += 1

                    n_traj[:,ind] = n0[0,:]
                    
                    n0 = np.exp(xbar0)*y0
                    n_mean_array += n0
                    n2_mean_array += n0**2
                    n_cross_mean_array += (np.sum(n0,axis=0)**2 - np.sum(n0**2,axis=0))/(D*(D-1))
                    nbar = np.mean(n0, axis=0).reshape((1,-1)) # island averaged abundance
                    temp_rats = np.divide(nbar, n0) # remove infinities
                    temp_rats[~(np.isfinite(temp_rats))] = 0

                    mig_mean_array += np.mean(temp_rats, axis=0)

                    lambda_mean_array += np.einsum('di,ij,dj->d',n0,V1,n0)


    
                ######### Step abundances forward
                y1, xbar1 = step_forward(y0,xbar0,m,dt,deriv)
                y1, xbar1 = Extinction(y1,xbar1,thresh)
                y1, xbar1 = Normalize(y1,xbar1,N)

                ######### If extinctions occur, record ext time.
                new_extinct = np.logical_and(xbar1[0,:]==-np.inf,self.extinct_time_array[species_indices]==np.inf)                
                if np.any(new_extinct):
                    new_extinct_indices = species_indices[new_extinct]
                    self.extinct_time_array[new_extinct_indices] = (current_time)
                    new_extinct_bool = True
                    

                ######### Prep for next time step
                y0 = y1
                xbar0 = xbar1

                current_time += dt
                ######### end epoch time steps

            ####### Compute averages
            n_mean_array *= 1/count_short
            n2_mean_array *= 1/count_short
            n_cross_mean_array *= 1/count_short
            mig_mean_array *= 1/count_short
            lambda_mean_array *= 1/count_short

            n_mean_ave = np.zeros((K))
            n2_mean_ave = np.zeros((K))
            n_cross_mean = np.zeros((K))
            mig_mean_ave = np.zeros((K))

            n_mean_std = np.zeros((K))
            n2_mean_std = np.zeros((K))

            # Average and standard dev across islands.
            n_mean_ave[species_indices] = np.mean(n_mean_array,axis=0)
            n2_mean_ave[species_indices] = np.mean(n_mean_array,axis=0)
            n_cross_mean[species_indices] = n_cross_mean_array
            mig_mean_ave[species_indices] = mig_mean_array
            lambda_mean_ave = np.mean(lambda_mean_array,axis=0)


            n_mean_std[species_indices] = np.std(n_mean_array,axis=0)
            n2_mean_std[species_indices]= np.std(n_mean_array,axis=0)
            lambda_mean_std = np.std(lambda_mean_array,axis=0)


            # compute estimate of etas, from mean field calculations. Assuming close to antisymmetric.

            sig_V = np.sqrt(np.var(V1)) # standard deviation of interaction matrix
            K_surv = np.sum(surviving_bool)
            chi = sig_V*np.sqrt(K_surv) # numerical estimate of chi
            eta_mean_ave = -self.gamma*chi*n_mean_ave+np.dot(V, n_mean_ave)-m*(mig_mean_ave-1)

            self.Calculate(n_traj)

            ######## Save data

            self.n_mean_ave_list.append(n_mean_ave)
            self.n2_mean_ave_list.append(n2_mean_ave)
            self.n_cross_mean_list.append(n_cross_mean)
            self.mig_mean_list.append(mig_mean_ave)
            self.eta_mean_list.append(eta_mean_ave)
            self.lambda_mean_ave_list.append(lambda_mean_ave)

            self.n_mean_std_list.append(n_mean_std)
            self.n2_mean_std_list.append(n2_mean_std)
            self.lambda_mean_std_list.append(lambda_mean_std)

            

            class_dict = vars(self)

            np.savez(file_name, class_obj = class_dict)

            ####### Change number of surviving species.
            if new_extinct_bool is True:
                surviving_bool = xbar0[0, :] > -np.inf  # surviving species out of K1 current species.
                species_indices = species_indices[surviving_bool]

                K1 = np.sum(surviving_bool)
                V1_new = V1[surviving_bool, :][:, surviving_bool]
                V1 = V1_new

                self.K1 = K1
                self.V1 = V1

                y0 = y0[:, surviving_bool]
                xbar0 = xbar0[:, surviving_bool]

                deriv = define_deriv_many_islands(V1, N, u, m, normed)  # redefine deriv with new V1.
                step_forward = step_rk4_many_islands

            ###### End simulation after too many extinctions
            
            if K1 <= 4:
                break

            
            print(epoch)
            ######## end of current epoch


    def Calculate(self,n_traj):
        # Compute correlation function and add to autocorr_list
        
        rows, cols = np.shape(n_traj)
        # autocorr_list = np.zeros((rows,cols))
        autocorr_sum = np.zeros((cols))
        nonzero_num = 0

        for ii in range(rows):
            n = self.K*n_traj[ii,:]  #convert to conventional normalization
            length = len(n)
            timepoint_num_vec = length - np.abs(np.arange(-length//2,length//2))  #number of time points used to evaluate autocorrelation
            autocorr = scipy.signal.fftconvolve(n,n[::-1],mode='same')/timepoint_num_vec
            autocorr_sum = autocorr_sum+autocorr
            nonzero_num = nonzero_num+1
            # autocorr_list[step,:] = autocorr

        corr_time_vec = self.sample_time*np.arange(-length//2,length//2)
        corr_window = 2*self.K**(3/2)
        window = np.logical_and(corr_time_vec>-corr_window,corr_time_vec<corr_window)

        # autocorr_list = autocorr_list[:,window]
        autocorr_sum = autocorr_sum[window]
        corr_tvec = corr_time_vec[window]

        self.corr_tvec = np.float32(corr_tvec)
        self.autocorr_list.append(autocorr_sum)   # 1/K comes from normalization of V ~ 1/sqrt(K)
        

    def Normalize(self,y,xbar,N):
        n = np.exp(xbar)*y
        Nhat = np.sum(n,axis=1,keepdims=True)
        Yhat = np.mean(y*N/Nhat,axis=0,keepdims=True)

        Yhat[Yhat==0] = 1

        y1 = (y*N/Nhat)/Yhat
        xbar1 = xbar + np.log(Yhat)

        return y1, xbar1


    def Extinction(self,y,xbar,thresh):
        
        local_ext_ind = xbar+np.log(y)<thresh
        y[local_ext_ind] = 0

        global_ext_ind = np.all(y==0,axis=0)
        xbar[:,global_ext_ind] = -np.inf
        y[:,global_ext_ind] = 0
        
        return y, xbar


###############################

class AntisymEvo:
    # Simulation with many islands with random invasion of new types.
    # Mutations come in slowly, spaced out in epochs of O((K**(0.5))*M) in natural time units
    # Corresponds to long time between mutations - bursts likely to happen sooner than new mutants coming in
    # Pre-computes interaction matrix and works with appropriate slices.
    # Saves data after each epoch: island-averages, extinction times, correlation function, eta over each epoch

    # Frequency normalization
    # Time normalization: simulation is in "natural normalization". Short timescale is dt = K**(1/2)*M**(-1/2).
    # Long timescale is t = K**(1/2)*(temperature)

    def __init__(self, file_name, D, K, M, gamma, thresh, inv_fac, dt, mu, seed, epoch_timescale, epoch_num,
                 sample_num, c_A=None):

        # input file_name: name of save file
        # input D: number of islands
        # input K: number of types
        # input M: log(1/m) where m = migration rate
        # input gamma: symmetry parameter
        # input thresh: extinction threshold for log-frequency equal to log(1/N) for total population N.
        # input inv_fac: invasion factor. Invading types start at number inv_fac*exp(thresh)
        # input dt: rescaled step size, typically choose 0.1
        # input mu: number of types invading per epoch
        # input seed: random seed
        # input epoch_timescale: time for each epoch, units of M*K**0.5
        # input epoch_num: number of epochs
        # input sample_num: sampling period
        # input c_A: correlation of new types with previous ones

        self.file_name = file_name
        self.D = D
        self.K = K
        self.M = M
        self.gamma = gamma
        self.thresh = thresh
        self.inv_fac = inv_fac
        self.epoch_timescale = epoch_timescale
        self.epoch_num = epoch_num
        self.dt = dt
        self.mu = mu
        self.seed = seed
        self.sample_num = sample_num
        self.sim_start_time = time.time()
        self.c_A = c_A

        self.SetParams()

        self.EvoSim()

    def SetParams(self):

        D = self.D
        K = self.K


        self.N = 1
        self.m = np.exp(-self.M)

        self.K_tot = self.K + self.epoch_num * self.mu  # total number of types through simulation

        np.random.seed(seed=self.seed)

        if self.c_A==None:
            # if children are random, generate full interaction matrix at start
            self.V = generate_interactions_with_diagonal(self.K_tot, self.gamma)

        self.increment = 0.01 * self.M  # increment for histogram

    def EvoSim(self):

        # Run evolutionary dynamics and save data


        self.dt_list = [] # list of dt for each epoch
        self.epoch_time_list = [] # list of total times for each epoch
        self.eta_list = []  # etas at end of each epoch

        self.n_init_list = [] # initial distribution of n
        self.n_mean_ave_list = [] # <n> over epoch
        self.n2_mean_ave_list = [] # <n^2> over epoch
        self.n_cross_mean_list = []
        self.mig_mean_list = []  # ratio nbar/n
        self.eta_mean_list = []  # eta computed from V, <n>
        self.lambda_mean_ave_list = [] # <\lambda> over epoch

        self.n_mean_std_list = []
        self.n2_mean_std_list = []
        self.lambda_mean_std_list = []

        self.extinct_time_array = np.inf * np.ones((self.K_tot)) # extinction times
        self.n_alive = np.zeros((self.K_tot,self.epoch_num+2), dtype=bool)  # K x (epoch_num+2) matrix of which \
        # types are alive at a given time

        # initial setup
        self.n_alive[0:self.K,0] = True
        n0 = initialization_many_islands(self.D, self.K, self.N, self.m, 'flat')


        # run first epoch to equilibrate
        if self.c_A==None:
            V = self.V[np.ix_(np.arange(0,self.K),np.arange(0,self.K))] # pick right subset of V
        else:
            # if children related, generate original interactions
            V = generate_interactions_with_diagonal(self.K, self.gamma)
        
        n0, n_traj_eq, V = self.evo_step(V,n0,1) # equilibration dynamics
        self.n_traj_eq = n_traj_eq  # save first trajectory for debugging purposes
        # evolution
        for i in range(2,self.epoch_num+2):

            V,n0 = self.mut_step(V,n0,i) # add new types
            n0, n_traj_f, V = self.evo_step(V,n0,i) # dynamics
            self.n_traj_f = n_traj_f  #save last epoch trajectories for debugging purposes

            # periodically save in case simulation terminates in cluster.
            if i % 10 ==0:
                self.sim_end_time = time.time()
                class_dict = vars(self)
                np.savez(self.file_name, class_obj=class_dict)

        # save last V for correlated evolution
        if self.c_A!=None:
            self.V = V

        # save data
        self.sim_end_time = time.time()
        class_dict = vars(self)

        np.savez(self.file_name, class_obj=class_dict)

    def evo_step(self,V,n0,cur_epoch):
        # Runs one epoch for the linear abundances. Returns nf (abundances for next timestep) and n_traj (sampled
        # trajectories for a single island)
        # n: (D,K) abundances
        # xbar0: (1,K) log of island averaged abundances.

        K = np.shape(n0)[1]
        D = self.D
        m = self.m
        N = self.N
        thresh = self.thresh
        M = self.M

        # rescale time to appropriate fraction of short timescale
        dt = self.dt * (K ** 0.5) * (M ** -0.5)
        self.dt_list.append(dt)

        # epoch time
        epoch_time = self.epoch_timescale*(K**0.5)*M  # epoch_timescale*(long timescale)
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

        u = 0
        normed = True
        deriv = define_deriv_many_islands(V, N, u, m, normed)
        step_forward = step_rk4_many_islands

        nbar = np.mean(n0, axis=0, keepdims=True)
        xbar0 = np.log(nbar)
        y0 = n0 / nbar

        n_alive = self.n_alive[:,cur_epoch-1]
        species_indices = np.arange(self.K_tot)[n_alive]  # list of current species indices
        surviving_bool = xbar0[0, :] > -np.inf

        current_time = 0


        #### initialize for epoch
        count_short = 0

        n_mean_array = np.zeros((D, K))
        n2_mean_array = np.zeros((D, K))
        n_cross_mean_array = np.zeros((K))
        lambda_mean_array = np.zeros((D))
        mig_mean_array = np.zeros((K))


        n0 = y0 * np.exp(xbar0)

        self.n_init_list.append(n0)

        sample_steps = int(epoch_steps / self.sample_num) + 1
        n_traj = np.zeros((K, sample_steps))

        # dynamics
        for step in range(epoch_steps):

            ##### Save values each dt = sample_time
            if step % self.sample_num == 0:
                ind = int(step // self.sample_num)
                count_short += 1

                n_traj[:, ind] = n0[0, :]

                n0 = np.exp(xbar0) * y0
                n_mean_array += n0
                n2_mean_array += n0 ** 2
                n_cross_mean_array += (np.sum(n0, axis=0) ** 2 - np.sum(n0 ** 2, axis=0)) / (D * (D - 1))
                nbar = np.mean(n0, axis=0).reshape((1, -1))  # island averaged abundance
                temp_rats = np.divide(nbar, n0)  # remove infinities
                temp_rats[~(np.isfinite(temp_rats))] = 0

                mig_mean_array += np.mean(temp_rats, axis=0)

                lambda_mean_array += np.einsum('di,ij,dj->d', n0, V, n0)

            ######### Step abundances forward
            y1, xbar1 = step_forward(y0, xbar0, m, dt, deriv)
            y1, xbar1 = Extinction(y1, xbar1, thresh)
            y1, xbar1 = Normalize(y1, xbar1, N)

            ######### If extinctions occur, record ext time.
            new_extinct = np.logical_and(xbar1[0, :] == -np.inf, self.extinct_time_array[species_indices] == np.inf)
            if np.any(new_extinct):
                new_extinct_indices = species_indices[new_extinct]
                self.extinct_time_array[new_extinct_indices] = current_time+t0
                new_extinct_bool = True

            ######### Prep for next time step
            y0 = y1
            xbar0 = xbar1

            current_time += dt
            ######### end epoch time steps

        ####### Compute averages
        n_mean_array *= 1 / count_short
        n2_mean_array *= 1 / count_short
        n_cross_mean_array *= 1 / count_short
        mig_mean_array *= 1 / count_short
        lambda_mean_array *= 1 / count_short


        # Average and standard dev across islands.
        n_mean_ave = np.mean(n_mean_array, axis=0)
        n2_mean_ave = np.mean(n_mean_array, axis=0)
        n_cross_mean = n_cross_mean_array
        mig_mean_ave = mig_mean_array
        lambda_mean_ave = np.mean(lambda_mean_array, axis=0)

        n_mean_std = np.std(n_mean_array, axis=0)
        n2_mean_std = np.std(n_mean_array, axis=0)
        lambda_mean_std = np.std(lambda_mean_array, axis=0)

        # compute estimate of etas, from mean field calculations. Assuming close to antisymmetric.

        sig_V = np.sqrt(np.var(V))  # standard deviation of interaction matrix
        K_surv = np.sum(surviving_bool)
        chi = sig_V * np.sqrt(K_surv)  # numerical estimate of chi
        eta_mean_ave = -self.gamma * chi * n_mean_ave + np.dot(V, n_mean_ave) - m * (mig_mean_ave - 1)

        #self.Calculate(n_traj)

        ######## Save data

        self.n_mean_ave_list.append(n_mean_ave)
        self.n2_mean_ave_list.append(n2_mean_ave)
        self.n_cross_mean_list.append(n_cross_mean)
        self.mig_mean_list.append(mig_mean_ave)
        self.eta_mean_list.append(eta_mean_ave)
        self.lambda_mean_ave_list.append(lambda_mean_ave)

        self.n_mean_std_list.append(n_mean_std)
        self.n2_mean_std_list.append(n2_mean_std)
        self.lambda_mean_std_list.append(lambda_mean_std)

        # starting frequencies for next step
        nf = np.exp(xbar0) * y0

        ####### Change number of surviving species.
        surviving_bool = xbar0[0, :] > -np.inf  # surviving species out of K1 current species.
        species_indices = species_indices[surviving_bool]

        self.n_alive[species_indices,cur_epoch] = True

        # only pass surviving types to next timestep
        nf = nf[:,surviving_bool]
        V_surv = V[np.ix_(surviving_bool,surviving_bool)]

        print(cur_epoch)
        ######## end of current epoch
        return nf, n_traj, V_surv

    def mut_step(self,V,n0,cur_epoch):
        """
        Generate new mutants and update list of alive types accordingly
        :param n0: Current distribution of types (DxK matrix)
        :param cur_epoch: Current epoch
        :return: V - current interaction matrix, n0_new - distribution of types with new mutants
        """

        species_indices = np.arange(self.K_tot)[self.n_alive[:,cur_epoch-1]] # indices of alive types
        K = len(species_indices)
        D = self.D
        # set new invasion types
        n0_new = np.zeros((D,K+self.mu))
        n0_new[:,0:K] = n0  # old types
        n0_new[:,K:] = self.inv_fac*np.exp(self.thresh) # new types

        # set alive types
        K0 = self.mu*(cur_epoch-2)+self.K
        self.n_alive[K0:K0+self.mu, cur_epoch - 1] = True
        # normalize
        for i in range(D):
            n0_new[i,:] = n0_new[i,:]/np.sum(n0_new[i,:])
        
        n_alive = self.n_alive[:,cur_epoch-1]
        if self.c_A==None:
            V_new = self.V[np.ix_(n_alive,n_alive)] # active interactions
        else:
            V_new = self.gen_related_interactions(V) # generate new interactions via descent

        return V_new,n0_new

    def gen_related_interactions(self,V):
        c_A = self.c_A
        mu = self.mu
        K = np.shape(V)[0]
        V_new = np.zeros((K+self.mu,K+self.mu))
        V_new[0:K, 0:K] = V

        # generate new types from random parents
        cov_mat = [[1.,self.gamma],[self.gamma,1.]]
        for k in range(mu):
            par_idx = np.random.randint(0, K)  # parent index
            if self.gamma==-1:
                z_vec = np.random.randn(K+k)
                V_new[K + k, 0:(K + k)] = c_A * V_new[par_idx, 0:(K + k)] + np.sqrt(1 - c_A ** 2) * z_vec  # row
                V_new[0:(K + k), K + k] = c_A * V_new[0:(K + k), par_idx] - np.sqrt(1 - c_A ** 2) * z_vec  # column
                V_new[K + k, K + k] = 0  # diagonal
            else:
                z_mat = np.random.multivariate_normal([0,0],cov=cov_mat,size=(K+k))  # 2 x (K+k) matrix of differences from parent
                V_new[K + k, 0:(K + k)] = c_A * V_new[par_idx, 0:(K + k)] + np.sqrt(1 - c_A ** 2) * z_mat[:,0]  # row
                V_new[0:(K + k), K + k] = c_A * V_new[0:(K + k),par_idx] + np.sqrt(1 - c_A ** 2) * z_mat[:,1]  # column
                V_new[K+k,K+k] = 0 # diagonal

        return V_new

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

    def Normalize(self, y, xbar, N):
        n = np.exp(xbar) * y
        Nhat = np.sum(n, axis=1, keepdims=True)
        Yhat = np.mean(y * N / Nhat, axis=0, keepdims=True)

        Yhat[Yhat == 0] = 1

        y1 = (y * N / Nhat) / Yhat
        xbar1 = xbar + np.log(Yhat)

        return y1, xbar1

    def Extinction(self, y, xbar, thresh):

        local_ext_ind = xbar + np.log(y) < thresh
        y[local_ext_ind] = 0

        global_ext_ind = np.all(y == 0, axis=0)
        xbar[:, global_ext_ind] = -np.inf
        y[:, global_ext_ind] = 0

        return y, xbar


class bp_evo:
    # Simulation of bacteria-phage evolution with many islands with random invasion of new types.
    # Mutations come in slowly, spaced out in epochs of O((K**(0.5))*M) in natural time units
    # Corresponds to long time between mutations - bursts likely to happen sooner than new mutants coming in
    # Pre-computes interaction matrix and works with appropriate slices.
    # Saves data after each epoch: island-averages, extinction times, correlation function, eta over each epoch

    # Dynamical equations: db/dt = b(Hp-\lambda_b)b
    #                      dp/dt = p(Gb-lambda_p)p

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

        self.lambda_b_mean_ave_list = []  # <\lambda> over epoch
        self.lambda_b_mean_std_list = []

        self.lambda_p_mean_ave_list = []  # <\lambda> over epoch
        self.lambda_p_mean_std_list = []

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

                np.savez(self.file_name, class_obj=class_dict)

        # save last trajectory for debugging purposes
        self.b_traj_f = b_traj_f
        self.p_traj_f = p_traj_f

        # save data
        class_dict = vars(self)

        np.savez(self.file_name, class_obj=class_dict)
        

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
        b_lambda_mean_array = np.zeros((D))
        b_mig_mean_array = np.zeros((B))

        p_mean_array = np.zeros((D, P))
        p2_mean_array = np.zeros((D, P))
        p_lambda_mean_array = np.zeros((D))
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
                b_lambda_mean_array += np.einsum('di,ij,dj->d', b0, H, p0)

                p_mean_array += p0
                p2_mean_array += p0 ** 2
                pbar = np.mean(p0, axis=0).reshape((1, -1))  # island averaged abundance
                temp_rats = np.divide(pbar, p0)  # remove infinities
                temp_rats[~(np.isfinite(temp_rats))] = 0
                p_mig_mean_array += np.mean(temp_rats, axis=0)
                p_lambda_mean_array += np.einsum('di,ij,dj->d', p0, G, b0)

                
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
        b_lambda_mean_array *= 1 / count_short

        p_mean_array *= 1 / count_short
        p2_mean_array *= 1 / count_short
        p_mig_mean_array *= 1 / count_short
        p_lambda_mean_array *= 1 / count_short


        # Average and standard dev across islands.
        b_mean_ave = np.mean(b_mean_array, axis=0)
        b2_mean_ave = np.mean(b_mean_array, axis=0)
        b_mig_mean_ave = b_mig_mean_array
        b_lambda_mean_ave = np.mean(b_lambda_mean_array, axis=0)

        p_mean_ave = np.mean(p_mean_array, axis=0)
        p2_mean_ave = np.mean(p_mean_array, axis=0)
        p_mig_mean_ave = p_mig_mean_array
        p_lambda_mean_ave = np.mean(p_lambda_mean_array, axis=0)

        b_mean_std = np.std(b_mean_array, axis=0)
        b2_mean_std = np.std(b_mean_array, axis=0)
        b_lambda_mean_std = np.std(b_lambda_mean_array, axis=0)

        p_mean_std = np.std(p_mean_array, axis=0)
        p2_mean_std = np.std(p_mean_array, axis=0)
        p_lambda_mean_std = np.std(p_lambda_mean_array, axis=0)

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
        self.lambda_b_mean_ave_list.append(b_lambda_mean_ave)

        self.b_mean_std_list.append(b_mean_std)
        self.b2_mean_std_list.append(b2_mean_std)
        self.lambda_b_mean_std_list.append(b_lambda_mean_std)

        self.p_mean_ave_list.append(p_mean_ave)
        self.p2_mean_ave_list.append(p2_mean_ave)
        self.p_mig_mean_list.append(p_mig_mean_ave)
        self.eta_p_list.append(eta_p)
        self.lambda_p_mean_ave_list.append(p_lambda_mean_ave)

        self.p_mean_std_list.append(p_mean_std)
        self.p2_mean_std_list.append(p2_mean_std)
        self.lambda_p_mean_std_list.append(p_lambda_mean_std)



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

def sample_calculations(dt,sample_time,sample_time_start,sample_time_end):

    sample_num = int(round(sample_time/dt))
    sample_start = int(round(sample_time_start/dt))
    sample_end = int(round(sample_time_end/dt))
    sample_tvec = dt*np.arange(sample_start,sample_end,sample_num)

    return sample_num, sample_start, sample_end, sample_tvec

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

def define_deriv_many_islands_selective_diffs(V, S, N, u, m, normed):
    # :input V: interaction matrix
    # :input u: self interaction strength
    # :input m: migration rate
    # :input xbar: (D,K) log of nbar abundance
    # :input normed: logical bit for whether to normalize derivative or not
    # :output deriv: function of x that computes derivative
    # y = n/nbar

    def deriv(y, xbar, m):
        n = np.exp(xbar) * y
        D = np.shape(n)[0]
        growth_rate = S - u * (n) + np.einsum('ij,dj', V, n)

        if normed is True:
            stabilizing_term = np.einsum('di,di->d', n, growth_rate) / N  # normalization factor
            stabilizing_term = np.reshape(stabilizing_term, (D, 1))
        else:
            stabilizing_term = 0

        y_dot0 = y * (growth_rate - stabilizing_term)

        if m == 0:
            y_dot = y_dot0
        else:
            y_dot = y_dot0 + m * (1 - y)

        return y_dot, stabilizing_term

    return deriv

def define_deriv_many_islands(V,N,u,m,normed):
    # :input V: interaction matrix
    # :input u: self interaction strength
    # :input m: migration rate
    # :input xbar: (D,K) log of nbar abundance
    # :input normed: logical bit for whether to normalize derivative or not
    # :output deriv: function of x that computes derivative
    # y = n/nbar

    def deriv(y,xbar,m):
        n = np.exp(xbar)*y
        D = np.shape(n)[0]
        growth_rate = -u*(n) + np.einsum('dj,ij',n,V)
        
        if normed is True:
            norm_factor = np.einsum('di,di->d',n,growth_rate)/N  #normalization factor
            norm_factor = np.reshape(norm_factor,(D,1))
        else:
            norm_factor = 0

        y_dot0 = y*(growth_rate-norm_factor)
        
        if m==0:
            y_dot = y_dot0 
        else:
            y_dot = y_dot0  + m*(1-y)

        return y_dot

    return deriv

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


def define_deriv_infinite_islands(V,N,u,m,normed):
    # :input V: interaction matrix
    # :input u: self interaction strength
    # :input m: migration rate
    # :input xbar: (D,K) log of nbar abundance
    # :input normed: logical bit for whether to normalize derivative or not
    # :output deriv: function of x that computes derivative
    # y = n/nbar

    def deriv(y,xbar,xbar_inf,m):
        n = np.exp(xbar)*y
        D = np.shape(n)[0]
        growth_rate = -u*(n) + np.einsum('dj,ij',n,V)
        
        if normed is True:
            norm_factor = np.einsum('di,di->d',n,growth_rate)/N  #normalization factor
            norm_factor = np.reshape(norm_factor,(D,1))
        else:
            norm_factor = 0

        y_dot0 = y*(growth_rate-norm_factor)
        
        if m==0:
            y_dot = y_dot0 
        else:
            y_dot = y_dot0  + m*(np.exp(xbar_inf-xbar)-y)

        return y_dot

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


def adaptive_step(y0, xbar0, m, deriv, max_frac_change):
    # :input xbar0: (K,) vec of log island average
    # :input y0: (K,) vec of abundance/island average
    # :function deriv: gives derivative of y0.
    # :param max_frac_change: maximum change in abundance across islands.
    # :output y1: (K,) vec of new abundances (relative to island average exp(xbar0) )

    ydot, stabilizing_term = deriv(y0, xbar0, m)
    # freq_deriv = np.exp(xbar0)*ydot
    # dt = max_frac_change / np.max(np.abs(freq_deriv))  #Choose dt so that the maxo of exp(xbar0)*ydot* dt equals max_frac_change
    log_deriv = ydot[y0>0]/y0[y0>0]
    dt_pos = max_frac_change/np.max(log_deriv)  #from max of positive log_deriv
    dt_neg = -max_frac_change/np.min(log_deriv)
    dt = np.min([dt_pos,dt_neg])
    y1 = y0+ydot*dt

    return y1, xbar0, dt, stabilizing_term


def step_rk4_many_islands(y0,xbar0,m,dt,deriv):
    # :input x0: (K,) vec of current log variables
    # :param dt: step size
    # :function deriv: gives derivative of x. Function of x.
    # :output x1: (K,) vec of log variables after one time step.
    

    k1, _ = deriv(y0,xbar0,m)
    k2, _ = deriv(y0+(dt/2)*k1,xbar0,m)
    k3, _ = deriv(y0+(dt/2)*k2,xbar0,m)
    k4, _ = deriv(y0+dt*k3,xbar0,m)
    
    y1 = y0 + (dt/6)*(k1 + 2*k2 + 2*k3+k4)
    return y1, xbar0

def step_rk4_infinite_islands(y0,xbar0,xbar_inf,m,dt,deriv):
    # :input x0: (K,) vec of current log variables
    # :param dt: step size
    # :function deriv: gives derivative of x. Function of x.
    # :output x1: (K,) vec of log variables after one time step.
    

    k1 = deriv(y0,xbar0,xbar_inf,m)
    k2 = deriv(y0+(dt/2)*k1,xbar0,xbar_inf,m)
    k3 = deriv(y0+(dt/2)*k2,xbar0,xbar_inf,m)
    k4 = deriv(y0+dt*k3,xbar0,xbar_inf,m)
    
    y1 = y0 + (dt/6)*(k1 + 2*k2 + 2*k3+k4)
    return y1, xbar0


def antisym_etas(V,sig_A = None):
    """
    For interaction matrix close to antisymmetric, compute average noise <eta> as if matrix were antisymmetric
    :param V: Interaction matrix
    :param sig_A: standard deviation of elements A_{ij} if known beforehand
    :return: array of length K containing estimated eta values
    """
    A = (V-V.T)/2 # closest antisymmetric matrix. Normalization preserves magnitude of V almost antisymmetric.
    K = np.shape(A)[0] # total number of types

    p_star = find_fixed_pt(A)  # saturated fixed point
    K_alive = count_alive(p_star, 0.1/K**2)  # count surviving types

    # compute response integral chi to convert <n> to <eta>
    if sig_A==None:
        # estimate magnitude of A numerically
        chi = np.sqrt(K_alive*np.sum(np.dot(A, A.T))/(K**2))
    else:
        # use given scale, assume half survive
        chi = sig_A*np.sqrt(K_alive)


    etas = np.dot(A,p_star) # negative fitnesses of extinct types
    etas = etas+chi*p_star # resccaled <n> for alive types

    return etas

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

def count_alive(fs,min_freq=None):
    """
    Count number of types with non-zero abundance
    :param fs: array of frequencies
    :param min_freq: cutoff frequency for measuring alive types
    :return: Number of types with non-zero abundance
    """
    if min_freq==None:
        min_freq = 0.01/len(fs)**2
    return np.sum([f>min_freq for f in fs])
