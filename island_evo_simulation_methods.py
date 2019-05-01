import numpy as np
import math
import scipy.signal
import time
import scipy.optimize as opt

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

    def __init__(self, file_name, D, K, M, gamma, thresh, inv_fac, dt, mu, seed, epoch_timescale, epoch_num, sample_num):

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

        self.SetParams()

        self.EvoSim()

    def SetParams(self):

        D = self.D
        K = self.K


        self.N = 1
        self.m = np.exp(-self.M)

        self.K_tot = self.K + self.epoch_num * self.mu  # total number of types through simulation

        np.random.seed(seed=self.seed)

        self.V = generate_interactions_with_diagonal(self.K_tot, self.gamma) # generate full interaction matrix at start

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
        V = self.V[np.ix_(np.arange(0,self.K),np.arange(0,self.K))] # pick right subset of V
        n0, n_traj_eq = self.evo_step(V,n0,1) # equilibration dynamics
        self.n_traj_eq = n_traj_eq  # save first trajectory for debugging purposes
        # evolution
        for i in range(2,self.epoch_num+2):
            V,n0 = self.mut_step(n0,i) # add new types
            n0,n_traj_f = self.evo_step(V,n0,i) # dynamics

        # save last trajectory for debugging purposes
        self.n_traj_f = n_traj_f

        # save data
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
        t0 = np.sum(self.epoch_time_list[0:cur_epoch])  # initial time, used for calculation of extinction time

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

        print(cur_epoch)
        ######## end of current epoch
        return nf, n_traj

    def mut_step(self,n0,cur_epoch):
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
        V = self.V[np.ix_(n_alive,n_alive)] # active interactions

        return V,n0_new

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


def step_rk4_many_islands(y0,xbar0,m,dt,deriv):
    # :input x0: (K,) vec of current log variables
    # :param dt: step size
    # :function deriv: gives derivative of x. Function of x.
    # :output x1: (K,) vec of log variables after one time step.
    

    k1 = deriv(y0,xbar0,m)
    k2 = deriv(y0+(dt/2)*k1,xbar0,m)
    k3 = deriv(y0+(dt/2)*k2,xbar0,m)
    k4 = deriv(y0+dt*k3,xbar0,m)
    
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