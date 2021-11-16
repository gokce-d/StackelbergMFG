import time
import tensorflow as tf
import numpy as np
from tensorflow.python.training.moving_averages import assign_moving_average
from scipy.stats import multivariate_normal as normal
from tensorflow.python.ops import control_flow_ops
from tensorflow import random_normal_initializer as norm_init
from tensorflow import random_uniform_initializer as unif_init
from tensorflow import constant_initializer as const_init
from scipy.stats import uniform
import more_itertools as mit
import os


class SolverMKV1:
    #def __init__(self, sess, equation, T, Nt, x0_mean, x0_var, batch_size, valid_size, n_maxstep): # ---------- TF1
    def __init__(self, equation, T, m0, batch_size, valid_size, n_maxstep): # ---------- TF2
        # initial condition and horizon
        self.m0 = m0
        self.T = T

        # equation (of mkvequation class)
        self.equation = equation

        # parameters for neural network and gradient descent
        self.batch_size =       batch_size#64
        self.valid_size =       valid_size#128
        self.n_maxstep =        n_maxstep#3000#25000
        self.n_displaystep =    5#100 # frequency of printing messages out: every "n_displaystep" iterations
        # self.learning_rate =    1.e-2  #5e-4
        self.n_savetofilestep = 10#1000 # frequency of saving data: every "n_savetofilestep" iterations
        self.stdNN =            1.e-2
        self.lr_boundaries =    [200, 500] # FOR LRSCHEDULE
        self.lr_values =        [1e-2, 3e-3, 1e-3] # FOR LRSCHEDULE
        #self.n_units = 10
        self.activation_fn_choice = tf.nn.sigmoid # tf.nn.leaky_relu
        self.Nmax_iter = 100000

        # FOR NOW: WE FIX ALPHA's HERE. LATER: COMPUTED FROM OPTIMALITY CONDITION, RELATED TO X,rho,Z
        self.alpha_S_indiv = 1.0 # initialization
        self.alpha_I_pop = self.equation.inter_lambda_2(0.)
        self.Nstates = 3 # S I R for now
        self.basis = np.eye(self.Nstates) # identity matrix

        # neural net
        self.model_y0z =    self.MyModelY0Z(self.Nstates)

        # timing
        self.time_init = time.time()

    def sample_noise_one_step(self, n_samples):
        # Sample noises increments
        T_sample = np.random.exponential(1.0, size = (n_samples,1))
        return T_sample

    def sample_x0(self, n_samples): #TODO: TO BE MODIFIED TO REALLY SAMPLE !
        # Sample initial positions (got rid of the randomness)
        X0_sample = np.concatenate((np.tile(np.insert(np.zeros(self.Nstates-1),0,1), (int(n_samples*self.m0[0]),1)),np.tile(np.insert(np.zeros(self.Nstates-1),1,1), (int(n_samples*self.m0[1]),1))))
        return X0_sample        


    # ========== MODEL
    class MyModelY0Z(tf.keras.Model):
        def __init__(self, Nstates):#, miniminibatch_size):
            super(tf.keras.Model, self).__init__()
            # FOR INITIAL Y
            self.layer1_Y0 = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer2_Y0 = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            # self.layer3_Y0 = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            # self.layer4_Y0 = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer5_Y0 = tf.keras.layers.Dense(units=1)#(1, input_shape=(4,))#(units=1)
            # FOR Z
            self.layer1_Z = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer2_Z = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            # self.layer3_Z = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            # self.layer4_Z = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer5_Z = tf.keras.layers.Dense(units=Nstates)#(1, input_shape=(4,))#(units=1)

        def call_y0(self, input):
            result = self.layer1_Y0(input)
            result = self.layer2_Y0(result)
            # result = self.layer3_Y0(result)
            # result = self.layer4_Y0(result)
            result = self.layer5_Y0(result)
            return result

        def call_z(self, input):
            result = self.layer1_Z(input)
            result = self.layer2_Z(result)
            # result = self.layer3_Z(result)
            # result = self.layer4_Z(result)
            result = self.layer5_Z(result)
            return result



    def forward_pass(self, n_samples):
        start_time = time.time()
        # SAMPLE INITIAL POINTS
        self.X0 = tf.cast(self.sample_x0(n_samples), tf.dtypes.int64)

        # BUILD THE MKV SDE DYNAMICS
        # INITIALIZATION
        t =       0.0
        X =       self.X0
        Y =       self.model_y0z.call_y0(X) # initial value Y_0 = output of neural network taking X_0 as an input
        t_stack = tf.reshape(tf.tile([tf.cast(t, tf.float64)],  tf.stack([n_samples])), [n_samples, 1]) # vectors of time steps, need same size as X
        X_and_t = tf.concat([tf.cast(X, tf.float64), t_stack], 1) # concatenate vectors of current X and t
        Z =       self.model_y0z.call_z(X_and_t) # compute the value of Z as NN function of (x,t)
        P_empirical = np.mean(X, axis=0)
        self.alpha_S_indiv = self.equation.get_a_hat_S(P_empirical, n_samples, Z, self.equation.inter_lambda_2(t), X, t) # TO RECORD AND VISUALIZE

        # STORE SOLUTION
        self.Y_path =           Y # used to store the sequence of Y_t's
        self.X_path =           X # used to store the sequence of X_t's
        self.Z_path =           Z
        self.t_path =       tf.reshape(tf.cast(t, tf.dtypes.float64), (1,1))
        self.prop_S_path =  tf.reshape(P_empirical[0], (1,1))
        self.prop_I_path =  tf.reshape(P_empirical[1], (1,1))
        self.prop_R_path =  tf.reshape(P_empirical[2], (1,1))
        self.a_hat_S_path = tf.reshape(self.alpha_S_indiv, (1,1))

        # FIRST STEP IN TIME
        # STEP OF X
        rates = tf.reshape(self.equation.qrates(X, P_empirical, self.alpha_S_indiv, self.alpha_I_pop),[n_samples,1])
        T_sample = self.sample_noise_one_step(n_samples)
        holding_times_tmp = T_sample / rates
        condition = tf.less(tf.zeros((n_samples,1), tf.dtypes.float64), rates)
        holding_times = tf.where(condition, holding_times_tmp, (self.T+1)*tf.ones((n_samples,1), tf.dtypes.float64))
        i_star = tf.argmin(holding_times)
        i_star_int = int(tf.argmin(holding_times))
        updates = tf.reshape(mit.circular_shifts(X[i_star_int])[-1],(1, self.Nstates))
        X_next = tf.tensor_scatter_nd_update(X, tf.constant([[i_star_int]]), updates)
        Delta_t = holding_times[i_star[0]]
        
        
        # STEP OF M
        qmatrix = self.equation.qmatrix(X, P_empirical, self.alpha_S_indiv, self.alpha_I_pop)
        DeltaM = tf.cast(X_next - X, tf.dtypes.float64) - tf.linalg.matmul(tf.cast(X, tf.dtypes.float64), qmatrix)*Delta_t # for now rates is just a vector because at most one jump possible for each state
              
    
        # STEP OF Y
        Y_next = Y \
                - self.equation.driver_f_empirical(X, Z, P_empirical, self.equation.inter_lambda_2(t),t)*Delta_t \
                + tf.reshape(tf.reduce_sum( tf.multiply( Z, DeltaM ), 1 ), [n_samples,1])
        t += Delta_t
        
        
        
        # UPDATES
        X =             X_next
        Y =             Y_next
        P_empirical =   np.mean(X_next, axis=0)

       
    
    # LOOP IN TIME
        for i_t in range(1, self.Nmax_iter):
            # GET STATE AND USE NEURAL NETWORK
            t_stack = tf.reshape(tf.tile([tf.cast(t[0], tf.float64)],  tf.stack([n_samples])), [n_samples, 1]) # vectors of time steps, need same size as X
            
            X_and_t = tf.concat([tf.cast(X, tf.float64), t_stack], 1) # concatenate vectors of current X and t
      #      print("X_and_t:", X_and_t)
            Z =  self.model_y0z.call_z(X_and_t) # compute the value of Z as NN function of (x,t)
            self.alpha_S_indiv = self.equation.get_a_hat_S(P_empirical, n_samples, Z, self.equation.inter_lambda_2(t), X, t) # TO RECORD AND VISUALIZE
            #print("lambda_2: ", self.equation.inter_lambda_2(t))
            # STEP OF X
            rates = tf.reshape(self.equation.qrates(X, P_empirical, self.alpha_S_indiv, self.alpha_I_pop),[n_samples,1])
            T_sample = self.sample_noise_one_step(n_samples)
            holding_times_tmp = T_sample / rates
            condition = tf.less(tf.zeros((n_samples,1), tf.dtypes.float64), rates)
            holding_times = tf.where(condition, holding_times_tmp, (self.T+1)*tf.ones((n_samples,1), tf.dtypes.float64))

            i_star = tf.argmin(holding_times)
            i_star_int = int(tf.argmin(holding_times))
            updates = tf.reshape(mit.circular_shifts(X[i_star_int])[-1],(1, self.Nstates))
            X_next = tf.tensor_scatter_nd_update(X, tf.constant([[i_star_int]]), updates)
            
            Delta_t = holding_times[i_star[0]]
            t += Delta_t
            # UPDATE STATE
            if (t[0] < self.T):
                qmatrix = self.equation.qmatrix(X, P_empirical, self.alpha_S_indiv, self.alpha_I_pop)
                DeltaM = tf.cast(X_next - X, tf.dtypes.float64) - tf.linalg.matmul(tf.cast(X, tf.dtypes.float64), qmatrix)*Delta_t # for now rates is just a vector because at most one jump possible for each state
            
                # STEP OF Y
                Y_next = Y \
                        - self.equation.driver_f_empirical(X, Z, P_empirical, self.equation.inter_lambda_2(t), t)*Delta_t \
                        + tf.reshape(tf.reduce_sum( tf.multiply( Z, DeltaM ), 1 ), [n_samples,1])
                

                # STORE SOLUTION
                self.prop_S_path =      tf.concat([self.prop_S_path, tf.reshape(P_empirical[0], (1,1))], axis=1)
                self.prop_I_path =      tf.concat([self.prop_I_path, tf.reshape(P_empirical[1], (1,1))], axis=1)
                self.prop_R_path =      tf.concat([self.prop_R_path, tf.reshape(P_empirical[2], (1,1))], axis=1)
                self.t_path =      tf.concat([self.t_path, tf.reshape(t, (1,1))], axis=1)
                self.X_path =      tf.concat([self.X_path, X], axis=1)
                self.Y_path =      tf.concat([self.Y_path, Y], axis=1)
                self.Z_path =      tf.concat([self.Z_path, Z], axis=1)
                self.a_hat_S_path =      tf.concat([self.a_hat_S_path, tf.reshape(self.alpha_S_indiv,(1,1))],axis=0)
#                print("a_path: ", self.a_hat_S_path)          
                X =             X_next
                Y =             Y_next
                P_empirical =   np.mean(X_next, axis=0)
            else:
                break
            
        # print('============================')
        # print('==========End Res.==========')
        # print('============================')            
        # print("ahat: ", self.a_hat_S_path)        
        # print("Z: ",Z)
        # print("Y: ",Y)  
        # print("X: ",X)
        # print('')
        

        # COMPUTE ERROR
        target = self.equation.terminal_g_empirical(X, P_empirical)
        error = Y - target # penalization term
        self.loss = tf.reduce_mean(error ** 2)

        self.time_forward_pass = time.time() - start_time

    #def train(self):
    def train(self):
        print('========== START TRAINING ==========')
        start_time = time.time()
        # TRAIN OPERATIONS
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay( # FOR LRSCHEDULE
            self.lr_boundaries, self.lr_values) # FOR LRSCHEDULE
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule) # ---------- TF2 # FOR LRSCHEDULE

        checkpoint_directory = "checkpoints/" #+ datetime.datetime.now().strftime("%Y:%m:%d-%H.%M")
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         model=self.model_y0z,
                                         # optimizer_step=tf.train.get_or_create_global_step())
                                         optimizer_step=tf.compat.v1.train.get_or_create_global_step()) # TF2
        
        
        self.loss_history = []
        # self.dW_valid = self.sample_noise(self.valid_size) # sample noise increments for all time steps and the whole population -- for validation purposes (sampled only once) # ---------- TF2
        # self.X0_valid = self.sample_x0(self.valid_size) # sample initial positions for the whole population -- for validation purposes (sampled only once) # ---------- TF2

        # INITIALIZATION
        _ = self.forward_pass(self.valid_size) # ---------- TF2
        temp_loss = self.loss # ---------- TF2
        self.loss_history.append(temp_loss)
        step = 0
        print("step: %5u, loss: %.4e " % (step, temp_loss) + \
              "runtime: %4u s" % (time.time() - start_time + self.time_forward_pass))
        file = open("res-console.txt","a") # save what is printed to the console in a text file
        file.write("step: %5u, loss: %.4e " % (0, temp_loss) + \
            "runtime: %4u s" % (time.time() - start_time + self.time_forward_pass) + "\n")
        file.close()
        # a_hat_S = self.equation.get_a_hat_S(self.X_path[-1], tf.shape(self.X_path[-1])[0], self.Z_path[-1])
        # print "a_hat_S = {}".format(a_hat_S.numpy())
        datafile_name = 'data/data_fbsdesolver_solution_iter{}.npz'.format(step)
        np.savez(datafile_name,
                    prop_S_path = self.prop_S_path,
                    prop_I_path = self.prop_I_path,
                    prop_R_path = self.prop_R_path,
                    t_path = self.t_path,
                    a_hat_S_path = self.a_hat_S_path)

        # BEGIN SGD ITERATION
        for step in range(1, self.n_maxstep + 1):
            # print("=============================================")
            # print("==================STEP ", step, "==================")
            # print("=============================================")
            # AT EACH ITERATION: make a forward run to compute the gradient of the loss and update the NN
            # self.dW =  self.sample_noise(self.batch_size) # sample noise increments for all time steps and the whole population -- new samples at each training iteration # ---------- TF2
            # self.X0 =  self.sample_x0(self.batch_size) # sample initial positions for the whole population -- new samples at each training iteration # ---------- TF2
            with tf.GradientTape() as tape: # use tape to compute the gradient; see below
                predicted = self.forward_pass(self.batch_size)
                curr_loss = self.loss
            # Compute gradient w.r.t. parameters of NN:
            grads = tape.gradient(curr_loss, self.model_y0z.variables)
            # Make one SGD step:
            optimizer.apply_gradients(zip(grads, self.model_y0z.variables))#, global_step=tf.train.get_or_create_global_step())

            # PRINT RESULTS TO SCREEN AND OUTPUT FILE
            if step == 1 or step % self.n_displaystep == 0:
                # self.dW = self.dW_valid
                # self.X0 = self.X0_valid
                _ = self.forward_pass(self.valid_size) # ---------- TF2
                temp_loss = self.loss # ---------- TF2
                self.loss_history.append(temp_loss)
                print("step: %5u, loss: %.4e " % (step, temp_loss) + \
                    "runtime: %4u s" % (time.time() - start_time + self.time_forward_pass))
                file = open("res-console.txt","a")
                file.write("step: %5u, loss: %.4e " % (step, temp_loss) + \
                    "runtime: %4u s" % (time.time() - start_time + self.time_forward_pass) + "\n")
                file.close()
                # a_hat_S = self.equation.get_a_hat_S(self.X_path[-1], tf.shape(self.X_path[-1])[0], self.Z_path[-1])
                # print "a_hat_S = {}".format(a_hat_S.numpy())
                if (step==1 or step % self.n_savetofilestep == 0):
                    # y_real =        self.Y_path
                    # x_path =        self.X_path
                    datafile_name = 'data/data_fbsdesolver_solution_iter{}.npz'.format(step)
                    np.savez(datafile_name,
                                prop_S_path = self.prop_S_path,
                                prop_I_path = self.prop_I_path,
                                prop_R_path = self.prop_R_path,
                                t_path = self.t_path,
                                a_hat_S_path = self.a_hat_S_path,
                                loss_history = self.loss_history)
            # SAVE RESULTS TO DATA FILE
            elif (step % self.n_savetofilestep == 0):
                print("SAVING TO DATA FILE...")
                # self.dW = self.dW_valid
                # self.X0 = self.X0_valid
                _ = self.forward_pass(self.valid_size) # ---------- TF2
                temp_loss = self.loss # ---------- TF2
                self.loss_history.append(temp_loss)
                # y_real =        self.Y_path
                # x_path =        self.X_path
                datafile_name = 'data/data_fbsdesolver_solution_iter{}.npz'.format(step)
                np.savez(datafile_name,
                            prop_S_path = self.prop_S_path,
                            prop_I_path = self.prop_I_path,
                            prop_R_path = self.prop_R_path,
                            t_path = self.t_path,
                            a_hat_S_path = self.a_hat_S_path,
                            loss_history = self.loss_history) 
        # COLLECT THE RESULTS
        datafile_name = 'data/data_fbsdesolver_solution_iter-final.npz'
        np.savez(datafile_name,
                    prop_S_path = self.prop_S_path,
                    prop_I_path = self.prop_I_path,
                    prop_R_path = self.prop_R_path,
                    t_path = self.t_path,
                    a_hat_S_path = self.a_hat_S_path,
                    loss_history = self.loss_history)
                    # y_real=y_real, x_path=x_path, loss_history=self.loss_history)
        end_time = time.time()
        print("running time: %.3f s" % (end_time - self.time_init))
        print('========== END TRAINING ==========')
