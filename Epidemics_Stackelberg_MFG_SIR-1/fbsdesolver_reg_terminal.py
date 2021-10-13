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
        self.n_savetofilestep = 10#1000 # frequency of saving data: every "n_savetofilestep" iterations
        self.stdNN =            1.e-2
        self.lr_boundaries =    [100, 200] # FOR LRSCHEDULE
        self.lr_values =        [1e-4, 1e-6, 1e-6] # FOR LRSCHEDULE
        self.activation_fn_choice = tf.nn.sigmoid # tf.nn.leaky_relu
        self.Nmax_iter = 100000

        # FOR NOW: WE FIX ALPHA's HERE. LATER: COMPUTED FROM OPTIMALITY CONDITION, RELATED TO X,rho,Z
        self.alpha_S_indiv = 1.0 # initialization
        self.Nstates = 3 # S I R for now
        self.basis = np.eye(self.Nstates) # identity matrix

        # neural net
        self.model_y0zlambda =    self.MyModelY0ZLAMBDA(self.Nstates)

        # timing
        self.time_init = time.time()

    def sample_noise_one_step(self, n_samples):
        # Sample noises increments
        T_sample = np.random.exponential(1.0, size = (n_samples,1))
        return T_sample

    def sample_x0(self, n_samples): #TODO: TO BE MODIFIED TO REALLY SAMPLE !
        # Sample initial positions
        # X0_sample = np.insert(np.random.multinomial(1, self.m0, size=n_samples-1), 0, np.insert(np.zeros(self.Nstates-1),1,1),axis=0)
        # Deterministic version
        X0_sample = np.concatenate((np.tile(np.insert(np.zeros(self.Nstates-1),0,1), (int(n_samples*self.m0[0]),1)),np.tile(np.insert(np.zeros(self.Nstates-1),1,1), (int(n_samples*self.m0[1]),1))))
        return X0_sample


    # ========== MODEL
    class MyModelY0ZLAMBDA(tf.keras.Model):
        def __init__(self, Nstates):#, miniminibatch_size):
            super(tf.keras.Model, self).__init__()
            # FOR INITIAL Y
            self.layer1_Y0 = tf.keras.layers.Dense(units=4, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer2_Y0 = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer3_Y0 = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer4_Y0 = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer5_Y0 = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer6_Y0 = tf.keras.layers.Dense(units=1, activation ='relu')#(1, input_shape=(4,))#(units=1)
            # FOR Z
            self.layer1_Z = tf.keras.layers.Dense(units=4, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer2_Z = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer3_Z = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer4_Z = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer5_Z = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer6_Z = tf.keras.layers.Dense(units=Nstates)#(1, input_shape=(4,))#(units=1)
            # For LAMBDA
            self.layer1_LAMBDA = tf.keras.layers.Dense(units=4, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer2_LAMBDA = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer3_LAMBDA = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer4_LAMBDA = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer5_LAMBDA = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer6_LAMBDA = tf.keras.layers.Dense(units=Nstates-1, activation='sigmoid')#(1, input_shape=(4,))#(units=1)

        def call_y0(self, input):
            result = self.layer1_Y0(input)
            result = self.layer2_Y0(result)
            result = self.layer3_Y0(result)
            result = self.layer4_Y0(result)
            result = self.layer5_Y0(result)
            result = self.layer6_Y0(result)
            return -1*result

        def call_z(self, input):
            result = self.layer1_Z(input)
            result = self.layer2_Z(result)
            result = self.layer3_Z(result)
            result = self.layer4_Z(result)
            result = self.layer5_Z(result)
            result = self.layer6_Z(result)
            return result

        def call_lambda(self, input):
            result = self.layer1_LAMBDA(input)
            result = self.layer2_LAMBDA(result)
            result = self.layer3_LAMBDA(result)
            result = self.layer4_LAMBDA(result)
            result = self.layer5_LAMBDA(result)
            result = self.layer6_LAMBDA(result)
            return result

    def forward_pass(self, n_samples):
        start_time = time.time()
        # SAMPLE INITIAL POINTS
        self.X0 = tf.cast(self.sample_x0(n_samples), tf.dtypes.int64)

        # BUILD THE MKV SDE DYNAMICS
        # INITIALIZATION
        t =       0.0
        X =       self.X0
        Y =       self.model_y0zlambda.call_y0(X) # initial value Y_0 = output of neural network taking X_0 as an input
        self.Y0 = np.mean(Y)
        t_stack = tf.reshape(tf.tile([tf.cast(t, tf.float64)],  tf.stack([n_samples])), [n_samples, 1]) # vectors of time steps, need same size as X
        X_and_t = tf.concat([tf.cast(X, tf.float64), t_stack], 1) # concatenate vectors of current X and t
        Z =       self.model_y0zlambda.call_z(X_and_t) # compute the value of Z as NN function of (x,t)
        P_empirical = np.mean(X, axis=0)
#        P_empirical_and_t = tf.concat([tf.cast(tf.reshape(P_empirical, [1, 3]), tf.float64), tf.cast(tf.reshape(t,[1,1]),tf.float64)], 1)
        LAMBDA = self.model_y0zlambda.call_lambda(tf.cast(tf.reshape(t,[1,1]),tf.float64))[0]
        self.alpha_I_pop = LAMBDA[1]
        self.alpha_S_indiv = self.equation.get_a_hat_S(P_empirical, n_samples, Z, self.alpha_I_pop, X, LAMBDA) # TO RECORD AND VISUALIZE


        # STORE SOLUTION
        self.Y_path =           Y # used to store the sequence of Y_t's
        self.X_path =           X # used to store the sequence of X_t's
        self.Z_path =           Z
        self.LAMBDA_path  =     tf.reshape(LAMBDA, (2,1))
        self.t_path       =  tf.reshape(tf.cast(t, tf.dtypes.float64), (1,1))
        self.prop_S_path  =  tf.reshape(tf.cast(P_empirical[0],tf.dtypes.float64), (1,1))
        self.prop_I_path  =  tf.reshape(tf.cast(P_empirical[1],tf.dtypes.float64), (1,1))
        self.prop_R_path  =  tf.reshape(tf.cast(P_empirical[2],tf.dtypes.float64), (1,1))
        self.a_hat_S_path =  tf.reshape(self.alpha_S_indiv, (1,1))

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
                - self.equation.driver_f_empirical(X, Z, P_empirical, self.alpha_I_pop, LAMBDA)*Delta_t \
                + tf.reshape(tf.reduce_sum( tf.multiply( Z, DeltaM ), 1 ), [n_samples,1])
        t += Delta_t


        # STEP OF REG
        reg_cost = self.equation.reg_driver_f_empirical(LAMBDA, P_empirical) * Delta_t


        # UPDATES
        X =             X_next
        Y =             Y_next
        P_empirical =   np.mean(X_next, axis=0)



        # LOOP IN TIME
        # print("P_empirical[1] = ", P_empirical[1])
        for i_t in range(1, self.Nmax_iter):
            if(P_empirical[1]<1.e-6): # TO AVOID REACHING 0 INFECTED AND HAVING NaN
                print("INNER P_empirical[1]<1.e-6 = ", P_empirical[1], "\t t = ", t[0].numpy())
                break
            self.nsteps_t = i_t
            # GET STATE AND USE NEURAL NETWORK
            t_stack = tf.reshape(tf.tile([tf.cast(t[0], tf.float64)],  tf.stack([n_samples])), [n_samples, 1]) # vectors of time steps, need same size as X
            X_and_t = tf.concat([tf.cast(X, tf.float64), t_stack], 1) # concatenate vectors of current X and t
      #      print("X_and_t:", X_and_t)

            Z =  self.model_y0zlambda.call_z(X_and_t) # compute the value of Z as NN function of (x,t)
            LAMBDA = self.model_y0zlambda.call_lambda(tf.cast(tf.reshape(t, [1,1]),tf.float64))[0]
            self.alpha_I_pop = LAMBDA[1]
            self.alpha_S_indiv = self.equation.get_a_hat_S(P_empirical, n_samples, Z, self.alpha_I_pop, X, LAMBDA) # TO RECORD AND VISUALIZE
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
            # print("t = {}, \t P_empirical[1] = {}".format(t[0], P_empirical[1]))
            if (t[0] < self.T):
                # if(P_empirical[1]<1.e-6): # TO AVOID REACHING 0 INFECTED AND HAVING NaN
                #     print("INNER P_empirical[1]<1.e-6 = %.4e, \t t = %.4e " %(P_empirical[1], t[0].numpy()))
                #     break
                # else:
                qmatrix = self.equation.qmatrix(X, P_empirical, self.alpha_S_indiv, self.alpha_I_pop)
                DeltaM = tf.cast(X_next - X, tf.dtypes.float64) - tf.linalg.matmul(tf.cast(X, tf.dtypes.float64), qmatrix)*Delta_t # for now rates is just a vector because at most one jump possible for each state

                # STEP OF Y
                Y_next = Y \
                        - self.equation.driver_f_empirical(X, Z, P_empirical, self.alpha_I_pop, LAMBDA)*Delta_t \
                        + tf.reshape(tf.reduce_sum( tf.multiply( Z, DeltaM ), 1 ), [n_samples,1])

#                 prop_S = self.equation.get_prop_S(P_empirical)
#                 prop_I = self.equation.get_prop_I(P_empirical)
#                 prop_R = self.equation.get_prop_R(P_empirical)
                # STORE SOLUTION
                self.prop_S_path =      tf.concat([self.prop_S_path, tf.reshape(tf.cast(P_empirical[0],tf.dtypes.float64), (1,1))], axis=1)
                self.prop_I_path =      tf.concat([self.prop_I_path, tf.reshape(tf.cast(P_empirical[1],tf.dtypes.float64), (1,1))], axis=1)
                self.prop_R_path =      tf.concat([self.prop_R_path, tf.reshape(tf.cast(P_empirical[2],tf.dtypes.float64), (1,1))], axis=1)
                self.t_path =      tf.concat([self.t_path, tf.reshape(t, (1,1))], axis=1)
                self.X_path =      tf.concat([self.X_path, X], axis=1)
                self.Y_path =      tf.concat([self.Y_path, Y], axis=1)
                self.Z_path =      tf.concat([self.Z_path, Z], axis=1)
                self.LAMBDA_path  =      tf.concat([self.LAMBDA_path, tf.reshape(LAMBDA, (2,1))], axis=1)
                self.a_hat_S_path =      tf.concat([self.a_hat_S_path, tf.reshape(self.alpha_S_indiv,(1,1))],axis=0)
#                print("a_path: ", self.a_hat_S_path)
                reg_cost += self.equation.reg_driver_f_empirical(LAMBDA, P_empirical) * Delta_t
                X =             X_next
                Y =             Y_next
                P_empirical =   np.mean(X_next, axis=0)
            else:
                break



        # COMPUTE ERROR
        self.avg_Y = np.mean(Y)
        reg_cost += self.equation.reg_driver_f_empirical(LAMBDA, P_empirical) * (self.T-t) + self.equation.reg_terminal_g_empirical(P_empirical) - tf.reduce_mean(Y)
#         target = self.equation.terminal_g_empirical(X, P_empirical)
#         error = Y - target # penalization term
        self.loss = reg_cost
#         self.reg_cost = reg_cost
#         self.target_cost = tf.reduce_mean(error ** 2)
#         print("reg cost: ", reg_cost)
#         print("empirical mean: ",P_empirical)
#         print("target match: ", self.target_importance * tf.reduce_mean(error ** 2))


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
                                         model=self.model_y0zlambda,
                                         # optimizer_step=tf.train.get_or_create_global_step())
                                         optimizer_step=tf.compat.v1.train.get_or_create_global_step()) # TF2


        self.loss_history = []
#         self.reg_cost_history = []
#         self.target_cost_history= []
        # self.dW_valid = self.sample_noise(self.valid_size) # sample noise increments for all time steps and the whole population -- for validation purposes (sampled only once) # ---------- TF2
        # self.X0_valid = self.sample_x0(self.valid_size) # sample initial positions for the whole population -- for validation purposes (sampled only once) # ---------- TF2

        # INITIALIZATION
        _ = self.forward_pass(self.valid_size) # ---------- TF2
        temp_loss = self.loss # ---------- TF2
        self.loss_history.append(temp_loss)
#         self.reg_cost_history.append(self.reg_cost)
#         temp_target_cost = self.target_cost
#         self.target_cost_history.append(temp_target_cost)
        step = 0
        print("step: %5u, loss: %.4e, Y: %.4e, Y0: %.4e " % (step, temp_loss, self.avg_Y, self.Y0) + \
            "runtime: %4u s" % (time.time() - start_time + self.time_forward_pass))
        file = open("res-console.txt","a")
        file.write("step: %5u, loss: %.4e, Y: %.4e, Y0: %.4e  " % (step, temp_loss,self.avg_Y, self.Y0) + \
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
                    a_hat_S_path = self.a_hat_S_path,
                    loss_history = self.loss_history,
                    LAMBDA_path  = self.LAMBDA_path,
                    Y_path       = self.Y_path)
#                     reg_cost_history = self.reg_cost_history,
#                     target_cost_history = self.target_cost_history)

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
            grads = tape.gradient(curr_loss, self.model_y0zlambda.variables)
            # Make one SGD step:
            # print("self.nsteps_t = ", self.nsteps_t, "  \t self.prop_I_path[-1] = ", self.prop_I_path[-1])
            # if (self.nsteps_t>1):
            optimizer.apply_gradients(zip(grads, self.model_y0zlambda.variables))#, global_step=tf.train.get_or_create_global_step())

            # PRINT RESULTS TO SCREEN AND OUTPUT FILE
            if step == 1 or step % self.n_displaystep == 0:
                # self.dW = self.dW_valid
                # self.X0 = self.X0_valid
                _ = self.forward_pass(self.valid_size) # ---------- TF2

                temp_loss = self.loss # ---------- TF2
                self.loss_history.append(temp_loss)
#                 self.reg_cost_history.append(self.reg_cost)
#                 temp_target_cost = self.target_cost
#                 self.target_cost_history.append(temp_target_cost)

                print("step: %5u, loss: %.4e, Y: %.4e, Y0: %.4e " % (step, temp_loss, self.avg_Y, self.Y0) + \
                    "runtime: %4u s" % (time.time() - start_time + self.time_forward_pass))
                file = open("res-console.txt","a")
                file.write("step: %5u, loss: %.4e, Y: %.4e, Y0: %.4e " % (step, temp_loss, self.avg_Y, self.Y0) + \
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
                                loss_history = self.loss_history,
                                LAMBDA_path  = self.LAMBDA_path,
                                Y_path       = self.Y_path)
#                                 reg_cost_history = self.reg_cost_history,
#                                 target_cost_history = self.target_cost_history)
            # SAVE RESULTS TO DATA FILE
            elif (step % self.n_savetofilestep == 0):
                print("SAVING TO DATA FILE...")
                # self.dW = self.dW_valid
                # self.X0 = self.X0_valid
                _ = self.forward_pass(self.valid_size) # ---------- TF2
                temp_loss = self.loss # ---------- TF2
                self.loss_history.append(temp_loss)
#                 self.reg_cost_history.append(self.reg_cost)
#                 temp_target_cost = self.target_cost
#                 self.target_cost_history.append(temp_target_cost)
                # y_real =        self.Y_path
                # x_path =        self.X_path
                datafile_name = 'data/data_fbsdesolver_solution_iter{}.npz'.format(step)
                np.savez(datafile_name,
                            prop_S_path = self.prop_S_path,
                            prop_I_path = self.prop_I_path,
                            prop_R_path = self.prop_R_path,
                            t_path = self.t_path,
                            a_hat_S_path = self.a_hat_S_path,
                            loss_history = self.loss_history,
                            LAMBDA_path  = self.LAMBDA_path,
                            Y_path       = self.Y_path)
#                             reg_cost_history = self.reg_cost_history,
#                             target_cost_history = self.target_cost_history)
        # COLLECT THE RESULTS
        datafile_name = 'data/data_fbsdesolver_solution_iter-final.npz'
        np.savez(datafile_name,
                    prop_S_path = self.prop_S_path,
                    prop_I_path = self.prop_I_path,
                    prop_R_path = self.prop_R_path,
                    t_path = self.t_path,
                    a_hat_S_path = self.a_hat_S_path,
                    loss_history = self.loss_history,
                    LAMBDA_path  = self.LAMBDA_path,
                    Y_path       = self.Y_path)
#                     reg_cost_history = self.reg_cost_history,
#                     target_cost_history = self.target_cost_history)
                    # y_real=y_real, x_path=x_path, loss_history=self.loss_history)
        end_time = time.time()
        print("running time: %.3f s" % (end_time - self.time_init))
        print('========== END TRAINING ==========')
