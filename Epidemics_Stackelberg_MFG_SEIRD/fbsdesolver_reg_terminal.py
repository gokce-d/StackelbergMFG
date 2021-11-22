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

# This class is a solver for the McKean-Vlasov equation (defined in mkvequation_reg.py)
#
# Summary of methods
# 1. sample_noise_one_step(i): generates i samples of the increment of the noise process (jump times)
# 2. sample_x0(i): sets the initial states of the i players
# 3. forward_pass(i): time-steps the McKean-Vlasov equation (i copies coupled through the empirical distribution)
# 4. train(): trains the neural networks
class SolverMKV1:
    #def __init__(self, sess, equation, T, Nt, x0_mean, x0_var, batch_size, valid_size, n_maxstep): # ---------- TF1
    def __init__(self, equation, T, m0, batch_size, valid_size, n_maxstep): # ---------- TF2
        # parameters for neural network and gradient descent
        self.batch_size =       batch_size
        self.valid_size =       valid_size
        self.n_maxstep =        n_maxstep
        self.n_displaystep =    5 # frequency of printing messages out: every "n_displaystep" iterations
        self.n_savetofilestep = 10 # frequency of saving data: every "n_savetofilestep" iterations
        self.stdNN =            1.e-2
        self.lr_boundaries =    [100, 200] # parameter for lrshcedule
        self.lr_values =        [1e-3, 1e-4, 1e-5]# parameter for lrshcedule
        self.activation_fn_choice = tf.nn.sigmoid
        self.Nmax_iter =        100000

        # problem parameters - see "Optimal incentives to mitigate epidemics: a Stackelberg mean field game approach" for interpretation
        self.m0 =               m0 # initial condition
        self.T =                T # time horizon
        self.equation =         equation # equation (of mkvequation_reg class)
        self.alpha_S_indiv =    1.0 # first iteration contact factor value in S (fixed here, later computed from opt. cond. related to X, rho, Z)
        self.Nstates =          5 # (S, E, I, R, D)
        self.model_y0zlambda =  self.MyModelY0ZLAMBDA(self.Nstates) # neural nets

        self.time_init = time.time() # timing

    def sample_noise_one_step(self, n_samples):
        T_sample = np.random.exponential(1.0, size = (n_samples,1))
        return T_sample

    def sample_x0(self, n_samples):
        # Sample initial positions
        # X0_sample = np.insert(np.random.multinomial(1, self.m0, size=n_samples-1), 0, np.insert(np.zeros(self.Nstates-1),1,1),axis=0)
        # Deterministic version
        X0_sample = np.concatenate((np.tile(np.insert(np.zeros(self.Nstates-1),0,1), (int(n_samples*self.m0[0]),1)),np.tile(np.insert(np.zeros(self.Nstates-1),2,1), (int(n_samples*self.m0[2]),1))))
        return X0_sample

    # This class is the deep neural network model for Y_0, Z, and LAMBDA
    #
    # Summary of methods
    # 1. call_y0(v): evaluates the NN for Y_0 at the input
    # 2. call_z(v): ...
    # 3. call_lambda(v): ...
    class MyModelY0ZLAMBDA(tf.keras.Model):
        def __init__(self, Nstates):
            super(tf.keras.Model, self).__init__()
            # Y_0: neural network architecture
            self.layer1_Y0 = tf.keras.layers.Dense(units=4, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer2_Y0 = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer3_Y0 = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer4_Y0 = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer5_Y0 = tf.keras.layers.Dense(units=1, activation ='relu')

            # Z: neural netwrok architecture
            self.layer1_Z = tf.keras.layers.Dense(units=4, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer2_Z = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer3_Z = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer4_Z = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer5_Z = tf.keras.layers.Dense(units=Nstates)

            # LAMBDA: neural netwrok architecture
            self.layer1_LAMBDA = tf.keras.layers.Dense(units=4, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer2_LAMBDA = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer3_LAMBDA = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer4_LAMBDA = tf.keras.layers.Dense(units=100, activation='sigmoid', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5), bias_initializer='zeros')
            self.layer5_LAMBDA = tf.keras.layers.Dense(units=Nstates-2, activation='sigmoid')

        def call_y0(self, input):
            result = self.layer1_Y0(input)
            result = self.layer2_Y0(result)
            result = self.layer3_Y0(result)
            result = self.layer4_Y0(result)
            result = self.layer5_Y0(result)
            return -1*result 

        def call_z(self, input):
            result = self.layer1_Z(input)
            result = self.layer2_Z(result)
            result = self.layer3_Z(result)
            result = self.layer4_Z(result)
            result = self.layer5_Z(result)
            return result

        def call_lambda(self, input):
            result = self.layer1_LAMBDA(input)
            result = self.layer2_LAMBDA(result)
            result = self.layer3_LAMBDA(result)
            result = self.layer4_LAMBDA(result)
            result = self.layer5_LAMBDA(result)
            return result

    def forward_pass(self, n_samples):
        start_time = time.time() # to record run time

        self.X0 = tf.cast(self.sample_x0(n_samples), tf.dtypes.int64) # get initial values for X

        # forward_pass - Step 1. initialization, building the MVK dynamics
        t =       0.0 # initial time t: 0
        X =       self.X0 # initial X path: X0
        Y =       self.model_y0zlambda.call_y0(X) # initial Y_0-value: output of neural network taking X_0 as an input
        self.Y0 = np.mean(Y)
        t_stack = tf.reshape(tf.tile([tf.cast(t, tf.float64)],  tf.stack([n_samples])), [n_samples, 1]) # vectorizes the time steps (must match the size of X)
        X_and_t = tf.concat([tf.cast(X, tf.float64), t_stack], 1) # concatenate vectors of current X and t
        Z =       self.model_y0zlambda.call_z(X_and_t) # initial Z: the value NN model, a function of (X,t)
        P_empirical = np.mean(X, axis=0) # initial empirical distribution over states of infection
        LAMBDA = self.model_y0zlambda.call_lambda(tf.cast(tf.reshape(t,[1,1]),tf.float64))[0] # initial lambda
        self.alpha_I_pop = LAMBDA[2]
        self.alpha_E_pop = LAMBDA[1]
        self.alpha_S_indiv = self.equation.get_a_hat_S(P_empirical, n_samples, Z, self.alpha_I_pop, X, LAMBDA) # for recording and visualizing

        # forward_pass - Step 2. store solution
        self.Y_path =           Y # stores the discretized path (Y_t; t)
        self.X_path =           X # ...
        self.Z_path =           Z
        self.LAMBDA_path  =     tf.reshape(LAMBDA, (3,1))
        self.t_path       =  tf.reshape(tf.cast(t, tf.dtypes.float64), (1,1))
        self.prop_S_path  =  tf.reshape(tf.cast(P_empirical[0],tf.dtypes.float64), (1,1))
        self.prop_E_path  =  tf.reshape(tf.cast(P_empirical[1],tf.dtypes.float64), (1,1))
        self.prop_I_path  =  tf.reshape(tf.cast(P_empirical[2],tf.dtypes.float64), (1,1))
        self.prop_D_path  =  tf.reshape(tf.cast(P_empirical[3],tf.dtypes.float64), (1,1))
        self.prop_R_path  =  tf.reshape(tf.cast(P_empirical[4],tf.dtypes.float64), (1,1))
        self.a_hat_S_path =  tf.reshape(self.alpha_S_indiv, (1,1))

        # forward_pass - Step 3. first time step
            # 3.(i) step X
        T_sample = self.sample_noise_one_step(n_samples)
        Iter_results = self.equation.qrates(X, T_sample, P_empirical, self.alpha_S_indiv, self.alpha_I_pop, self.T, self.Nstates)
        X_next = tf.reshape(Iter_results[0],[n_samples,self.Nstates])
        Delta_t = Iter_results[1]

            # 3.(ii) step M
        qmatrix = self.equation.qmatrix(X, P_empirical, self.alpha_S_indiv, self.alpha_I_pop)
        DeltaM = tf.cast(X_next - X, tf.dtypes.float64) - tf.linalg.matmul(tf.cast(X, tf.dtypes.float64), qmatrix)*Delta_t # single vector because at most one jump possible for each state

            # 3.(iii) step Y
        Y_next = Y \
                - self.equation.driver_f_empirical(X, Z, P_empirical, self.alpha_I_pop, LAMBDA)*Delta_t \
                + tf.reshape(tf.reduce_sum( tf.multiply( Z, DeltaM ), 1 ), [n_samples,1])
        t += Delta_t

            # 3.(iv) step regulators cost
        reg_cost = self.equation.reg_driver_f_empirical(LAMBDA, P_empirical) * Delta_t

            # 3.(v) update solution
        X =             X_next
        Y =             Y_next
        P_empirical =   np.mean(X_next, axis=0)

        # forward_pass - Step 4. the rest of the time stepping
        for i_t in range(1, self.Nmax_iter):
            if(P_empirical[2]<1.e-6): # capped to avoid reaching 0 infected (resulting in NaNs)
                print("INNER P_empirical[2]<1.e-6 = ", P_empirical[2], "\t t = ", t.numpy())
                break
            self.nsteps_t = i_t
                # 4.(i) get state, Z (using NN model), LAMBDA, control (to record and visualize)
            t_stack = tf.reshape(tf.tile([tf.cast(t, tf.float64)],  tf.stack([n_samples])), [n_samples, 1]) # vectorizes time steps (must be same size as X)
            X_and_t = tf.concat([tf.cast(X, tf.float64), t_stack], 1) # concatenate vectors of current X and t
            Z =  self.model_y0zlambda.call_z(X_and_t) # compute the value of Z as NN function of (X,t)
            LAMBDA = self.model_y0zlambda.call_lambda(tf.cast(tf.reshape(t, [1,1]),tf.float64))[0]
            self.alpha_I_pop = LAMBDA[2]
            self.alpha_E_pop = LAMBDA[1]
            self.alpha_S_indiv = self.equation.get_a_hat_S(P_empirical, n_samples, Z, self.alpha_I_pop, X, LAMBDA) # to record and visualize
            
                # 4.(ii) step X
            T_sample = self.sample_noise_one_step(n_samples)
            Iter_results = self.equation.qrates(X, T_sample, P_empirical, self.alpha_S_indiv, self.alpha_I_pop, self.T, self.Nstates)
            X_next = tf.reshape(Iter_results[0],[n_samples,self.Nstates])
            
                # 4.(iii) step time
            Delta_t = Iter_results[1]
            t += Delta_t

                # 4.(iv) step M, Y, and store solution
            if (t < self.T):
                qmatrix = self.equation.qmatrix(X, P_empirical, self.alpha_S_indiv, self.alpha_I_pop)
                DeltaM = tf.cast(X_next - X, tf.dtypes.float64) - tf.linalg.matmul(tf.cast(X, tf.dtypes.float64), qmatrix)*Delta_t # one vector because at most one jump possible for each state

                Y_next = Y \
                        - self.equation.driver_f_empirical(X, Z, P_empirical, self.alpha_I_pop, LAMBDA)*Delta_t \
                        + tf.reshape(tf.reduce_sum( tf.multiply( Z, DeltaM ), 1 ), [n_samples,1])
                self.prop_S_path =      tf.concat([self.prop_S_path, tf.reshape(tf.cast(P_empirical[0],tf.dtypes.float64), (1,1))], axis=1)
                self.prop_E_path =      tf.concat([self.prop_E_path, tf.reshape(tf.cast(P_empirical[1],tf.dtypes.float64), (1,1))], axis=1)
                self.prop_I_path =      tf.concat([self.prop_I_path, tf.reshape(tf.cast(P_empirical[2],tf.dtypes.float64), (1,1))], axis=1)
                self.prop_D_path =      tf.concat([self.prop_D_path, tf.reshape(tf.cast(P_empirical[3],tf.dtypes.float64), (1,1))], axis=1)
                self.prop_R_path =      tf.concat([self.prop_R_path, tf.reshape(tf.cast(P_empirical[4],tf.dtypes.float64), (1,1))], axis=1)
                self.t_path =      tf.concat([self.t_path, tf.reshape(t, (1,1))], axis=1)
                self.X_path =      tf.concat([self.X_path, X], axis=1)
                self.Y_path =      tf.concat([self.Y_path, Y], axis=1)
                self.Z_path =      tf.concat([self.Z_path, Z], axis=1)
                self.LAMBDA_path  =      tf.concat([self.LAMBDA_path, tf.reshape(LAMBDA, (3,1))], axis=1)
                self.a_hat_S_path =      tf.concat([self.a_hat_S_path, tf.reshape(self.alpha_S_indiv,(1,1))],axis=0)
                reg_cost += self.equation.reg_driver_f_empirical(LAMBDA, P_empirical) * Delta_t
                X =             X_next
                Y =             Y_next
                P_empirical =   np.mean(X_next, axis=0)
            else:
                break

        # forward_pass - Step 5. compute the loss
        self.avg_Y = np.mean(Y)
        reg_cost += self.equation.reg_driver_f_empirical(LAMBDA, P_empirical) * (self.T-t) + self.equation.reg_terminal_g_empirical(P_empirical) - tf.reduce_mean(Y)
        self.loss = reg_cost

        self.time_forward_pass = time.time() - start_time # store run time

    def train(self):
        print('========== START TRAINING ==========')
        start_time = time.time() # to record run time

        # train - Step 1. training operations
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay( # for lrschedule
            self.lr_boundaries, self.lr_values) # for lrschedule
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule) # ---------- TF2 # for lrschedule
        checkpoint_directory = "checkpoints/"
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         model=self.model_y0zlambda,
                                         optimizer_step=tf.compat.v1.train.get_or_create_global_step()) # TF2
        self.loss_history = []
    
        # train - Step 2. initializaiton
        _ = self.forward_pass(self.valid_size) # ---------- TF2
        temp_loss = self.loss # ---------- TF2
        self.loss_history.append(temp_loss)
        step = 0
        print("step: %5u, loss: %.4e, Y: %.4e, Y0: %.4e " % (step, temp_loss, self.avg_Y, self.Y0) + \
            "runtime: %4u s" % (time.time() - start_time + self.time_forward_pass))
        file = open("res-console.txt","a")
        file.write("step: %5u, loss: %.4e, Y: %.4e, Y0: %.4e  " % (step, temp_loss,self.avg_Y, self.Y0) + \
            "runtime: %4u s" % (time.time() - start_time + self.time_forward_pass) + "\n")
        file.close()
        datafile_name = 'data/data_fbsdesolver_solution_iter{}.npz'.format(step)
        np.savez(datafile_name,
                    prop_S_path = self.prop_S_path,
                    prop_E_path = self.prop_E_path,
                    prop_I_path = self.prop_I_path,
                    prop_D_path = self.prop_D_path,
                    prop_R_path = self.prop_R_path,
                    t_path = self.t_path,
                    a_hat_S_path = self.a_hat_S_path,
                    loss_history = self.loss_history,
                    LAMBDA_path  = self.LAMBDA_path,
                    Y_path       = self.Y_path)

        # train - step 3. start SGD iteration
        for step in range(1, self.n_maxstep + 1):
            # print("=============================================")
            # print("==================STEP ", step, "==================")
            # print("=============================================")

                # 3.(i). make a forward run to compute the gradient of the loss and update the NN
            with tf.GradientTape() as tape: # use tape to compute the gradient; see below
                predicted = self.forward_pass(self.batch_size)
                curr_loss = self.loss

                # 3.(ii). compute gradient w.r.t. parameters of NN:
            grads = tape.gradient(curr_loss, self.model_y0zlambda.variables)
                
                # 3.(iii). take one SGD step:
            optimizer.apply_gradients(zip(grads, self.model_y0zlambda.variables))#, global_step=tf.train.get_or_create_global_step())

                # 3.(iv). print and write results
            if step == 1 or step % self.n_displaystep == 0:
                _ = self.forward_pass(self.valid_size) # ---------- TF2
                temp_loss = self.loss # ---------- TF2
                self.loss_history.append(temp_loss)
                print("step: %5u, loss: %.4e, Y: %.4e, Y0: %.4e " % (step, temp_loss, self.avg_Y, self.Y0) + \
                    "runtime: %4u s" % (time.time() - start_time + self.time_forward_pass))
                file = open("res-console.txt","a")
                file.write("step: %5u, loss: %.4e, Y: %.4e, Y0: %.4e " % (step, temp_loss, self.avg_Y, self.Y0) + \
                    "runtime: %4u s" % (time.time() - start_time + self.time_forward_pass) + "\n")
                file.close()
                if (step==1 or step % self.n_savetofilestep == 0):
                    datafile_name = 'data/data_fbsdesolver_solution_iter{}.npz'.format(step)
                    np.savez(datafile_name,
                                prop_S_path = self.prop_S_path,
                                prop_E_path = self.prop_E_path,
                                prop_I_path = self.prop_I_path,
                                prop_D_path = self.prop_D_path,
                                prop_R_path = self.prop_R_path,
                                t_path = self.t_path,
                                a_hat_S_path = self.a_hat_S_path,
                                loss_history = self.loss_history,
                                LAMBDA_path  = self.LAMBDA_path,
                                Y_path       = self.Y_path)

                # 3.(v). save results
            elif (step % self.n_savetofilestep == 0):
                print("SAVING TO DATA FILE...")
                _ = self.forward_pass(self.valid_size) # ---------- TF2
                temp_loss = self.loss # ---------- TF2
                self.loss_history.append(temp_loss)
                datafile_name = 'data/data_fbsdesolver_solution_iter{}.npz'.format(step)
                np.savez(datafile_name,
                            prop_S_path = self.prop_S_path,
                            prop_E_path = self.prop_E_path,
                            prop_I_path = self.prop_I_path,
                            prop_D_path = self.prop_D_path,
                            prop_R_path = self.prop_R_path,
                            t_path = self.t_path,
                            a_hat_S_path = self.a_hat_S_path,
                            loss_history = self.loss_history,
                            LAMBDA_path  = self.LAMBDA_path,
                            Y_path       = self.Y_path)

        # train - step 4. collect the results
        datafile_name = 'data/data_fbsdesolver_solution_iter-final.npz'
        np.savez(datafile_name,
                    prop_S_path = self.prop_S_path,
                    prop_E_path = self.prop_E_path,
                    prop_I_path = self.prop_I_path,
                    prop_D_path = self.prop_D_path,
                    prop_R_path = self.prop_R_path,
                    t_path = self.t_path,
                    a_hat_S_path = self.a_hat_S_path,
                    loss_history = self.loss_history,
                    LAMBDA_path  = self.LAMBDA_path,
                    Y_path       = self.Y_path)

        end_time = time.time() # record and print run time
        print("running time: %.3f s" % (end_time - self.time_init))
        print('========== END TRAINING ==========')
