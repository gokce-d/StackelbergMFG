import tensorflow as tf
import numpy as np
import more_itertools as mit





class MckeanVlasovEquation(object):
   # def __init__(self, beta, gamma, kappa, lambda1, lambda2, lambda3, cost_I, cost_lambda1):
    def __init__(self, beta, gamma, LAM, delta, eta, cost_I, cost_lambda1, c_sq, minorcost_D, regcost_D, beta_reg, lambda_goal):
        # parameters of the MFG
        self.beta = beta
        self.gamma = gamma
        self.LAM = LAM
        self.delta = delta
        self.eta = eta
        self.cost_I = cost_I # penalty for being in I
        self.minorcost_D = minorcost_D
        self.regcost_D = regcost_D
        self.cost_lambda1 = cost_lambda1
        self.c_sq = c_sq
        self.beta_reg = beta_reg
        self.lambda_goal = lambda_goal


    def qrates(self, X, T_sample, P_empirical, alpha_S_indiv, alpha_I_pop, T, Nstates):
        n_samples = tf.shape(X)[0]
        T_normal_sample = np.random.uniform(0.,1., size = (n_samples,1))
        rateSE = self.beta * P_empirical[2] * alpha_S_indiv * alpha_I_pop
        rateEI = self.LAM
        rateRS = self.eta
        rateIR = self.gamma
        rateID = self.delta
        rateRS = self.eta
        condS = tf.reduce_all(tf.equal(X,[1,0,0,0,0]),1)
        condE = tf.reduce_all(tf.equal(X,[0,1,0,0,0]),1)
        condI_R = tf.reduce_all(tf.equal(X,[0,0,1,0,0]) & np.less(T_normal_sample, rateIR/(rateIR+rateID)),1)
        condI_D = tf.reduce_all(tf.equal(X,[0,0,1,0,0]) & np.greater(T_normal_sample, rateIR/(rateIR+rateID)),1)
        condD = tf.reduce_all(tf.equal(X,[0,0,0,1,0]),1)
        condR = tf.reduce_all(tf.equal(X,[0,0,0,0,1]),1)
        
        rates = tf.where(condS, rateSE*tf.ones((1), tf.dtypes.float64), \
                         tf.where(condE, rateEI*tf.ones((1), tf.dtypes.float64), \
                                  tf.where(condI_R, rateIR*tf.ones((1), tf.dtypes.float64), \
                                           tf.where(condI_D, rateID*tf.ones((1), tf.dtypes.float64), \
                                                    tf.where(condR, rateRS*tf.ones((1), tf.dtypes.float64), \
                                                             0*tf.ones((1), tf.dtypes.float64))))))
    
        holding_times_tmp = np.reshape(T_sample,(n_samples,)) / rates  
        condition = tf.less(tf.zeros((n_samples,), tf.dtypes.float64), rates)
        holding_times = tf.where(condition, holding_times_tmp, (T+1)*tf.ones((n_samples,), tf.dtypes.float64))
        i_star = tf.argmin(holding_times)
        i_star_int = int(i_star)
        if condI_R[i_star_int]:
            updates = tf.reshape(mit.circular_shifts(X[i_star_int])[-2],(1, Nstates))
            X_next = tf.tensor_scatter_nd_update(X, tf.constant([[i_star_int]]), updates)
        else:             
            updates = tf.reshape(mit.circular_shifts(X[i_star_int])[-1],(1, Nstates))
            X_next = tf.tensor_scatter_nd_update(X, tf.constant([[i_star_int]]), updates)
        return (X_next,  holding_times[i_star_int])
            

    def qmatrix(self, X, P_empirical, alpha_S_indiv, alpha_I_pop):
        n_samples = tf.shape(X)[0]
        qmatrix = tf.constant([[-float(self.beta*P_empirical[2] * alpha_S_indiv * alpha_I_pop), float(self.beta*P_empirical[2] * alpha_S_indiv * alpha_I_pop), 0., 0., 0.],
                              [0., -self.LAM, self.LAM, 0., 0.],
                              [0., 0., -(self.gamma+self.delta), self.gamma, self.delta],
                              [0., 0., 0., 0., 0.],
                              [self.eta, 0., 0., -self.eta, 0.]], dtype=tf.float64)
        return qmatrix

    def get_a_hat_S(self, P_empirical, n_samples, Z, alpha_I_pop, X, LAMBDA):
        return LAMBDA[0] + (1./self.cost_lambda1)*self.beta*P_empirical[2]*alpha_I_pop*self.get_dZ_02(Z, X)

    def get_dZ_02(self,Z, X):
        # return tf.reshape(Z[:,1] - Z[:,0], [n_samples,1]) # OR THE OPPOSITE??
        cond0 = tf.reduce_all(tf.equal(X,[1,0,0,0,0]),1)
        # first_susc = int(min(tf.where(cond0), default=0))
        first_susc = int(tf.math.reduce_min(tf.where(cond0)))
        #print("susc index", first_susc)
        #print("Z: ", Z[:,0])
        return (Z[first_susc,0] - Z[first_susc,2]) # OR THE OPPOSITE??

    def driver_f_empirical(self, X, Z, P_empirical, alpha_I_pop, LAMBDA):
        # driver of the BSDE
        n_samples = tf.shape(X)[0]
        a_hat_S = self.get_a_hat_S(P_empirical, n_samples, Z, alpha_I_pop, X, LAMBDA)
        condS = tf.reduce_all(tf.equal(X,[1,0,0,0,0]),1)
        # condE = tf.reduce_all(tf.equal(X,[0,1,0,0,0]),1)
        condI = tf.reduce_all(tf.equal(X,[0,0,1,0,0]),1)
        condD = tf.reduce_all(tf.equal(X,[0,0,0,1,0]),1)
        # condR = tf.reduce_all(tf.equal(X,[0,0,0,0,1]),1)
        
        cost_S = tf.where(condS, 0.5*self.cost_lambda1*(LAMBDA[0] - a_hat_S)**2, tf.zeros((1), tf.dtypes.float64))
        cost_I = tf.where(condI, self.cost_I, tf.zeros((1), tf.dtypes.float64))
        cost_D = tf.where(condD, self.minorcost_D, tf.zeros((1), tf.dtypes.float64))
        return tf.reshape(cost_S + cost_I + cost_D,(n_samples,1))

    def terminal_g_empirical(self, X, P_empirical):
        return 0.0

    def reg_driver_f_empirical(self, LAMBDA, P_empirical):
        return 0.5 * self.c_sq * (P_empirical[1]**2) + 0.5 * self.beta_reg[0] * ((LAMBDA[0] - self.lambda_goal[0])**2) + \
    0.5 * self.beta_reg[1] * ((LAMBDA[1] - self.lambda_goal[1])**2) + \
    0.5 * self.beta_reg[2] * ((LAMBDA[2] - self.lambda_goal[2])**2) 


    def reg_terminal_g_empirical(self, P_empirical):
        return self.regcost_D* P_empirical[3]
