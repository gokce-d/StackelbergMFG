import tensorflow as tf
import numpy as np


class MckeanVlasovEquation(object):
    def __init__(self, beta, gamma, kappa, inter_lambda_1, inter_lambda_2, lambda3, cost_I, cost_lambda1):
        # parameters of the MFG
        self.beta = beta
        self.gamma = gamma
        self.inter_lambda_1 = inter_lambda_1
        self.inter_lambda_2 = inter_lambda_2
        self.lambda3 = lambda3
        self.cost_I = cost_I # penalty for being in I
        self.cost_lambda1 = cost_lambda1
        self.kappa = kappa


    def get_prop_S(self, P_empirical):
        prop_S = tf.cast(P_empirical[0],tf.dtypes.float64)
        return prop_S

    def get_prop_I(self, P_empirical):
        prop_I = tf.cast(P_empirical[1],tf.dtypes.float64)
        return prop_I

    def get_prop_R(self, P_empirical):
        prop_R = tf.cast(P_empirical[2],tf.dtypes.float64)
        return prop_R

    def qrates(self, X, P_empirical, alpha_S_indiv, alpha_I_pop):
        #n_samples = tf.shape(X)[0]
        prop_I = self.get_prop_I(P_empirical)
        rate01 = self.beta * prop_I * alpha_S_indiv * alpha_I_pop
        rate12 = self.gamma
        rate20 = self.kappa
        cond0 = tf.reduce_all(tf.equal(X,[1,0,0]),1)
        cond1 = tf.reduce_all(tf.equal(X,[0,1,0]),1)
        return tf.where(cond0, rate01*tf.ones((1), tf.dtypes.float64), tf.where(cond1, rate12*tf.ones((1), tf.dtypes.float64), rate20*tf.ones((1), tf.dtypes.float64)))

    def qmatrix(self, X, P_empirical, alpha_S_indiv, alpha_I_pop):
        n_samples = tf.shape(X)[0]   
        prop_I = self.get_prop_I(P_empirical)
        qmatrix = tf.constant([[-float(self.beta*prop_I * alpha_S_indiv * alpha_I_pop), float(self.beta*prop_I * alpha_S_indiv * alpha_I_pop), 0.],
                              [0., -self.gamma, self.gamma],
                              [self.kappa, 0., -self.kappa]], dtype=tf.float64)
        return qmatrix
    
    
    def get_a_hat_S(self, P_empirical, n_samples, Z, alpha_I_pop, X, t):
        lambda1 = self.inter_lambda_1(t)
        return lambda1 + (1./self.cost_lambda1)*self.beta*self.get_prop_I(P_empirical)*alpha_I_pop*self.get_dZ_01(Z, X)

    def get_dZ_01(self,Z, X):
        # return tf.reshape(Z[:,1] - Z[:,0], [n_samples,1]) # OR THE OPPOSITE??
        cond0 = tf.reduce_all(tf.equal(X,[1,0,0]),1)
        first_susc = int(min(tf.where(cond0), default=0))
        #print("susc index", first_susc)
        #print("Z: ", Z[:,0])       
        return (Z[first_susc,0] - Z[first_susc,1]) # OR THE OPPOSITE??

    def driver_f_empirical(self, X, Z, P_empirical, alpha_I_pop, t):
        # driver of the BSDE
        lambda1 = self.inter_lambda_1(t)        
        n_samples = tf.shape(X)[0]
        a_hat_S = self.get_a_hat_S(P_empirical, n_samples, Z, alpha_I_pop, X, t)
        cond0 = tf.reduce_all(tf.equal(X,[1,0,0]),1)
        cond1 = tf.reduce_all(tf.equal(X,[0,1,0]),1)
        cost_S = tf.where(cond0, 0.5*self.cost_lambda1*(lambda1 - a_hat_S)**2, tf.zeros((1), tf.dtypes.float64))
        cost_I = tf.where(cond1, self.cost_I, tf.zeros((1), tf.dtypes.float64))
        return tf.reshape(cost_S + cost_I,(n_samples,1))

    def terminal_g_empirical(self, X, P_empirical):
        return 0.0
