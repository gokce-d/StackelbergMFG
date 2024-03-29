import tensorflow as tf
import numpy as np

# This class defines the McKean-Vlasov equation
#
# Summary of methods
# 1. get_prop_S(P): evaluates discrete distribution P in state S, casts to dtype float64
# 2. get_prop_I(P): ...
# 3. get_prop_R(P): ...
# 4. qrates(X, P, aS, aI_pop): calculates the transition rates for a player (X) when the population is distributed according to P, 
#                               aS is the players contact factor, and aI_pop is the infected populations contact factor control
# 5. qmatrix(X, P, aS, aI_pop): ...
# 6. get_a_hat_S(P, nrSamples, Z, aI_pop, X, t): returns contact factor control equilibrium function evaluated at the input
# 7. get_dZ_01(Z, X): returns increments of Z path
# 8. driver_f_empirical(X, Z, P, aI_pop, t): computes BSDE driver at input
# 9. terminal_g_empirical(X, P): conputes BSDE terminal condition at input
# 10. reg_driver_f_empirical(lambda, P): computes regulator BSDE driver at input
# 11. reg_terminal_g_empirical(P): computes regulator terminal cost at input
class MckeanVlasovEquation(object):
    def __init__(self, beta, gamma, kappa, cost_I, cost_lambda1, clog, beta_reg, lambda_goal):
        # equation parameters - see "Optimal incentives to mitigate epidemics: a Stackelberg mean field game approach" for interpretation
        self.beta = beta
        self.gamma = gamma
        self.cost_I = cost_I
        self.cost_lambda1 = cost_lambda1
        self.kappa = kappa
        self.clog = clog
        self.beta_reg = beta_reg
        self.lambda_goal = lambda_goal

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

    def get_a_hat_S(self, P_empirical, n_samples, Z, alpha_I_pop, X, LAMBDA):
        return LAMBDA[0] + (1./self.cost_lambda1)*self.beta*self.get_prop_I(P_empirical)*alpha_I_pop*self.get_dZ_01(Z, X)

    def get_dZ_01(self,Z, X):
        cond0 = tf.reduce_all(tf.equal(X,[1,0,0]),1)
        first_susc = int(tf.math.reduce_min(tf.where(cond0)))
        return (Z[first_susc,0] - Z[first_susc,1])

    def driver_f_empirical(self, X, Z, P_empirical, alpha_I_pop, LAMBDA):
        n_samples = tf.shape(X)[0]
        a_hat_S = self.get_a_hat_S(P_empirical, n_samples, Z, alpha_I_pop, X, LAMBDA)
        cond0 = tf.reduce_all(tf.equal(X,[1,0,0]),1)
        cond1 = tf.reduce_all(tf.equal(X,[0,1,0]),1)
        cost_S = tf.where(cond0, 0.5*self.cost_lambda1*(LAMBDA[0] - a_hat_S)**2, tf.zeros((1), tf.dtypes.float64))
        cost_I = tf.where(cond1, self.cost_I, tf.zeros((1), tf.dtypes.float64))
        return tf.reshape(cost_S + cost_I,(n_samples,1))

    def terminal_g_empirical(self, X, P_empirical):
        return 0.0

    def reg_driver_f_empirical(self, LAMBDA, P_empirical):
        return 0.5 * self.clog * (P_empirical[1]**2) + 0.5 * self.beta_reg[0] * ((LAMBDA[0] - self.lambda_goal[0])**2) + 0.5 * self.beta_reg[1] * ((LAMBDA[1] - self.lambda_goal[1])**2)

    def reg_terminal_g_empirical(self, P_empirical):
        return 0.0
