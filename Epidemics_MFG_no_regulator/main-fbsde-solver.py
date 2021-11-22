import mkvequation as mkvequation
import fbsdesolver as fbsdesolver
import tensorflow as tf
import numpy as np
from scipy.interpolate import interp1d


print('\n\ntf.VERSION = ', tf.__version__, '\n\n')
print('\n\ntf.keras.__version__ = ', tf.keras.__version__, '\n\n')

n_seed = 1 # random seed


def main():
    # parameters for neural network and gradient descent
    batch_size =    200
    valid_size =    400
    n_maxstep =     1500

    # problem parameters - see "Optimal incentives to mitigate epidemics: a Stackelberg mean field game approach" for interpretation
    T =             50.0 # time horizon
    Nt =            10000 # number of time steps
    t_grid =        np.linspace(0, T, Nt, endpoint=True) # discretized time 
    I0, R0 =        0.1, 0.0 #initial value for proportion of population in state I and R
    S0 =            1.0 - I0 - R0
    m0 =            [S0,I0,R0] # initial distribution of states
    beta =          0.25 # contact rate
    gamma =         1./10 # recovery rate
    kappa =         0.0 # revurnability rate
    lambda1 =       np.array([1.0, 0.7]) # Regulator policy in state S
    lambda2 =       np.array([0.9, 0.6]) # Regulator policy in state I
    lambda3 =       1.0 # Regulator policy in state R
    duration =      np.array([8000 ,Nt-8000])
    lambda_1 =      np.repeat(lambda1, duration.astype(int))
    inter_lambda_1 = interp1d(t_grid, lambda_1, kind="next")
    lambda_2 =      np.repeat(lambda2, duration.astype(int))
    inter_lambda_2 = interp1d(t_grid, lambda_2, kind="next")
    duration_1 =    np.array([])
    cost_I =        1.0
    cost_lambda1 =  10.0

    # save parameter date to file
    datafile_name = 'data/data_fbsdesolver_params.npz'
    np.savez(datafile_name,
                beta=beta, gamma=gamma, kappa=kappa, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
                cost_I=cost_I, cost_lambda1=cost_lambda1,
                m0=m0,
                batch_size=batch_size, valid_size=valid_size, n_maxstep=n_maxstep)
                
    # solve FBSDE with SolverMKV1 (fbsdesolver.py)
    tf.random.set_seed(n_seed) # ---------- TF2
    tf.keras.backend.set_floatx('float64') # change all layers to be of dtype float64 by default, useful when porting from TF 1.X to TF 2.0  # ---------- TF2
    print("========== BEGIN SOLVER MKV FBSDE ==========")
    print("================ PARAMETERS ================")
    print("particle #: ",batch_size, "time: ", T, "infected: ", I0, "beta: " , beta, "gamma: ", gamma, "kappa: ", kappa, "c_I: ", cost_I, "c_lambda: ", cost_lambda1, "max_step: ", n_maxstep)
    print("================ PARAMETERS ================")
    mkv_equation = mkvequation.MckeanVlasovEquation(beta, gamma, kappa, inter_lambda_1, inter_lambda_2, lambda3, cost_I, cost_lambda1)
    mkv_solver = fbsdesolver.SolverMKV1(mkv_equation, T, m0, batch_size, valid_size, n_maxstep) # ---------- TF2
    mkv_solver.train()
    
    # save result to file
    print("SAVING FILES...")
    datafile_name = 'data/data_fbsdesolver_solution_final.npz'
    np.savez(datafile_name,
                prop_S_path = mkv_solver.prop_S_path,
                prop_I_path = mkv_solver.prop_I_path,
                prop_R_path = mkv_solver.prop_R_path,
                t_path = mkv_solver.t_path)
    print("========== END SOLVER MKV FBSDE ==========")

# run program
if __name__ == '__main__':
    np.random.seed(n_seed)
    main()