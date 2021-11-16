import mkvequation as mkvequation
import fbsdesolver as fbsdesolver
import tensorflow as tf
import numpy as np
from scipy.interpolate import interp1d


print('\n\ntf.VERSION = ', tf.__version__, '\n\n')
print('\n\ntf.keras.__version__ = ', tf.keras.__version__, '\n\n')

# SEED FOR RANDOMNESS
n_seed = 1


def main():
    # PARAMETERS
    # Nt =        20 #100
    batch_size = 200

    T =         50.0
    Nt      =    10000 # number of points
    t_grid  = np.linspace(0, T, Nt, endpoint=True)
    
    I0, R0 = 0.1, 0.0 #1.e-2, 0.0
    S0 = 1.0 - I0 - R0
    m0 =        [S0,I0,R0] # initial distribution
    beta =      0.25
    gamma =     1./10
    kappa =     0.0
    lambda1 =   np.array([1.0, 0.7])
    lambda2 =   np.array([0.9, 0.6])
    lambda3 =   1.0
    duration = np.array([8000 ,Nt-8000])
    
    lambda_1 = np.repeat(lambda1, duration.astype(int))
    inter_lambda_1 = interp1d(t_grid, lambda_1, kind="next")
    lambda_2 = np.repeat(lambda2, duration.astype(int))
    inter_lambda_2 = interp1d(t_grid, lambda_2, kind="next")
    #lambda_3 = np.repeat(lambda_3, Nt)
    
    
    duration_1 = np.array([])
    cost_I =    1.0
    cost_lambda1 = 10.0
    valid_size = 400#2048 # 2*batch_size#4096
    n_maxstep = 1500
    # SAVE DATA TO FILE
    datafile_name = 'data/data_fbsdesolver_params.npz'
    np.savez(datafile_name,
                beta=beta, gamma=gamma, kappa=kappa, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
                cost_I=cost_I, cost_lambda1=cost_lambda1,
                m0=m0,
                batch_size=batch_size, valid_size=valid_size, n_maxstep=n_maxstep)
                # LxUpper=LxUpper, LxLower=LxLower, Nx=Nx, T=T, #Nt=Nt,
                # m0=m0,
                # batch_size=batch_size, valid_size=valid_size, n_maxstep=n_maxstep)
    # SOLVE FBSDE USING NN
    tf.random.set_seed(n_seed) # ---------- TF2
    tf.keras.backend.set_floatx('float64') # To change all layers to have dtype float64 by default, useful when porting from TF 1.X to TF 2.0  # ---------- TF2
    print("========== BEGIN SOLVER MKV FBSDE ==========")
    print("================ PARAMETERS ================")
    print("particle #: ",batch_size, "time: ", T, "infected: ", I0, "beta: " , beta, "gamma: ", gamma, "kappa: ", kappa, "c_I: ", cost_I, "c_lambda: ", cost_lambda1, "max_step: ", n_maxstep)
    print("================ PARAMETERS ================")
    mkv_equation = mkvequation.MckeanVlasovEquation(beta, gamma, kappa, inter_lambda_1, inter_lambda_2, lambda3, cost_I, cost_lambda1)
    mkv_solver = fbsdesolver.SolverMKV1(mkv_equation, T, m0, batch_size, valid_size, n_maxstep) # ---------- TF2
    mkv_solver.train()
    # SAVE TO FILE
    print("SAVING FILES...")
    # y_real =        mkv_solver.Y_real_full
    # x_path =        mkv_solver.X_path
    datafile_name = 'data/data_fbsdesolver_solution_final.npz'
    np.savez(datafile_name,
                prop_S_path = mkv_solver.prop_S_path,
                prop_I_path = mkv_solver.prop_I_path,
                prop_R_path = mkv_solver.prop_R_path,
                t_path = mkv_solver.t_path)
                # y_real=y_real, x_path=x_path)
    print("========== END SOLVER MKV FBSDE ==========")


if __name__ == '__main__':
    np.random.seed(n_seed)
    main()
