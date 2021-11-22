import mkvequation_reg as mkvequation
import fbsdesolver_reg_terminal as fbsdesolver
import tensorflow as tf
import numpy as np

print('\n\ntf.VERSION = ', tf.__version__, '\n\n')
print('\n\ntf.keras.__version__ = ', tf.keras.__version__, '\n\n')

n_seed = 1 # random seed

def main():
    # parameters for neural network and gradient descent
    batch_size =        200
    valid_size =        500
    n_maxstep =         1500

    # problem parameters - see "Optimal incentives to mitigate epidemics: a Stackelberg mean field game approach" for interpretation
    T =                 30.0
    E0, I0, R0, D0 =    0.0, 0.1, 0.0, 0.0 
    S0 =                1.0 - E0 - I0 - R0 - D0
    m0 =                [S0,E0, I0, D0, R0]
    beta =              0.25
    gamma =             1./10
    LAM =               2./1
    delta =             1./100
    eta =               1./100
    cost_I =            1.0
    cost_lambda1 =      10.0
    c_sq =              1.0
    beta_reg =          [0.2, 0.2, 1.0]
    lambda_goal =       [1.0, 1.0, 0.7]
    minorcost_D =       20.
    regcost_D =         20.
    
    # save parameter data to file
    datafile_name = 'data/data_fbsdesolver_params.npz'
    np.savez(datafile_name,
                beta=beta, gamma=gamma, LAM=LAM, delta=delta, eta=eta,
                cost_I=cost_I, cost_lambda1=cost_lambda1, c_sq=c_sq, 
                minorcost_D=minorcost_D, regcost_D=regcost_D, beta_reg=beta_reg,
                lambda_goal=lambda_goal, m0=m0,
                batch_size=batch_size, valid_size=valid_size, n_maxstep=n_maxstep)

    # solve FBSDE with SolverMKV1 (fbsdesolver_reg_terminal.py)
    tf.random.set_seed(n_seed) # ---------- TF2
    tf.keras.backend.set_floatx('float64') 
    print("========== BEGIN SOLVER MKV FBSDE ==========")
    mkv_equation = mkvequation.MckeanVlasovEquation(beta, gamma, LAM, delta, eta, cost_I, cost_lambda1, c_sq, minorcost_D, regcost_D, beta_reg, lambda_goal)
    mkv_solver = fbsdesolver.SolverMKV1(mkv_equation, T, m0, batch_size, valid_size, n_maxstep) # ---------- TF2
    mkv_solver.train()

    # save result to file
    print("SAVING FILES...")
    datafile_name = 'data/data_fbsdesolver_solution_final.npz'
    np.savez(datafile_name,
                prop_S_path  = mkv_solver.prop_S_path,
                prop_E_path  = mkv_solver.prop_E_path,
                prop_I_path  = mkv_solver.prop_I_path,
                prop_D_path  = mkv_solver.prop_D_path,
                prop_R_path  = mkv_solver.prop_R_path,
                t_path       = mkv_solver.t_path,
                loss_history = mkv_solver.loss_history,
                LAMBDA_path  = mkv_solver.LAMBDA_path,
                Y_path       = mkv_solver.Y_path)
    print("========== END SOLVER MKV FBSDE ==========")

# run program
if __name__ == '__main__':
    np.random.seed(n_seed)
    main()
