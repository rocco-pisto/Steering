from RDCP import *   

if __name__ == '__main__':
    dT = float('1e-1')

    N_sim = 35
    time = dT* np.array(range(N_sim))

    V_tau_req = np.zeros((3, N_sim))
    V_theta_dot = np.zeros((2, 2, N_sim))
    V_tau_prod = np.zeros((3, N_sim))
    V_H_pair = np.zeros((N_sim, 2, 2))
    V_H_pair_abs = np.zeros((N_sim, 2))
    V_H_pair_ph = np.zeros((N_sim, 2))
    V_m = np.zeros((N_sim))


    M = RDCP(skew=45, k_torque=100, delta_unit_cost=5, dt=dT)
    M.v_theta_set = np.array(((90, -90), (90, -90)))

    test_n = 2

    tau_sigma = Pair.d_H_max / dT
    tau_req = np.zeros(3)
    for i in range(N_sim):
        ########## random test ##########
        if test_n == 1:
            tau_req = np.random.normal(0, tau_sigma/10, 3)
        elif test_n == 2:
            tau_req[Rooftop.X] = 0
            tau_req[Rooftop.Y] = 0
            tau_req[Rooftop.Z] = tau_sigma / 2

        V_tau_req[:, i] = tau_req
        V_theta_dot[:, :, i], V_tau_prod[:, i], V_m[i] = M.require(tau_req, None)

        for p in range(2):
            V_H_pair[i, :, p] = M.Pair[p].H_meas
    
    for p in range(2):
        V_H_pair_abs[:, p]  = np.linalg.norm(V_H_pair[:, :, p], ord=2, axis=1) / Pair.h
        V_H_pair_ph[:, p]   = np.arctan2(
                                V_H_pair[:, Rooftop.IM, p], \
                                V_H_pair[:, Rooftop.RE, p])
        
    ######## PLOT TORQUE
    fig0, axs = plt.subplots(1, 3, figsize=(18, 4))
    for i in range(3):

        # error plot
        axs[i].plot(time, V_tau_req[i, :], label='H commanded', color='red')
        axs[i].plot(time, V_tau_prod[i, :], label='H produced', color='blue', marker='o', linestyle='--', linewidth=0.2)

        # axis labels
        axs[i].set_xlabel('time [s]')
        axs[i].set_ylabel(f'Torque')

        # grid
        axs[i].grid(True, which='both', linestyle='-', linewidth=0.5)

        axs[i].legend()

    fig0.savefig(f'Torque.png', format='png', dpi=300)

    ######## PLOT GB_RATES
    fig1, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(2):
        for j in range(2):

            # error plot
            axs[i, j].plot(time, V_theta_dot[i, j, :], label=f'PAIR {i} - CMG {j}', color='black')



            # axis labels
            axs[i, j].set_xlabel('time [s]')
            axs[i, j].set_ylabel(f'GB RATE [deg/s]')

            # grid
            axs[i, j].grid(True, which='both', linestyle='-', linewidth=0.5)

            axs[i, j].legend()


    fig1.savefig(f'GB_Rate.png', format='png', dpi=300)

    ######## PLOT PAIRS COMPLEX PLANE
    fig2, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 6))

    axs[0].set_title("PAIR 0 : (X, U)")
    axs[0].set_ylim([0, 1])

    axs[1].set_title("PAIR 1 : (X, V)")
    axs[1].set_ylim([0, 1])

    for p in range(2):
        line, = axs[p].plot(V_H_pair_ph[:, p], V_H_pair_abs[:, p])

        axs[p].scatter(V_H_pair_ph[0,p],    V_H_pair_abs[0,p],      marker='s')
        axs[p].scatter(V_H_pair_ph[-1,p],   V_H_pair_abs[-1,p],     marker='o')
        axs[p].scatter(V_H_pair_ph[1:-1,p], V_H_pair_abs[1:-1,p],   marker='.')

    fig2.savefig(f'Pairs_Complex.png', format='png', dpi=300)

    plt.show()
