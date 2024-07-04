from SAC import *   

if __name__ == '__main__':
    dT = float('1e-1')
    fw_H = 1

    gb_omega_max = 2*np.pi/360 * 90
    gb_alpha_max = gb_omega_max*dT

    N_sim = 35
    time = dT* np.array(range(N_sim))

    V_H_req = np.zeros((3, N_sim))
    V_H_prod = np.zeros((3, N_sim))
    V_m = np.zeros((N_sim))

    M = Manager(fw_H, gb_alpha_max, k_torque=50, k_sing=50, delta_lim=1/64, dT=dT)

    test_n = 2

    if test_n == 1:
    ########## random test ##########
        tau_sigma = 2*fw_H*gb_alpha_max / dT
        for i in range(N_sim):
            tau_req = np.random.normal(0, tau_sigma/100, 3)

            V_theta_dot, tau_prod, V_m[i] = M.require(tau_req)

            H_req = M.H_meas + tau_req*M.dT
            V_H_req[:, i] = H_req
            H_prod = M.H_meas + tau_prod*M.dT
            V_H_prod[:, i] = H_prod

            M.update()
            

        fig0, axs = plt.subplots(1, 3, figsize=(18, 4))
        for i in range(3):

            # error plot
            axs[i].plot(time, V_H_req[i, :], label='H commanded', color='red')
            axs[i].plot(time, V_H_prod[i, :], label='H produced', color='blue', marker='o', linestyle='--', linewidth=0.2)


            plt.legend()

            # axis labels
            axs[i].set_xlabel('time [s]')
            axs[i].set_ylabel(f'Torque {Axis(i)}')

            # grid
            axs[i].grid(True, which='both', linestyle='-', linewidth=0.5)

        fig0.savefig(f'Torque_Rand.png', format='png', dpi=300)

        plt.show()

    elif test_n == 2:
    ########## limit test ##########
        # Create a figure with two subplots arranged horizontally
        fig0, (ax0, ax1) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 6))


        # enable update figure
        # plt.ion()

        # Plot Pair 0
        # line0, = ax0.plot(0, 0)
        # ax0.annotate('z', xy=(0, 0), xytext=(0, 1.1), textcoords='data', ha='center', va='center', fontsize=12)
        # ax0.annotate('', xy=(0, 1), xytext=(0, 0), arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10))
        # ax0.annotate('x', xy=(0, 0), xytext=(np.pi/2, 1.1), textcoords='data', ha='center', va='center', fontsize=12)
        # ax0.annotate('', xy=(np.pi/2, 1), xytext=(0, 0), arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10))
        ax0.set_title("PAIR 0 : (x, z)")
        ax0.set_ylim([0, 1])

        # Plot Pair 1
        # line1, = ax1.plot(0, 0)
        # ax1.annotate('z', xy=(0, 0), xytext=(0, 1.1), textcoords='data', ha='center', va='center', fontsize=12)
        # ax1.annotate('', xy=(0, 1), xytext=(0, 0), arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10))
        # ax1.annotate('y', xy=(0, 0), xytext=(np.pi/2, 1.1), textcoords='data', ha='center', va='center', fontsize=12)
        # ax1.annotate('', xy=(np.pi/2, 1), xytext=(0, 0), arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10))
        ax1.set_title("PAIR 1 : (y, z)")
        ax1.set_ylim([0, 1])

        # plt.tight_layout()

        H_meas_pair_abs = np.zeros((2, N_sim))
        H_meas_pair_ph = np.zeros((2, N_sim))
        tau_res_norm = np.zeros((3, N_sim))
        tau_sigma = 2*fw_H*gb_alpha_max / dT /3
        for i in range(N_sim):
            tau_req = np.array((0, 0, tau_sigma))

            V_theta_dot, tau_prod, V_m[i] = M.require(tau_req)

            tau_res_norm[:, i] = 100* abs(tau_prod-tau_req) / abs(tau_req)

            M.update()

            for j, pair in enumerate(M.Pair):
                H_meas_pair_3v = pair.H_meas
                H_meas_pair= np.array((H_meas_pair_3v[pair.H_axis.Real], H_meas_pair_3v[pair.H_axis.Imag]))
                print(f'PAIR {j}: {H_meas_pair[0]}')
                H_meas_pair_abs[j, i] = np.linalg.norm(H_meas_pair) / pair.h
                H_meas_pair_ph[j, i] = np.arctan2(H_meas_pair[1], H_meas_pair[0])

            print('\n')

            # line0.set_xdata(H_meas_pair_ph[0, 0:i])
            # line0.set_ydata(H_meas_pair_abs[0, 0:i])

            # line1.set_xdata(H_meas_pair_ph[1, 0:i])
            # line1.set_ydata(H_meas_pair_abs[1, 0:i])

            # fig.canvas.draw()
            # fig.canvas.flush_events()

            # plt.pause(0.5)

        line0, = ax0.plot(H_meas_pair_ph[0,:], H_meas_pair_abs[0,:])
        line1, = ax1.plot(H_meas_pair_ph[1,:], H_meas_pair_abs[1,:])

        ax0.scatter(H_meas_pair_ph[0,0], H_meas_pair_abs[0,0], marker='s')
        ax0.scatter(H_meas_pair_ph[0,-1], H_meas_pair_abs[0,-1], marker='o')
        ax0.scatter(H_meas_pair_ph[0,1:-1], H_meas_pair_abs[0,1:-1], marker='.')

        ax1.scatter(H_meas_pair_ph[1,0], H_meas_pair_abs[1,0], marker='s')
        ax1.scatter(H_meas_pair_ph[1,-1], H_meas_pair_abs[1,-1], marker='o')
        ax1.scatter(H_meas_pair_ph[1,1:-1], H_meas_pair_abs[1,1:-1], marker='.')

        plt.figure(2)
        plt.plot(time, V_m)
        plt.figure(3)
        plt.plot(time, tau_res_norm[2,:])
        
        plt.show()
        