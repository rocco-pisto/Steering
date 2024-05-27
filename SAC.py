
from enum import IntEnum
from collections import namedtuple
from math import cos, sin, asin, acos, atan2
import numpy as np
from matplotlib import pyplot as plt

class Axis(IntEnum):
    X = 0
    Y = 1
    Z = 2

    def __add__(self, other):
        if isinstance(other, (int, Axis)):
            value = (self.value + int(other)) % 3
            return Axis(value)
        return NotImplemented

class Manager():
    #  how the system is started up
    def __init__(self, dt_update, k_torque, k_sing, sigma_lim):
        # Pair[0] produce on X, Z
        # Pair[1] produce on Y, Z
        self.cluster = (Pair(Axis.Y, dt_update), \
                        Pair(Axis.X, dt_update) )
        
        self.axis_cmn = Axis.Z
        self.dT = dt_update

        self.k_torque = k_torque
        self.k_sing = k_sing
        self.sigma_lim = sigma_lim

    def require(self, tau_req, V_delta=None):
        if V_delta:
            i = 0
            for pair in self.cluster:
                for cmg in pair.CMG:
                    cmg.theta_meas = V_delta[i]
                    i += 1

        
        V_delta_h_req = tau_req*self.dT  # the residual momentum to satisfy

        V_delta_h_req_pair = (np.zeros(3), np.zeros(3))
        V_delta_h_req_pair[0][Axis.X] = V_delta_h_req[Axis.X]
        V_delta_h_req_pair[1][Axis.Y] = V_delta_h_req[Axis.Y]

        delta_h_cmn = V_delta_h_req[Axis.Z]

        N_p = 21
        V_alpha = np.linspace(-2, 2, N_p)
        V_rew = np.zeros(N_p)
        V_delta_h_prod = np.zeros((3, N_p))

        sigma_fin = np.zeros((2, N_p))
        delta_fin = np.zeros((2, N_p))

        for i, alpha in enumerate(V_alpha):
            V_delta_h_req_pair[0][Axis.Z] = alpha* delta_h_cmn
            V_delta_h_req_pair[1][Axis.Z] = (1-alpha)* delta_h_cmn

            V_delta_h_res = np.zeros(3)
            for j, pair in enumerate(self.cluster):
                delta_H_res, sigma_fin[j, i], delta_fin[j, i]  = pair.require(V_delta_h_req_pair[j])
                V_delta_h_res = V_delta_h_res + delta_H_res

            V_delta_h_prod[:, i] = V_delta_h_req - V_delta_h_res


            V_rew[i] = self.rewardTorque(V_delta_h_req, V_delta_h_prod[:, i]) + \
                        self.rewardSing(sigma_fin[:, i])
        
        # reward based choice
        i = np.argmax(V_rew)

        # angle set and update
        for j, pair in enumerate(self.cluster):
            pair.set(sigma_fin[j, i], delta_fin[j, i])

        V_theta_dot = (self.theta_set-self.theta_meas)/self.dT
        tau_prod = V_delta_h_prod[:, i] / self.dT
        m = V_rew[i]

        return V_theta_dot, tau_prod, m 
    
    def update(self):
        for pair in self.cluster:
            pair.update()

    def rewardTorque(self, V_delta_h_req, V_delta_h_prod):
        # max value = k
        return self.k_torque* (1 - np.linalg.norm(V_delta_h_req - V_delta_h_prod) / np.linalg.norm(V_delta_h_req))
    
    def rewardSing(self, sigma_fin):
        sigma_fin = np.abs(sigma_fin)

        sigma_lim = np.pi / 4
        sigma_in = np.zeros(2)
        for i, pair in enumerate(self.cluster):
            sigma_in[i] = np.abs(pair.sigma_meas)
        
        d_sigma_max = self.cluster[0].CMG[0].alpha_max
        d_sigma_norm = (sigma_fin-sigma_in)/d_sigma_max

        cost_delta = np.power((sigma_fin-np.pi/2)/(self.sigma_lim-np.pi/2), 3)

        # max value = k for sigma
        return self.k_sing* np.sum(cost_delta * d_sigma_norm)

    @property
    def theta_meas(self):
        V_theta_meas = np.zeros(4)
        i = 0
        for pair in self.cluster:
            for cmg in pair.CMG:
                V_theta_meas[i] = cmg.theta_meas
                i += 1

        return V_theta_meas
    
    @property
    def theta_set(self):
        V_theta_set = np.zeros(4)
        i = 0
        for pair in self.cluster:
            for cmg in pair.CMG:
                V_theta_set[i] = cmg.theta_set
                i += 1
                
        return V_theta_set

    @property
    def H_meas(self):
        H_meas = np.zeros(3)
        for pair in self.cluster:
            H_meas = H_meas + pair.H_meas

        return H_meas


class CMG():
    gb_speed_max = 2*np.pi/360*90

    gb_axis = None
    
    fw_inertia = 1
    fw_speed_range = 2*np.pi/60*np.array([500, 10_000])


    H_plane = None

    def __init__(self, axis: Axis, theta_0, dt_update, omega_fw, I_fw, omega_gb_max):
        self.dT = dt_update

        self.gb_axis = axis
        self.H_plane = (axis+1, axis+2)

        self.theta_meas = theta_0 # measured theta
        self.theta_set = theta_0 # setpoint theta

        self.fw_inertia = I_fw
        self.fw_speed = omega_fw
        self.fw_H = I_fw* omega_fw**2

        self.gb_speed_max = omega_gb_max

        self.alpha_max = self.gb_speed_max*self.dT

    # gives the momentum variation with the actual setting
    @property
    def delta_H(self):
        delta_H = np.zeros(3)
        for axis in Axis:
            if axis == self.H_plane[0]:
                delta_H[axis] = self.fw_H* (cos(self.theta_set) - cos(self.theta_meas))
            elif axis == self.H_plane[1]:
                delta_H[axis] = self.fw_H* (sin(self.theta_set) - sin(self.theta_meas))
            else:
                delta_H[axis] = 0

        return delta_H

   

class Pair():
    def __init__(self, axis: Axis, dt_update):
        self.CMG = (CMG(axis, 0, dt_update), CMG(axis, np.pi/2, dt_update))
        self.h = 2*self.CMG[0].fw_H

    def require(self, V_delta_h_req):

        H_req = self.v2h_plane(self.H_meas + V_delta_h_req)

        # never put H_req[i] > self.h
        sigma_req = asin(np.linalg.norm(H_req) / self.h)

        # arctan of Imag [1] and Real [0]
        delta_req = atan2(H_req[1], H_req[0])

        sigma_done, delta_done = self.checkSaturation(sigma_req, delta_req)

        V_delta_h_done = self.deltaH_produced(sigma_done, delta_done)

        V_delta_h_res = V_delta_h_req - V_delta_h_done

        return V_delta_h_res, sigma_done, delta_done
    
    def checkSaturation(self, sigma_req, delta_req):
        theta_req = self.sd2theta(sigma_req, delta_req)

        # coefficient to be inside saturation zone with the same direction
        gamma = self.CMG[0].alpha_max / np.abs(theta_req - self.theta_meas)

        gamma = np.min(gamma)
        if gamma > 1: gamma = 1

        theta_done = gamma* theta_req

        sigma_done, delta_done = self.theta2sd(theta_done)

        return sigma_done, delta_done
    
    def deltaH_produced(self, sigma_set, delta_set):
        H_in = self.H_meas
        H_fin = self.H(sigma_set, delta_set)

        return H_fin-H_in

    def set(self, sigma_set, delta_set):
        theta_set = self.sd2theta(sigma_set, delta_set)
        self.CMG[0].theta_set = theta_set[0]
        self.CMG[1].theta_set = theta_set[1]

    def update(self):
        self.CMG[0].theta_meas = self.CMG[0].theta_set
        self.CMG[1].theta_meas = self.CMG[1].theta_set

    def theta2sd(self, theta):
        sigma = (theta[0] + theta[1])/2
        delta = (theta[0] - theta[1])/2
        return sigma, delta
    
    def sd2theta(self, sigma, delta):
        theta_0 = sigma + delta
        theta_1 = sigma - delta
        return np.array([theta_0, theta_1])
    
    def v2h_plane(self, vect):
        v_hplane = np.array([vect[self.H_plane[0]], vect[self.H_plane[1]]])
        return v_hplane

    def H(self, sigma, delta):
        H = np.zeros(3)

        H_abs = self.h*sin(sigma)

        H[self.H_plane[0]] = H_abs * cos(delta)
        H[self.H_plane[1]] = H_abs * sin(delta)

        return  H
        

    @property
    def sigma_meas(self):
        sigma_meas = self.CMG[0].theta_meas
        sigma_meas += self.CMG[1].theta_meas

        return sigma_meas / 2

    @property
    def delta_meas(self):
        delta_meas = self.CMG[0].theta_meas
        delta_meas -= self.CMG[1].theta_meas

        return delta_meas / 2
    
    @property
    def theta_meas(self):
        theta = np.array([self.CMG[0].theta_meas, self.CMG[1].theta_meas])
        return theta
    
    @property
    def H_meas(self):
        return  self.H(self.sigma_meas, self.delta_meas)
    
    @property
    def theta_set(self):
        theta0_set = self.CMG[0].theta_set
        theta1_set = self.CMG[1].theta_set
        return theta0_set, theta1_set

    @property
    def H_plane(self):
        return self.CMG[0].H_plane
    

if __name__ == '__main__':
    N_sim = 100
    V_H_req = np.zeros((3, N_sim))
    V_H_prod = np.zeros((3, N_sim))
    V_m = np.zeros((N_sim))
    dT = float('1e-3')
    M = Manager(dT)
    tau_sigma = 500
    for i in range(N_sim):
        tau_req = np.random.normal(0, tau_sigma, 3)

        V_theta_dot, tau_prod, V_m[i] = M.require(tau_req)

        H_req = M.H_meas + tau_req*M.dT
        V_H_req[:, i] = H_req
        H_prod = M.H_meas + tau_prod*M.dT
        V_H_prod[:, i] = H_prod

        M.update()

    time = M.dT* np.array(range(N_sim))

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
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

    fig.savefig(f'Torque.png', format='png', dpi=300)

    plt.show()




        
        