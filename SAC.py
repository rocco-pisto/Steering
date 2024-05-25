
from enum import IntEnum
from collections import namedtuple
from math import cos, sin, asin, acos, atan2
import numpy as np

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
    def __init__(self, dt_update):
        # Pair[0] produce on X, Z
        # Pair[1] produce on Y, Z
        self.cluster = (Pair(Axis.Y, dt_update), \
                        Pair(Axis.X, dt_update) )
        
        self.axis_cmn = Axis.Z
        self.dT = dt_update

    def require(self, tau):
        
        V_delta_h_req = tau*self.dT  # the residual momentum to satisfy

        V_delta_h_req_pair = (np.zeros(3), np.zeros(3))
        V_delta_h_req_pair[0][Axis.X] = V_delta_h_req[Axis.X]
        V_delta_h_req_pair[1][Axis.Y] = V_delta_h_req[Axis.Y]

        delta_h_cmn = V_delta_h_req[Axis.Z]

        N_p = 21
        V_alpha = np.linspace(-2, 2, N_p)
        V_rew = np.zeros(N_p)
        V_delta_h_prod = np.zeros(3, N_p)

        for i, alpha in enumerate(V_alpha):
            V_delta_h_req_pair[0][Axis.Z] = alpha* delta_h_cmn
            V_delta_h_req_pair[1][Axis.Z] = (1-alpha)* delta_h_cmn

            V_delta_H_res = np.zeros(3)
            sigma_fin = np.array(2, N_p)
            delta_fin = np.array(2, N_p)
            for j, pair in enumerate(self.cluster):
                delta_H_res, sigma_fin[j][i], delta_fin[j][i]  = pair.require(V_delta_h_req_pair[j])
                V_delta_h_res = V_delta_h_res + delta_H_res

            V_delta_h_prod[:, i] = V_delta_h_req - V_delta_H_res


            V_rew[i] = self.rewardTorque(V_delta_h_req, V_delta_h_prod) + \
                        self.rewardSing(sigma_fin)
        
        # reward based choice
        i = np.argmax(V_rew)

        # angle set and update
        for j, pair in enumerate(self.cluster):
            pair.set(sigma_fin[j, i], delta_fin[j, i])

            pair.update()

        return V_delta_h_prod[:, i]
    
    def rewardTorque(self, V_delta_h_req, V_delta_h_prod):
        # max value = k
        k = 1
        return k* (1 - np.abs(V_delta_h_req - V_delta_h_prod) / np.abs(V_delta_h_req))
    
    def rewardSing(self, sigma_done):
        # max value = k for sigma
        k = 1
        return k* np.sum(np.sin(sigma_done))



class CMG():
    gb_speed_max = 90

    gb_axis = None
    
    fw_inertia = 1
    fw_speed_range = np.array([500, 10_000])
    fw_H_range = fw_inertia * fw_speed_range

    H_plane = None

    Cost = namedtuple('Cost', ['d_theta', 'd_h', 'd_S'])

    def __init__(self, axis: Axis, theta_0, dt_update):
        self.dT = dt_update

        self.gb_axis = axis
        self.H_plane = (axis+1, axis+2)

        self.theta_meas = theta_0 # measured theta
        self.theta_set = theta_0 # setpoint theta
        self.fw_H = self.fw_H_range[1]

        self.alpha_max = self.gb_speed_max*self.dT

    # we try to set an angle with the goal to make a momentum variation
    # the actual possible momentum and the variation on the other axis are returned together
    def set(self, theta_set):

        range_set = (theta_set-self.theta_meas) / self.alpha_max
        if range_set > 1:
            self.theta_set = self.theta_meas + self.alpha_max
        elif range_set < -1:
            self.theta_set = self.theta_meas - self.alpha_max
        else:
            self.theta_set = theta_set
        
        return self.delta_H

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
        self.h = 2*CMG[0].fw_inertia

    def require(self, V_delta_h_req):

        delta_h_req = np.array([ V_delta_h_req[self.H_plane[0]], \
                                    V_delta_h_req[self.H_plane[1]] ])

        H_req = self.H_meas + delta_h_req

        # never put H_req[i] > self.h
        sigma_req = asin(H_req / self.h)

        # arctan of Imag [1] and Real [0]
        delta_req = atan2(H_req[self.H_plane[1]], H_req[self.H_plane[0]])

        sigma_done, delta_done = self.checkSaturation(sigma_req, delta_req)

        delta_h_done = self.deltaH_produced(sigma_done, delta_done)

        V_delta_h_done = np.array(3)
        V_delta_h_done[self.H_plane[0]] = delta_h_done[0]
        V_delta_h_done[self.H_plane[1]] = delta_h_done[1]

        V_delta_h_res = V_delta_h_req - V_delta_h_done

        return V_delta_h_res, sigma_done, delta_done
    
    def checkSaturation(self, sigma_req, delta_req):
        theta_req = self.sd2theta(sigma_req, delta_req)

        # coefficient to be inside saturation zone with the same direction
        gamma = self.CMG[0].alpha_max / np.abs(theta_req - self.theta_meas)

        gamma = np.min(gamma)

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
        return np.array(theta_0, theta_1)
    
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
        return sigma_meas

    @property
    def delta_meas(self):
        delta_meas = self.CMG[0].theta_meas
        delta_meas -= self.CMG[1].theta_meas
        return delta_meas
    
    @property
    def theta_meas(self):
        theta = np.array(self.CMG[0].theta_meas, self.CMG[1].theta_meas)
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
        return CMG.H_plane
    

if __name__ == '__main__':
    M = Manager(float('1e-3'))
    tau_sigma = 10/3
    while True:
        tau_req = np.random.normal(0, tau_sigma, 3)
        tau_prod = M.require(tau_req)
        
        