
from enum import IntEnum
from collections import namedtuple
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
        self.cluster = (CMG(Axis.X, 0, dt_update), \
                        CMG(Axis.X, np.pi/2, dt_update), \
                        CMG(Axis.Y, 0, dt_update), \
                        CMG(Axis.Y, np.pi/2, dt_update))
        self.dT = dt_update

    def require(self, tau):
        tol = float('1e-9')
        delta_Vh_res = tau*self.dT  # the residual momentum to satisfy

        M_cost = []
        for i, cmg in enumerate(self.cluster):
            M_cost[i] = cmg.costTot(delta_Vh=delta_Vh_res)

        for i in range(len(self.cluster)):

        

        delta_Vh_not_served = np.zeros(3) # the final momentum not served
        while not np.all(abs(delta_Vh_res) < tol):
            axis_dh_max = Axis( np.argmax(abs(delta_Vh_res)))

            cmgs_cost_indx = np.argsort(M_cost[:, axis_dh_max])
            for i_cmg in cmgs_cost_indx:
                delta_h_req = delta_Vh_res[axis_dh_max]
                theta_fin = self.cluster[i_cmg].require(delta_h_req, axis_dh_max)
                delta_Vh_prod = self.cluster[i_cmg].set(theta_fin)

                delta_Vh_res -= delta_Vh_prod
                if abs(delta_Vh_res[axis_dh_max]) < tol:
                    break

            delta_Vh_not_served[axis_dh_max] = delta_Vh_res[axis_dh_max]
            delta_Vh_res[axis_dh_max] = 0

        return delta_Vh_not_served/self.dT
    
    def update(self):
        for cmg in self.cluster:
            cmg.theta_meas = cmg.theta_set


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

        self.d_theta_max = self.gb_speed_max*self.dT

    # we try to set an angle with the goal to make a momentum variation
    # the actual possible momentum and the variation on the other axis are returned together
    def set(self, theta_set):

        range_set = (theta_set-self.theta_meas) / self.d_theta_max
        if range_set > 1:
            self.theta_set = self.theta_meas + self.d_theta_max
        elif range_set < -1:
            self.theta_set = self.theta_meas - self.d_theta_max
        else:
            self.theta_set = theta_set
        
        return self.delta_H

    # gives the momentum variation with the actual setting
    @property
    def delta_H(self):
        delta_H = np.zeros(3)
        for axis in Axis:
            if axis == self.H_plane[0]:
                delta_H[axis] = self.fw_H* (np.cos(self.theta_set) - np.cos(self.theta_meas))
            elif axis == self.H_plane[1]:
                delta_H[axis] = self.fw_H* (np.sin(self.theta_set) - np.sin(self.theta_meas))
            else:
                delta_H[axis] = 0

        return delta_H
            

    # return the require delta_theta for a certain momentum
    def require(self, delta_h, axis: Axis):
        if axis == self.H_plane[0]:
            cos_theta_s = delta_h / self.fw_H + np.cos(self.theta_meas)
            if cos_theta_s > 1:
                cos_theta_s = 1
            elif cos_theta_s < -1:
                cos_theta_s = -1

            theta_s = np.arccos(cos_theta_s)
            if abs((theta_s) - self.theta_meas) \
                > abs((-theta_s) - self.theta_meas):
                theta_s = -theta_s
            
        elif axis == self.H_plane[1]:
            sin_theta_s = delta_h / self.fw_H + np.sin(self.theta_meas)
            if sin_theta_s > 1:
                sin_theta_s = 1
            elif sin_theta_s < -1:
                sin_theta_s = -1

            theta_s = np.arcsin(sin_theta_s)
            if abs((theta_s) - self.theta_meas) \
                > abs((np.pi-theta_s) - self.theta_meas):
                theta_s = np.pi-theta_s
            elif abs((theta_s) - self.theta_meas) \
                > abs((-np.pi-theta_s) - self.theta_meas):
                theta_s = -np.pi-theta_s

        else:
            theta_s = np.Inf
        
        d_theta = theta_s - self.theta_meas
        return d_theta, theta_s

        
    def deltaV_prod(self, theta_s):

        delta_Vh_prod = np.zeros(3)
        if theta_s != np.Inf:
            axis_a = self.H_plane[0]
            axis_b = self.H_plane[1]
            axis_n = self.gb_axis

            delta_Vh_prod[axis_a] = (np.cos(theta_s) - np.cos(self.theta_meas)) * self.fw_H
            delta_Vh_prod[axis_b] = (np.sin(theta_s) - np.sin(self.theta_meas)) * self.fw_H
            delta_Vh_prod[axis_n] = 0

        return delta_Vh_prod
        
    def costDiff(self, theta):
        axis = [self.H_plane[0], self.H_plane[1], self.gb_axis]

        cost_Vdiff = np.zeros(3)
        for i, ax in enumerate(axis):
            if i == 0:
                diff_H = self.fw_H* np.sin(theta)
            elif i == 1:
                diff_H = self.fw_H* np.cos(theta)
            else: 
                diff_H = 0
            
            try:
                cost_Vdiff[ax] = 1 / abs(diff_H)
            except ZeroDivisionError:
                cost_Vdiff[ax] = np.Inf

    def varCostDiff(self, theta_fin, theta_in):
        if theta_fin == np.Inf:
            return np.Inf * np.ones(3)
        else:
            return self.costDiff(theta_fin) - self.costDiff(theta_in)

    def costTot(self, delta_Vh):
        delta_Vtheta = np.zeros(3)
        delta_Vh_prod = np.zeros((3,3))
        delta_Vcost_diff = np.zeros((3,3))
        for axis in Axis:
            delta_Vtheta[axis], theta_s = self.require(delta_Vh[axis], axis)
            delta_Vh_prod[axis, :] = self.deltaV_prod(theta_s)
            delta_Vcost_diff[axis, :] = self.varCostDiff(theta_s, self.theta_meas)

        delta_Vh_prod = np.diag(delta_Vh_prod)

        return self.Cost(delta_Vtheta, delta_Vh_prod, delta_Vcost_diff)



if __name__ == '__main__':
    M = Manager(float('1e-3'))
    tau_sigma = 10/3
    while True:
        tau_req = np.random.normal(0, tau_sigma, 3)
        tau_not_served = M.require(tau_req)
        M.update()
