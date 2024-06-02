
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
    def __init__(self, fw_H, gb_alpha_max, k_torque, k_sing, sigma_lim, dT):
        # Pair[0] produce on X, Z
        # Pair[1] produce on Y, Z
        theta_set = np.pi/3
        self.cluster = (Pair(Axis.Y, fw_H, gb_alpha_max, theta_set), \
                        Pair(Axis.X, fw_H, gb_alpha_max, 2*theta_set) )
        
        # print(self.H_meas)
        
        self.axis_cmn = Axis.Z

        self.k_torque = k_torque
        self.k_sing = k_sing
        self.sigma_lim = sigma_lim

        self.dT = dT
        self.V_rew_store = None

    def require(self, tau_req, Theta=None):
        if Theta:
            i = 0
            for pair in self.cluster:
                for cmg in pair.CMG:
                    cmg.theta_meas = Theta[i]
                    i += 1

        
        Delta_H_req = tau_req*self.dT  # the residual momentum to satisfy

        Delta_H_req_pair = (np.zeros(3), np.zeros(3))
        Delta_H_req_pair[0][Axis.X] = Delta_H_req[Axis.X]
        Delta_H_req_pair[1][Axis.Y] = Delta_H_req[Axis.Y]

        Delta_H_cmn = Delta_H_req[Axis.Z]

        N_p = 21
        V_alpha = np.linspace(-2, 2, N_p)
        V_rew = np.zeros((N_p, 1))
        Delta_H_prod = np.zeros((3, N_p))

        sigma_fin = np.zeros((2, N_p))
        delta_fin = np.zeros((2, N_p))
        theta_fin = (np.zeros((2, N_p)), np.zeros((2, N_p)))

        for i, alpha in enumerate(V_alpha):
            Delta_H_req_pair[0][Axis.Z] = alpha* Delta_H_cmn
            Delta_H_req_pair[1][Axis.Z] = (1-alpha)* Delta_H_cmn

            Delta_H_res = np.zeros((3,2))
            for j, pair in enumerate(self.cluster):
                Delta_H_res[:, j], sigma_fin[j, i], delta_fin[j, i], theta_fin[j][:, i]  = pair.require(Delta_H_req_pair[j])

            Delta_H_prod[:, i] = Delta_H_req - np.sum(Delta_H_res, axis=1)

            
            V_rew[i] = self.rewardTorque(Delta_H_req, Delta_H_prod[:, i]) + \
                        self.rewardSing(delta_fin[:, i])
        if type(self.V_rew_store) == type(None):
            self.V_rew_store = V_rew
        else:
            self.V_rew_store = np.column_stack((self.V_rew_store, V_rew))

        # reward based choice
        i = np.argmax(V_rew)

        # angle set and update
        for j, pair in enumerate(self.cluster):
            pair.set(theta_fin[j][:, i])

        V_theta_dot = (self.theta_set-self.theta_meas)/self.dT
        tau_prod = Delta_H_prod[:, i] / self.dT
        m = V_rew[i]

        return V_theta_dot, tau_prod, m 
    
    def update(self):
        for pair in self.cluster:
            pair.update()

    def rewardTorque(self, Delta_H_req, Delta_H_prod):
        # max value = k
        return self.k_torque* (1 - np.linalg.norm(Delta_H_req - Delta_H_prod) / np.linalg.norm(Delta_H_req))
    
    def rewardSing(self, delta_fin):
        # delta is in range [0, pi/2] -> [external, internal]

        delta_in = np.zeros(2)
        for i, pair in enumerate(self.cluster):
            delta_in[i] = np.abs(pair.delta_meas)
        
        d_delta_max = self.cluster[0].CMG[0].gb_alpha_max

        delta_fin = Pair.angleReference(delta_in, delta_fin)
        d_delta_norm = (delta_fin-delta_in)/d_delta_max

        # reward given by y = ((x-m)/(p-m))^3
        ## m is the central point: pi/4
        ## x is the variable delta_fin
        ## p is the point where the cost is 1 (parameter)

        rew_delta = np.power((delta_fin-np.pi/4)/-(self.sigma_lim-np.pi/4), 3)

        # max value = k * rew_delta
        # the actual reward depends on the sign (direction) of delta
        return self.k_sing* np.sum( - rew_delta * d_delta_norm)

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




class Pair():
    def __init__(self, axis: Axis, fw_H, gb_alpha_max, theta_0):
        self.CMG = (CMG(axis, fw_H, gb_alpha_max, theta_0), CMG(axis, fw_H, gb_alpha_max, -theta_0))
        # this is maximum dTorque * dT / max dSigma or dTheta (CMG.alpha_max) 
        self.h = 2*fw_H                                                                                                                                         

    def require(self, Delta_H_req):

        H_req = self.cart2h_axis(self.H_meas + Delta_H_req)

        # never put H_req[i] > self.h
        H_req_norm = np.linalg.norm(H_req)
        # external singularity
        if H_req_norm > self.h:
            H_req_norm = self.h
            delta_req = 0
        else:
            delta_req = acos( H_req_norm / self.h)

        # arctan of Imag [1] (X | Y) and Real [0] (Z)
        sigma_req = atan2(H_req[1], H_req[0])

        # gimbal max speed saturation
        sigma_done, delta_done, theta_done = self.checkSaturation(sigma_req, delta_req)

        Delta_H_done = self.deltaH_produced(sigma_done, delta_done)

        Delta_H_res = Delta_H_req - Delta_H_done

        return Delta_H_res, sigma_done, delta_done, theta_done
    
    def checkSaturation(self, sigma_req, delta_req):
        theta_req = self.sd2theta(sigma_req, delta_req)
        # express theta_req as number to have minimum distance from theta meas
        # correct also eventualy the theta_0,1 swap
        theta_req = self.thetaRefMeas(theta_req)

        # coefficient to be inside saturation zone with the same direction
        gamma = np.abs(theta_req - self.theta_meas) / self.CMG[0].gb_alpha_max 

        # important to get the maximum to get the axis with more saturation
        gamma = np.max(gamma)

        # rescaling of the vector based on relative saturation
        if gamma > 1:
            theta_done = theta_req / gamma
        else:
            theta_done = theta_req

        sigma_done, delta_done = self.theta2sd(theta_done)

        return sigma_done, delta_done, theta_done
    
    def deltaH_produced(self, sigma_set, delta_set):
        H_in = self.H_meas
        H_fin = self.H(sigma_set, delta_set)

        return H_fin-H_in

    def set(self, theta_set):
        self.CMG[0].theta_set = theta_set[0]
        self.CMG[1].theta_set = theta_set[1]

    def update(self):
        for cmg in self.CMG:
            cmg.update()

    @staticmethod
    def theta2sd(theta):
        sigma = Pair.satAngle((theta[0] + theta[1])/2)
        delta = Pair.satAngle((theta[0] - theta[1])/2)
        # we keep delta in [0, pi/2] (cos(delta) in [0, 1]) and sigma in [-pi, pi]
        # explanation: one degree of fredoom lost beacuse of the irrilevant 
        # position order

        # check if cos(delta) < 0 (II and III quadrant) -> fold back in (I an IV quadrant)
        if delta <= -np.pi/2:
            delta = -np.pi - delta
            sigma = sigma + np.pi
        elif delta >  np.pi/2:
            delta = +np.pi - delta
            sigma = sigma + np.pi
        
        # fold II quadrant in I quadrant
        if delta < 0:
            delta = -delta
        return sigma, delta
    
    @staticmethod
    def sd2theta(sigma, delta):
        # this give the two angles in the usual range (-pi, pi]
        theta_0 = Pair.satAngle(sigma + delta)
        theta_1 = Pair.satAngle(sigma - delta)
        return np.array((theta_0, theta_1))
    
    @staticmethod
    def satAngle(theta):
        while theta > np.pi:
            theta -= 2*np.pi
        while theta <= -np.pi:
            theta += 2*np.pi

        return theta
        
    @staticmethod
    def angleReference(v_theta_ref, v_theta_new):
        for i in range(len(v_theta_ref)):
            theta_ref, theta_new = v_theta_ref[i], v_theta_new[i]
            # going anti-clockwise change sign on -pi
            if theta_ref > 0 and theta_new < 0 and (theta_ref-theta_new) > np.pi:
                theta_new += 2*np.pi
            # going clockwise change sign on -pi
            elif theta_ref < 0 and theta_new > 0 and (theta_new-theta_ref) > np.pi:
                theta_new -= 2*np.pi
            v_theta_ref[i], v_theta_new[i] = theta_ref, theta_new

        return v_theta_new


    def thetaRefMeas(self, theta_set):
        theta_meas = self.theta_meas

        theta_set_forw = Pair.angleReference(theta_meas, theta_set)
        theta_set_back = Pair.angleReference(theta_meas, np.flip(theta_set))

        if np.sum(np.abs(theta_set_forw-theta_meas)) > \
            np.sum(np.abs(theta_set_back-theta_meas)):
            theta_set = theta_set_back
        else:
            theta_set = theta_set_forw

        return theta_set

    
    def cart2h_axis(self, vect):
        v_h_axis = np.array([vect[self.H_axis.Real], vect[self.H_axis.Imag]])
        return v_h_axis

    def H(self, sigma, delta):
        H = np.zeros(3)

        H_abs = self.h*cos(delta)

        H[self.H_axis.Real] = H_abs * cos(sigma)
        H[self.H_axis.Imag] = H_abs * sin(sigma)

        return  H
    
    # TO DO: implement check on theta_min 

    @property
    def sigma_meas(self):
        sigma_meas, delta_meas = self.theta2sd(self.theta_meas)
        
        return sigma_meas

    @property
    def delta_meas(self):
        sigma_meas, delta_meas = self.theta2sd(self.theta_meas)
        
        return delta_meas
    

    
    @property
    def theta_meas(self):
        theta = np.array([self.CMG[0].theta_meas, self.CMG[1].theta_meas])
        return theta
    
    @property
    def theta_set(self):
        return np.array([self.CMG[0].theta_set, self.CMG[1].theta_set])

    @property
    def H_meas(self):
        return  self.H(self.sigma_meas, self.delta_meas)

    @property
    def H_axis(self):
        return self.CMG[0].H_axis



class CMG():
    AxName = namedtuple('AxName', ['Real', 'Imag'])        
    def __init__(self, axis: Axis, fw_H, gb_alpha_max, theta_0):

        self.gb_axis = axis
        # highest index axis (Z) is the main
        H_axis = [axis+1, axis+2]
        # z will always be the Real axis [0]
        H_axis.sort(reverse=True)

        self.H_axis = self.AxName(Real=H_axis[0], Imag=H_axis[1])

        self.n_round = 0 # numer of round of gimbal (clkspring)

        self.theta_meas = 0 # measured theta
        self.theta_set = theta_0 # setpoint theta
        self.update()

        self.fw_H = fw_H

        self.gb_alpha_max = gb_alpha_max

    def update(self):
        # on -pi we go back on (-pi, pi] range
        if self.theta_set > np.pi:
            self.theta_set -= 2*np.pi
        elif self.theta_set <= -np.pi:
            self.theta_set += 2*np.pi

        # keep track on number of revolution
        elif self.theta_set > 0 and self.theta_meas < 0:
            self.n_round += 1
        elif self.theta_set < 0 and self.theta_meas > 0:
            self.n_round -= 1

        self.theta_meas = self.theta_set
        
        

        
    # gives the momentum variation with the actual setting
    @property
    def Delta_H(self):
        Delta_H = np.zeros(3)
        for axis in Axis:
            if axis == self.H_axis.Real:
                Delta_H[axis] = self.fw_H* (sin(self.theta_set) - sin(self.theta_meas))
            elif axis == self.H_axis.Imag:
                Delta_H[axis] = self.fw_H* (cos(self.theta_set) - cos(self.theta_meas))
            else:
                Delta_H[axis] = 0

        return Delta_H

       

if __name__ == '__main__':
    dT = float('1e-3')
    fw_H = 1

    gb_omega_max = 2*np.pi/360 * 90
    gb_alpha_max = gb_omega_max*dT

    N_sim = 100

    V_H_req = np.zeros((3, N_sim))
    V_H_prod = np.zeros((3, N_sim))
    V_m = np.zeros((N_sim))

    M = Manager(fw_H, gb_alpha_max, k_torque=1, k_sing=1, sigma_lim=np.pi/8, dT=dT)
    tau_sigma = 2*fw_H*gb_alpha_max / dT
    for i in range(N_sim):
        tau_req = np.random.normal(0, tau_sigma/100, 3)

        V_theta_dot, tau_prod, V_m[i] = M.require(tau_req)

        H_req = M.H_meas + tau_req*M.dT
        V_H_req[:, i] = H_req
        H_prod = M.H_meas + tau_prod*M.dT
        V_H_prod[:, i] = H_prod

        M.update()

    time = M.dT* np.array(range(N_sim))

    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
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




        
        