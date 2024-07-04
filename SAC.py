
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
    def __init__(self, fw_H, gb_alpha_max, k_torque, k_sing, delta_lim, dT):
        # Pair[0] produce on X, Z
        # Pair[1] produce on Y, Z
        theta_set = 1/3
        self.Pair = (Pair(Axis.Y, fw_H, gb_alpha_max, theta_set), \
                        Pair(Axis.X, fw_H, gb_alpha_max, 2*theta_set) )
        
        # print(self.H_meas)
        
        self.axis_cmn = Axis.Z

        self.k_torque = k_torque
        self.k_sing = k_sing
        self.delta_lim = delta_lim

        self.dT = dT
        self.V_rew_store = None

    def require(self, tau_req, Theta=None):
        if Theta:
            i = 0
            for pair in self.Pair:
                for cmg in pair.CMG:
                    cmg.theta_meas = Theta[i]
                    i += 1

        
        Delta_H_req = tau_req*self.dT  # the residual momentum to satisfy

        Delta_H_req_pair = (np.zeros(3), np.zeros(3))
        Delta_H_req_pair[0][Axis.X] = Delta_H_req[Axis.X]
        Delta_H_req_pair[1][Axis.Y] = Delta_H_req[Axis.Y]

        Delta_H_cmn = Delta_H_req[Axis.Z]
        delta_H_max = self.h*self.D_alpha_max
        alpha_max = 5*delta_H_max / Delta_H_cmn 

        N_p = 41
        V_alpha = np.linspace(-alpha_max, alpha_max, N_p)
        V_rew = np.zeros((N_p, 1))
        Delta_H_prod = np.zeros((N_p, 3))

        sigma_fin = np.zeros((N_p, 2))
        delta_fin = np.zeros((N_p, 2))
        theta_fin = np.zeros((N_p, 2, 2))

        for i, alpha in enumerate(V_alpha):
            if i == 40:
                pass
            Delta_H_req_pair[0][Axis.Z] = (0.5+alpha)* Delta_H_cmn
            Delta_H_req_pair[1][Axis.Z] = (0.5-alpha)* Delta_H_cmn

            Delta_H_res = np.zeros((2, 3))
            for j, pair in enumerate(self.Pair):
                Delta_H_res[j, :], sigma_fin[i, j], delta_fin[i, j], theta_fin[i, :, j]  = pair.require(Delta_H_req_pair[j])

            Delta_H_prod[i, :] = Delta_H_req - np.sum(Delta_H_res, axis=0)

            
            V_rew[i] = self.costRequest(Delta_H_prod[i, :], Delta_H_req) + \
                        self.costSingularity(delta_fin[i, :])
        
        # plt.figure(3)
        # plt.plot(V_alpha, V_rew)
        # plt.show()
        pass
        if type(self.V_rew_store) == type(None):
            self.V_rew_store = V_rew
        else:
            self.V_rew_store = np.column_stack((self.V_rew_store, V_rew))

        # cost based choice
        i = np.argmin(V_rew)

        # angle set and update
        for j, pair in enumerate(self.Pair):
            pair.set(theta_fin[i, :, j])

        V_theta_dot = Pair.angleDistance(self.theta_meas, self.theta_set)/self.dT
        tau_prod = Delta_H_prod[i, :] / self.dT
        m = V_rew[i]

        return V_theta_dot, tau_prod, m
    
    def update(self):
        for pair in self.Pair:
            pair.update()

    def costRequest(self, Delta_H_prod, Delta_H_req):
        # max value = k
        Delta_H_res = Delta_H_req - Delta_H_prod
        return self.k_torque* np.linalg.norm(Delta_H_res) / np.linalg.norm(Delta_H_req)
    
    def costSingularity(self, delta_fin):
        # delta is in range [0, pi/2] -> [external, internal]

        costEff = 0
        for i, pair in enumerate(self.Pair):
            costEff += np.tan(np.pi*2*self.delta_lim)**2 *pair.effCost(delta_fin[i])

        return self.k_sing* costEff

    @property
    def theta_meas(self):
        V_theta_meas = np.zeros(4)
        i = 0
        for pair in self.Pair:
            for cmg in pair.CMG:
                V_theta_meas[i] = cmg.theta_meas
                i += 1

        return V_theta_meas
    
    @property
    def theta_set(self):
        V_theta_set = np.zeros(4)
        i = 0
        for pair in self.Pair:
            for cmg in pair.CMG:
                V_theta_set[i] = cmg.theta_set
                i += 1
                
        return V_theta_set

    @property
    def H_meas(self):
        H_meas = np.zeros(3)
        for pair in self.Pair:
            H_meas = H_meas + pair.H_meas

        return H_meas
    
    @property
    def h(self):
        return self.Pair[0].h

    @property
    def D_alpha_max(self):
        return self.Pair[0].D_alpha_max



class Pair():
    def __init__(self, axis: Axis, fw_H, gb_alpha_max, theta_0):
        self.CMG = (CMG(axis, fw_H, gb_alpha_max, theta_0), CMG(axis, fw_H, gb_alpha_max, -theta_0))
        # this is maximum dTorque * dT / max dSigma or dTheta (CMG.alpha_max) 
        self.h = 2*fw_H                                                                                                                                         

    def require(self, Delta_H_req):

        H_req = self.cart2h_axis(self.H_meas + Delta_H_req)

        # |H|=2*h0*|cos(delta)|
        H_req_norm = np.linalg.norm(H_req)
        # external singularity (delta = 0)
        if H_req_norm > self.h:
            H_req_norm = self.h
            # flywheel Momentum parallel same sign
            delta_req = 0
        else:
            # is in the range [0 , 0.5]
            delta_req = 1/np.pi * acos( H_req_norm / self.h)

        # arctan of Imag [1] (X | Y) and Real [0] (Z)
        sigma_req = 1/np.pi * atan2(H_req[1], H_req[0]) # range (-1, 1]
        
        Sigma_req = np.array((sigma_req, sigma_req+1))
        Delta_req = np.array(((delta_req, -delta_req), (delta_req+1, -delta_req+1)))
        SigDel_meas = np.array((self.sigma_meas, self.delta_meas))

        SigDel_dist = np.Inf*np.ones(2)
        for i in range(2):
            for j in range(2):
                # i: select 0 or 1 shift
                # j: select + or - delta sign
                _SigDel_req = np.array((Sigma_req[i], Delta_req[i][j]))
                _SigDel_dist = Pair.angleDistance(SigDel_meas, _SigDel_req)

                if np.sum(np.abs(_SigDel_dist)) < np.sum(np.abs(SigDel_dist)):
                    SigDel_dist = _SigDel_dist


        

        # gimbal max speed saturation
        SigDel_dist = self.saturateRequest(SigDel_dist)

        sigma_done = self.sigma_meas + SigDel_dist[0]
        delta_done = self.delta_meas + SigDel_dist[1]
        theta_done = self.sd2theta(sigma_done, delta_done)
        
        Delta_H_done = self.deltaHprod(sigma_done, delta_done)

        Delta_H_res = Delta_H_req - Delta_H_done

        return Delta_H_res, sigma_done, delta_done, theta_done
    
    
    def saturateRequest(self, SigDel_dist):
        '''If request higher than possible, it saturate to maximum keeping same direction'''
        # # coefficient to be inside saturation zone with the same direction
        # gamma_SigDel = np.abs(SigDel_dist) / self.CMG[0].gb_alpha_max 

        # # important to get the maximum to get the axis with more saturation
        # gamma_max = np.max(gamma_SigDel)

        # # rescaling of the vector based on relative saturation
        # if gamma_max > 1:
        #     # unique division of the algorithm
        #     SigDel_dist = SigDel_dist / gamma_max



        ########### New ################
        rad_Delta = np.sum(SigDel_dist**2)
        if rad_Delta > self.D_alpha_max**2:
            gamma_SigDel = np.sqrt(rad_Delta) / self.D_alpha_max
            SigDel_dist = SigDel_dist / gamma_SigDel

        return SigDel_dist
    
    def deltaHprod(self, sigma_set, delta_set):
        H_in = self.H_meas
        H_fin = self.H(sigma_set, delta_set)

        return H_fin-H_in
    
    #### change  class property ####
    def set(self, theta_set):
        for i, cmg in enumerate(self.CMG):
            cmg.theta_set = theta_set[i]

    def update(self):
        for cmg in self.CMG:
            cmg.update()

    #### formulas utitily ####
    @staticmethod
    def theta2sd(theta):
        sigma = (theta[0] + theta[1])/2
        delta = (theta[0] - theta[1])/2

        return sigma, delta
    
    @staticmethod
    def sd2theta(sigma, delta):
        theta_0 = sigma + delta
        theta_1 = sigma - delta
        return np.array((theta_0, theta_1))
    
    @staticmethod
    def angleDistance(theta_ref, theta_new):
        return Pair.wrapAngle(theta_new - theta_ref, c=0)

    @staticmethod
    def wrapAngle(theta, c=0): # center of [0, 2) range
        '''Wrap angle in range centered in c [-1+c, 1+c) (*π)'''
        shift = 1 - c # shift to be in centered range
        theta = (theta + shift) % 2 - shift # wrap and shift back in center

        return theta

    def relativeDdelta(self, delta_fin):
        '''Movement respect to max available'''
        return Pair.angleDistance(delta_fin, self.delta_meas)/self.D_alpha_max
    
    def cart2h_axis(self, vect):
        '''Select /Real/ and /Imag/ axis from 3D vector'''
        v_h_axis = np.array([vect[self.H_axis.Real], vect[self.H_axis.Imag]])
        return v_h_axis

    
    @property
    def sigma_meas(self):
        sigma_meas, delta_meas = self.theta2sd(self.theta_meas)
        
        return sigma_meas

    @property
    def delta_meas(self):
        sigma_meas, delta_meas = self.theta2sd(self.theta_meas)
        
        return delta_meas
    
    @property
    def D_alpha_max(self):
        return self.CMG[0].gb_alpha_max
    

    #### CMG middleware ####
    @property
    def theta_meas(self):
        theta = np.array([self.CMG[0].theta_meas, self.CMG[1].theta_meas])
        return theta
    
    @property
    def theta_set(self):
        return np.array([self.CMG[0].theta_set, self.CMG[1].theta_set])


    def H(self, sigma, delta):
        H = np.zeros(3)

        H_abs = self.h* cos(np.pi*delta)

        H_re = H_abs * cos(np.pi*sigma)
        H_im = H_abs * sin(np.pi*sigma)
        H[self.H_axis.Real] = H_re
        H[self.H_axis.Imag] = H_im

        return  H
    
    @property
    def H_meas(self):
        '''Measured Momentum'''
        return  self.H(self.sigma_meas, self.delta_meas)
    
    def Eff(self, delta):
        '''Measure the 2D torque production capability = cos^2(delta)sin^2(delta)'''
        return 1- cos(np.pi*4*delta)
        
    @staticmethod
    def effCost(delta):
        '''Measure the cost due to nearby singularity'''
        return 1 / np.tan(np.pi*2*delta)**2
    

    @property
    def H_axis(self):
        '''Axis on which Pair produce torque'''
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

        self.gb_alpha_max = gb_alpha_max / np.pi

    def update(self):
        self.theta_meas = self.theta_set
        
    # gives the momentum variation with the actual setting
    @property
    def Delta_H(self):
        Delta_H = np.zeros(3)
        for axis in Axis:
            if axis == self.H_axis.Real:
                Delta_H[axis] = self.fw_H* (sin(self.theta_set) - sin(self.theta_meas))
            elif axis == self.H_axis.Imag:
                Delta_H[axis] = self.fw_H* ( cos(np.pi*self.theta_set) -  cos(np.pi*self.theta_meas))
            else:
                Delta_H[axis] = 0

        return Delta_H

