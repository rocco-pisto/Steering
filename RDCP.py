
from collections import namedtuple
from math import cos, sin, asin, acos, atan2
import numpy as np
from scipy.optimize import fminbound
from matplotlib import pyplot as plt

CMG = namedtuple('CMG', ['fw_H', 'fw_rpm_min', 'fw_rpm_max', 'fw_acc_max', 'gb_rate_min', 'gb_rate_max'])

class Rooftop():
    '''Vector indexes for torque -> (X, Y, Z) in CARTISIAN <-> (X, U, V) in CLUSTER reference system'''
    X = 0
    Y = 1
    Z = 2
    U = 1
    V = 2

    RE = 0  # real (X) optimized by algorithm
    IM = 1  # imag (U | V) satistied by single pair

    def __init__(self, skew) -> None:
        '''Transformation matrices defined from the skew angle [degrees]'''
        skew = np.deg2rad(skew)

        self.beta = skew
        
        self.Mcart2clust = np.array((\
                            (1, 0, 0),                  \
                            (0, cos(skew), sin(skew)),  \
                            (0, cos(skew), -sin(skew))  \
                            ))
        
        inv_det = 1/(-2*cos(skew)*sin(skew))
        self.Mclust2cart = np.array((\
                            (1, 0, 0),                                  \
                            (0, -inv_det*sin(skew), -inv_det*sin(skew)),\
                            (0, -inv_det*cos(skew), inv_det*cos(skew))  \
                            ))

    def cart2clust(self, v_cart):
        v_clust = np.zeros(3)

        v_clust = np.matmul(self.Mcart2clust, v_cart)
        return v_clust

    def clust2cart(self, v_clust):
        v_cart = np.zeros(3)

        v_cart = np.matmul(self.Mclust2cart, v_clust)
        return v_cart
    

class RDCP():
    '''Rooftop Decoupled CMGs Pair'''
    cmg = CMG( \
        fw_H        = 1, 
        fw_rpm_min  = 1500, 
        fw_rpm_max  = 10000, 
        fw_acc_max  = 90,       # rpm/s
        gb_rate_min = 1/16,     # deg/s
        gb_rate_max = 90)       # deg/s
    

    def __init__(self, skew, k_torque, delta_unit_cost, dt):
        # Pair[0] produce on X, U
        # Pair[1] produce on X, V

        Pair.setPair(RDCP.cmg, dt, delta_unit_cost)

        # To have H_meas = 0 at startup we need:
            # Pair[0] -> theta_0 = 60,  theta_1 = -60
            # Pair[1] -> theta_0 = 120, theta_1 = -120

        self.Pair = (   Pair(60,  -60), \
                        Pair(120, -120) )
        self.v_theta_set = np.zeros((2,2))
        self.v_theta_set[0, 0] = 60
        self.v_theta_set[0, 1] = -60
        self.v_theta_set[1, 0] = 120
        self.v_theta_set[1, 1] = -120
        self.v_theta_meas = np.zeros((2,2)) 
        
        # print(self.H_meas)

        self.cluster = Rooftop(skew)
        
        self.k_torque = k_torque

        self.dt = dt

        self.simulate = True

    def require(self, tau_req, v_theta_meas=None):
        '''Require torque from the cluster'''
        
        self.updateMeas(v_theta_meas)

        d_H_req = tau_req*self.dt  # the delta momentum to satisfy [reference (X,Y,Z)]
        d_H_req_clust = self.cluster.cart2clust(d_H_req) # the delta momentum to satisfy [reference (X,U,V)]

        d_H_req_pair = np.zeros((2, 2))
        d_H_req_pair[0, Rooftop.IM] = d_H_req_clust[Rooftop.U]
        d_H_req_pair[1, Rooftop.IM] = d_H_req_clust[Rooftop.V]

        d_H_cmn = d_H_req_clust[Rooftop.X]

        N_p = 41
        V_alpha = np.linspace(-1, 1, N_p)
        cost_min = np.Inf
        alpha_opt = 0
    
        for i, alpha in enumerate(V_alpha):
            cost = self.cost(alpha, d_H_cmn, d_H_req_pair, d_H_req)
            
            if cost < cost_min:
                cost_min = cost
                alpha_opt = alpha
        

        d_H_err, cost, _theta_set = self.solve(alpha_opt, d_H_cmn, d_H_req_pair, d_H_req)
        self.v_theta_set = _theta_set / Pair.fact_deg2range

        d_H_done = d_H_req - d_H_err

        _theta_meas = self.v_theta_meas * Pair.fact_deg2range
        v_gb_rate = Pair.angleDistance(_theta_meas, _theta_set)/self.dt
        v_gb_rate /= Pair.fact_deg2range

        tau_prod = d_H_done / self.dt
        m = 1 / cost_min

        return v_gb_rate, tau_prod, m
    
    def solve(self, alpha: float, d_H_cmn: float, d_H_req_pair, d_H_req):
        '''Solve the request torque problem for a specific alpha'''
        d_H_req_pair[0, Rooftop.RE] = 0.5* d_H_cmn +alpha*Pair.d_H_max
        d_H_req_pair[1, Rooftop.RE] = 0.5* d_H_cmn -alpha*Pair.d_H_max

        d_H_err_pair = np.zeros((2, 2))
        v_theta_set = np.zeros((2, 2))
        cost_sing = np.zeros(2)

        for i, pair in enumerate(self.Pair):
            d_H_err_pair[i, :], cost_sing[i], v_theta_set[i, :]  = pair.require(d_H_req_pair[i])

        ######## COST SINGULARITY
        cost_sing = np.sum(cost_sing)

        ######## COST TORQUE ERROR
        d_H_err = self.pair2cart(d_H_err_pair)
        cost_torque_err =  self.k_torque * np.linalg.norm(d_H_err) / np.linalg.norm(d_H_req)

        cost = cost_torque_err + cost_sing

        return d_H_err, cost, v_theta_set

    def cost(self, alpha: float, d_H_cmn: float, d_H_req_pair, d_H_req) -> float:
        '''Compute the total cost to optimize'''
        _, cost, _ = self.solve(alpha, d_H_cmn, d_H_req_pair, d_H_req)

        return cost

    def updateMeas(self, v_theta_meas=None):
        '''v_theta_meas[2][2] is the measured angle for the two cmg of the two pairs'''
        if self.simulate:
            self.v_theta_meas = self.v_theta_set

        for i, pair in enumerate(self.Pair):
            pair.updateMeas(self.v_theta_meas[i])

    def pair2cart(self, v_pair):
        '''Transform 2x2 matrix of pairs to 3D cartisian vector'''
        v_cart = np.zeros(3)
        v_cart[Rooftop.X] = v_pair[0, Rooftop.RE] + v_pair[1, Rooftop.RE]
        v_cart[Rooftop.U] = v_pair[0, Rooftop.IM]
        v_cart[Rooftop.V] = v_pair[1, Rooftop.IM]
        v_cart = self.cluster.clust2cart(v_cart)

        return v_cart

class Pair():
    cmg = CMG
    h = 0
    d_theta_max = 0
    d_H_max = 0

    delta_unit_cost = 0

    fact_deg2range = 1/180 # factor to transform angle from [0, 360] -> [0, 2]
    fact_rad2range = 1/np.pi # factor to transform angle from [0, 2*pi] -> [0, 2]
    ext_th = 0.95

    def __init__(self, theta_0, theta_1):
        '''Set initial angle [degrees]'''
        self.theta_meas = np.array((theta_0, theta_1))
        self.theta_meas = self.theta_meas* Pair.fact_deg2range


    def updateMeas(self, v_theta_meas):
        self.theta_meas[:] = v_theta_meas[:] * Pair.fact_deg2range

    @staticmethod
    def setPair(cmg: CMG, dt, delta_unit_cost):
        Pair.cmg = cmg

        # max total momentum of the pair (amplitude)        
        Pair.h = 2*cmg.fw_H

        # max gimbal angle variation in the steering loop time step
        # gb_rate_max [deg/s] -> *dt 
        # = d_theta_max [degrees] -> *fact_deg2range
        # = d_theta_max         
        Pair.d_theta_max = cmg.gb_rate_max* dt * Pair.fact_deg2range

        # max delta momentum in steering loop time step
        Pair.d_H_max = Pair.h * (Pair.d_theta_max / Pair.fact_rad2range)

        Pair.delta_unit_cost = delta_unit_cost * np.pi/180

    ################################################################
    # ALGORITHM METHODS
    ################################################################
    def require(self, d_H_req):

        H_req = self.H_meas + d_H_req

        # |H|=2*h0*|cos(delta)|
        H_req_norm = np.linalg.norm(H_req)
        # external singularity (delta = 0)
        
        if H_req_norm > Pair.ext_th*self.h:
            H_req_norm = Pair.ext_th*self.h
            # flywheel Momentum parallel same sign
            delta_req = Pair.fact_rad2range * acos( Pair.ext_th)
        else:
            delta_req = Pair.fact_rad2range * acos( H_req_norm / self.h) # range [0 , 1]

        # arctan of Imag [1] (U | V) and Real [0] (Z)
        sigma_req = Pair.fact_rad2range * atan2(H_req[1], H_req[0]) # range (-1, 1]
        
        Sigma_req = np.array((sigma_req, sigma_req+1))
        Delta_req = np.array(((delta_req, -delta_req), (delta_req+1, -delta_req+1)))
        SigDel_meas = np.array((self.sigma_meas, self.delta_meas))

        d_SigDel = np.Inf*np.ones(2)
        for i in range(2):
            for j in range(2):
                # i: select 0 or 1 shift
                # j: select + or - delta sign
                _SigDel_req = np.array((Sigma_req[i], Delta_req[i][j]))
                _d_SigDel = Pair.angleDistance(SigDel_meas, _SigDel_req)

                if np.sum(np.abs(_d_SigDel)) < np.sum(np.abs(d_SigDel)):
                    d_SigDel = _d_SigDel


        

        # gimbal max speed saturation
        d_SigDel = self.saturateRequest(d_SigDel)

        sigma_delta_done = self.sigma_delta_meas + d_SigDel
        sigma_done = sigma_delta_done[0]
        delta_done = sigma_delta_done[1]
        theta_done = Pair.sd2theta(sigma_done, delta_done)
        
        d_H_done = self.deltaHprod(sigma_delta_done)

        d_H_err = d_H_req - d_H_done

        cost = Pair.costSingularity(sigma_delta_done[1])

        return d_H_err, cost, theta_done
    
    def saturateRequest(self, d_SigDel):
        '''If request higher than possible, it saturate to maximum keeping same direction'''

        d_Theta = Pair.sd2theta(d_SigDel[0], d_SigDel[1])

        if any(np.abs(d_Theta) > self.d_theta_max):
            k = np.max(np.abs(d_Theta)) / self.d_theta_max
            d_Theta /= k

        d_SigDel = Pair.theta2sd(d_Theta)
        
        return d_SigDel
    
    def deltaHprod(self, sigma_delta_set):
        H_in = self.H_meas
        H_fin = self.H(sigma_delta_set)

        return H_fin-H_in

    ################################################################
    # PROPERTY FUNCTIONAL COMPUTATION
    ################################################################
    @property
    def sigma_delta_meas(self):
        return Pair.theta2sd(self.theta_meas)

    @property
    def sigma_meas(self):
        return Pair.theta2sd(self.theta_meas)[0]

    @property
    def delta_meas(self):
        return Pair.theta2sd(self.theta_meas)[1]
        
    @property
    def H_meas(self):
        '''Measured Momentum'''
        return  Pair.H(self.sigma_delta_meas)

    


    ################################################################
    # UTILITY FORMULAS
    ################################################################
    @staticmethod
    def theta2sd(theta):
        sigma = (theta[0] + theta[1])/2
    
        delta = (theta[0] - theta[1])/2

        return np.array((sigma, delta))
    
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
    
    @staticmethod
    def efficiency(delta):
        '''Measure the 2D torque production capability = cos^2(delta)sin^2(delta)'''
        return 1 - cos(np.pi*4*delta)
        
    @staticmethod
    def costSingularity(delta):
        '''Measure the cost due to nearby singularity [cotangent(2*delta)^2]'''
        cost_sing = np.tan(Pair.delta_unit_cost)**2 / np.tan(2*delta / Pair.fact_rad2range)**2
        return cost_sing
        
    @staticmethod
    def H(sigma_delta):
        H = np.zeros(2)
        sigma = sigma_delta[0]
        delta = sigma_delta[1]

        H_abs = Pair.h* cos(np.pi*delta)

        H_re = H_abs * cos(np.pi*sigma)
        H_im = H_abs * sin(np.pi*sigma)
        H[0] = H_re
        H[1] = H_im

        return  H