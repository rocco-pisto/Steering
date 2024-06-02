classdef SAC < Steering
    properties
        sac = 0;
    end

    methods
        function obj = SAC(structure, tspan, k_torque_err, k_sing, sigma_sing, dT)
            fw_H = structure.h0;
            gb_alpha_max = structure.max_delta_dot * dT;
            obj@Steering(structure, m0, tspan);
            obj.sac = py.Mananager(fw_H, gb_alpha_max, k_torque_err, k_sing, sigma_sing, dT);

            % dummy fields to avoid problem of compatibility with the superclass storeSimData
            obj.deter = 0;
            obj.A = 0;
            obj.U = 0;
            obj.S = 0;
            obj.V = 0;
            obj.d = 0;
            obj.s = 0;
        end

        function [delta_dots, tau_r, m] = algorithm(obj, tau_c, delta, i)
            [delta_dots, tau_r, m] = obj.sac.require(tau_c, delta);
            obj.HCMG = obj.sac.H_meas;
            obj.m = m;

            storeSimData@Steering(obj, delta_dots, delta, i);
        end

    end
end


