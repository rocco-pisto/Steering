classdef SAC < Steering
    properties
        sac = 0;
    end

    methods
        function obj = SAC(tspan, k_torque_err, k_sing, sigma_sing)
            structure = 0;
            m0 = 0;
            dt_update = 0 % ????
            obj@Steering(structure, m0, tspan);
            obj.sac = py.Mananager(dt_update, k_torque_err, k_sing, sigma_sing);

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

        function storeSimData(obj, delta_dots, delta, i)
            storeSimData@Steering(obj, delta_dots, delta, i);
        end
    end
end


