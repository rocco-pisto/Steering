%{
Steering is a superclass, used to define all the steering algorithm as
steering object, and so avoiding to repeat the same calculation, as the
singularity metrics in each code.

%}



classdef Steering < handle
    %{ 
      Define all the variables used in the classes to have inside the obj. structure 
    %}
    properties
        %
        structure
        m0
        H
        HCMG
        A
        A_v
        d
        m
        null_motion
        deter
        HCMG_v
        m_v
        %null_motion_v
        deter_v
        delta_dots_v
        delta_cs
        delta
        singularity
        S
        S_v
        U 
        U_v
        V
        V_v
        rate_lg
        rate_sda
        OmegaFW
        Omegas_dot
        deltas_ddot
        P_wheels
        OmegaFW_initial
        d_v
        eigs_A
        s_v
        s
        flag
        tau_n
        eta_dot_n
        HCMG_max

    end

    methods
        function obj = Steering(structure, m0, tspan)
            obj.structure = structure;
            % Initialize the vector to improve efficiency of the code
            obj.m0 = m0;
            obj.flag = obj.m0;
            obj.delta_cs = zeros(4,1);
            obj.delta = NaN*zeros(4,length(tspan));
            obj.delta(:,1) = zeros(4,1);
            obj.delta_dots_v = NaN*zeros(4,length(tspan));
            obj.deltas_ddot =  NaN*zeros(4,length(tspan));
            obj.deter_v = zeros(1,length(tspan));
            obj.m_v =  zeros(1,length(tspan));
            obj.S_v = NaN*zeros(3,length(tspan));
            obj.V_v = NaN*zeros(4,4,length(tspan));
            obj.U_v = NaN*zeros(3,3,length(tspan));
            obj.rate_lg = NaN*zeros(4,length(tspan));
            obj.rate_sda = NaN*zeros(4,length(tspan));
            obj.d_v = NaN*zeros(4,length(tspan));
            obj.eigs_A = NaN*zeros(3,length(tspan));
            obj.s_v = NaN*zeros(3,length(tspan));
            obj.s = zeros(3,1);
            % obj.null_motion_v =  zeros();
            obj.HCMG_v = NaN*zeros(3,length(tspan));
            obj.A_v = NaN*zeros(3,4,length(tspan));
            obj.OmegaFW = NaN*zeros(4,length(tspan));
            obj.Omegas_dot = NaN*zeros(4,length(tspan));
            obj.P_wheels = NaN*zeros(2,length(tspan));
            obj.OmegaFW_initial = obj.structure.OmegaFW_nominal*ones(4,1);
            obj.eta_dot_n = NaN*zeros(8,length(tspan));
            obj.tau_n = NaN*zeros(3,length(tspan));
            obj.HCMG_max = NaN*zeros(3,length(tspan));
        end

        function [U, S, V, mu, Q,s] = instantParam(obj, delta)
            %% Compute instantaneous parameter
            % Total CMG system angular momentum
            obj.H = obj.structure.H(delta);

            % Angular momentum from the CMG
            obj.HCMG = obj.structure.HCMG(delta);

            % Jacobian matrix
            obj.A = obj.structure.A(delta);

            % null vector ( for the moment just copied the function d_function_4)
            obj.d = obj.structure.d_function(delta);

            % singularity measure
            obj.m = sqrt(abs(det(obj.A*obj.A')));
             % obj.m = (abs(det(obj.A*obj.A')));

            % Null motion
            obj.null_motion = sum(obj.HCMG);

            %% Internal singular metrics
            [U,S,V] = svd(obj.A, "vector");

            % check = abs(V(:,:))<0.01;
            % if ~isempty(check)
            %     V(check) = 0;
            % end

            obj.V = V;
            obj.U = U;
            % Checking if we are close to a singularity
            if obj.m>obj.m0 % 
                % away from singularity it should be R^(1x1)
                Q=zeros(1,1);
                obj.deter = NaN;
                mu = 0;
                obj.singularity= 'No';
                obj.s = zeros(3,1);
                obj.flag = obj.m0;
            else
                %  config.sing = config.sing +1;
                % singular direction
                obj.flag = 0.5;

                s = null(obj.A');
                obj.s = s;

                if isempty(s)

                    % last columns of the U matrix is the closest axis to the singularity
                    s=U(:,end);
                     % s = abs(s);
                    obj.s = s;
                    N=V(:,end-1:end);
                    % s_v = ~isnan(obj.s_v)
                    % s = s_v(:,end)
                else
                    % Null space
                    N = null(obj.A);
                end

                % check = abs(N(:,:))<0.01;
                % if ~isempty(check)
                %     N(check) = 0;
                % end 
                % N = abs(N)
                % N = null(obj.A)
                % disp(rank(obj.A))
                % % disp(N)
                % % disp(s)
                % disp(S)

                
    

                % singularity projection matrix
                P = diag(obj.H'*s);

                if sum(sign(obj.H'*s)) == obj.structure.nCMG
                     mu = 0;
                else
                    mu = 1;
                end

                % Q matrix
                Q = N'*P*N;

                % check type of singularity, if >0 elliptic, if 0 or <0
                % hyperbolic
                % separate from internal or external
                obj.deter = det(Q);

                toll = 0.6;
                 if abs(norm(obj.HCMG))>min(obj.structure.max_h0)-toll
                % if abs(norm(obj.HCMG))>3
                    obj.singularity = 'External';
                else
                    if obj.deter<=0
                        obj.singularity = 'Internal_Hyperbolic';
                    else
                        obj.singularity = 'Internal_Elliptic';
                    end
                end

            end

            obj.S = S;


        end

        function delta_dots = saturation(obj, delta_dots)

            %% UPPER saturation on the gimbal rate
            check = abs(delta_dots) >= obj.structure.max_delta_dot;
            ind = find(check==1);
            delta_dots(ind) = obj.structure.max_delta_dot(ind).*sign(delta_dots(ind));

          %% LOWER saturation on the gimbal rate
            check = abs(delta_dots) < obj.structure.min_delta_dot;
            ind = find(check==1);
            delta_dots(ind) = 0; %obj.structure.min_delta_dot.*sign(delta_dots(ind));


        end
     %% TO be improved, it has sense to not to all the calculation of the algorithm if we are in a external singularities
     % function delta_dots = Momentum_saturation(obj, h_dot,delta_dots,config)
     %        dt = 1/config.fs;
     %        threshold = 0.6;
     %        H_next = obj.HCMG +h_dot*dt;
     %        % if H_next>(min(obj.structure.max_h0)-threshold)
     %        if norm(H_next)>(min(obj.structure.max_h0)-threshold)
     %            delta_dots = zeros(4,1);
     %        end
     % 
     % 
     %    end


        function storeSimData(obj, delta_dots,delta,i)
            obj.m_v(:,i+1) =  obj.m;
            obj.HCMG_v(:,i+1) = obj.HCMG;
            obj.delta_dots_v(:,i+1) =  delta_dots;
            obj.deter_v(:,i+1) =  obj.deter;
            obj.S_v(:,i+1) = obj.S;
            obj.V_v(:,:,i+1) = obj.V;
            obj.U_v(:,:,i+1) = obj.U;
            obj.delta(:,i+1) = delta;
             obj.d_v(:,i+1) = obj.d;
            obj.eigs_A(:,i+1) = svd(obj.A,"vector");
            obj.s_v(:,i+1) = obj.s;
            obj.A_v(:,:,i+1) =obj.A;


            n = abs(obj.delta_cs(:))<abs(delta);
            len = find(n==1);
            if ~isempty(len)
                %check
                obj.delta_cs(len) = delta(len);

            end
        end
    end
end


