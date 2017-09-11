classdef YellowFin
% YellowFin Optimizer
    properties

        % About Gradient 
        g_t
        g_avg
        g_p_avg
        g_n_avg
        % About h
        h_t
        h_max
        h_min
        h_win
        h_avg
        % Parameters
        width
        beta
        Variance
        Distance
        % Momentum Parameters
        l_r
        l_r_t
        mu
        mu_t

    end

    methods


        function obj = YellowFin(beta,width,LearningRate)
            % Constructor
            obj.width = width;
            obj.beta  = beta;

            obj.h_max = 0;
            obj.h_min = 0;
            obj.h_win = []; 
            obj.h_avg = 0;

            obj.g_p_avg = 0;
            obj.g_avg   = 0;
            obj.g_n_avg = 0;

            obj.Distance  = 0;
            obj.mu  = 0;
            obj.l_r = LearningRate;
        end


        function obj = CurvatureRange(obj)    
        % CurvatureRange
            obj.h_t = obj.g_t'* obj.g_t;
            if length(obj.h_win)< obj.width
                obj.h_win = [obj.h_win,obj.h_t];
            else
                obj.h_win = [obj.h_win(2:end),obj.h_t];
            end
            obj.h_max = obj.beta*obj.h_max + (1-obj.beta)*max(obj.h_win);
            obj.h_min = obj.beta*obj.h_min + (1-obj.beta)*min(obj.h_win);
        end

        function obj = GradientVariance(obj)
            obj.g_p_avg = obj.beta*obj.g_p_avg + (1-obj.beta)*obj.g_t.*obj.g_t;
            obj.g_avg   = obj.beta*obj.g_avg   + (1-obj.beta)*obj.g_t;
            obj.Variance = norm((obj.g_p_avg-obj.g_avg .*obj.g_avg),1);
        end

        function obj = OptDistance(obj)
            obj.g_n_avg   = obj.beta*obj.g_n_avg  + (1-obj.beta)*(obj.h_t^(1/2));
            obj.h_avg     = obj.beta*obj.h_avg    + (1-obj.beta)*obj.h_t;
            obj.Distance  = obj.beta*obj.Distance + (1-obj.beta)* (obj.g_n_avg/obj.h_avg);
        end


        function obj = SingleStep(obj)
           % mu,l_r argmin mu*D^2 + l_r^2*C
           % x     = mu^(1/2) , y = x-1
           % f(y)  = (y+1)^2*D^2 + y^4*(C/h_min^2)
           % df/dy = 0 => y^3 + py = q , p = (h_min*D)^2/(2C) = -q
           % Let y = (z - p/(3z))
           % z^3 - p^3/(3z)^3 - q = 0 => (z^3)^2 - q*z^3 - (p^3/27)
           % z^3 = [ q +/- (q^2 + (4/27)*p^3)^(1/2) ]/ 2
            p  = (obj.h_min*obj.Distance)^2/(2*obj.Variance);
            q  = -p;
            z3 = (q - (q^2 + 4*(p^3/27))^(1/2))/2;
            z  = sign(z3)*abs(z3)^(1/3);
            y  = (z-p/(3*z));
            x  = y + 1;

            hh = (obj.h_max/obj.h_min)^(1/2);
            obj.mu_t  = max(x^2,((hh-1)/(hh+1))^2);
            obj.l_r_t = (1-obj.mu_t)^2/obj.h_min;
        end

        function obj = YELLOWFIN(obj,g_t)
           obj.g_t = g_t;
           obj = CurvatureRange(obj);   
           obj = GradientVariance(obj);
           obj = OptDistance(obj);
           obj = SingleStep(obj);
           obj.mu  = obj.beta*obj.mu  + (1-obj.beta)*obj.mu_t; 
           obj.l_r = obj.beta*obj.l_r + (1-obj.beta)*obj.l_r_t;
        end
    end
end
