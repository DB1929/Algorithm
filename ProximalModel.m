classdef ProximalModel
    properties 
    % Statistical Information
    %      E_p  : Mean of Postive  Data X Number of Postive 
    %      E_n  : Mean of Negative Data X Number of Postive
    %      E_pp : Mean of Postive  Data Square X Number of Postive  
    %      E_nn : Mean of Postive  Data Square X Number of Postive 
    %      n_p  : Number of Postive  Data
    %      n_n  : Number of Negative Data
        E_p
        E_pp
        E_n
        E_nn
        n_p
        n_n
    % Proximal Model
    %      wp      : Proximal Model weight
    %      bp      : Proximal Model bias
    %      grad_wp : Gradient of Proximal model
    %      grad_bp : Gradient of Proximal model
    %      C       : Trade-off parameter
        wp
        bp
        grad_wp
        grad_bp
        C
        
    end
    
    methods
        function obj = ProximalModel(C)
            obj.E_p  = 0;
            obj.E_pp = 0;
            obj.E_n  = 0;
            obj.E_nn = 0;
            obj.n_p  = 0;
            obj.n_n  = 0;           
            obj.wp   = 0;
            obj.bp   = 0;
            obj.grad_wp = 0;
            obj.grad_bp = 0;
            obj.C  = C;                  
        end
        
        
        function obj = StatInfo(obj,Inst,Label)
            ind_p = find(Label>0);
            ind_n = find(Label<0);
            obj.n_p = obj.n_p + size(ind_p,1);
            obj.n_n = obj.n_n + size(ind_n,1);
            X_p = Inst(ind_p,:);
            X_n = Inst(ind_n,:);
            obj.E_p   = obj.E_p  + sum(X_p,1);
            obj.E_n   = obj.E_n  + sum(X_n,1);
            obj.E_pp  = obj.E_pp + sum(X_p.^2,1);
            obj.E_nn  = obj.E_nn + sum(X_n.^2,1);      
        end
        
        function obj = getProx(obj)
            % Mean
            m_p = obj.E_p/obj.n_p;
            m_n = obj.E_n/obj.n_n;
            % Variance
            v_p = (obj.E_pp - (obj.E_n.^2)*obj.n_p)/(obj.n_p-1);
            v_n = (obj.E_nn - (obj.E_n.^2)*obj.n_n)/(obj.n_n-1);
            %Proximal Model
            obj.wp = ((m_p - m_n)./(v_p + v_n + eps))';
            obj.bp = (-1/2)*(m_p + m_n)*obj.wp;
        end
        
        function obj = Grad_prox(obj)
            obj.grad_wp = obj.C * obj.wp;
            obj.grad_bp = obj.C * obj.bp;       
        end

        
        
    end
    
    
end