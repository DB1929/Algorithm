classdef ProximalModel
    properties 
    % Statistical information
    %      E_p : Mean of Postive  Data X Number of Postive 
    %      E_n : Mean of Negative Data X Number of Postive
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
    end
    
    methods
        
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
        
        function obj = Prox(obj)
            
        
        
        end
        
        
        
        
        
        
    end
    
    
end