function dFdx = numerical_differentiation(x0,fun_handle,method)
% function for numerical differentiation of a function given as function
% handle. Implementation allows for forward and central differences.
%
% INPUT
%  x0 -- argument vector
%  fun_handle -- function handle 
%  method -- method switch (1 -- forward differences, 2 -- central differences)
%
% OUTPUT
%  dFdx -- functional matrix

% choose epsdif as variation according to Press et al. - Numerical recipes
% in Fortran
if method == 1
    epsdif = 1.e-8; % approx. eps^(1/2)
elseif method == 2
    epsdif = 1.e-6; % approx. eps^(1/3)
end

% set up argument vector
xv = x0;
xsav = x0;
h = epsdif*abs(xsav);
% correction of h to prevent from small values
logic_index_h = h<epsdif;
h(logic_index_h) = epsdif;
xph = xsav + h;
xpnh = xsav - h; % only required for central differences

switch method
    case 1 % forward differences
                
        % compute function values at given arguments
        Lf0 = fun_handle(x0);        
        % initialize functional matrix
        dFdx = zeros(length(Lf0),length(x0));
                
        % compute functional matrix using forward differences
        for i = 1 : length(x0)
            % change arguments
            xv(i) = xph(i);
            
            % evaluate function
            Lf1 = fun_handle(xv);
            
            % compute column in functional matrix
            dFdx(:,i) = (Lf1-Lf0)/h(i);
            
            % reset arguments
            xv(i) = xsav(i);
        end
        
    case 2 % central differences
                                   
        % compute functional matrix using central differences
        for i = 1 : length(x0)
            % change arguments
            xv(i) = xph(i);   

            % evaluate function for x+h
            Lf1 = fun_handle(xv);
   
            if i == 1
                % initialize functional matrix
                dFdx = zeros(length(Lf1),length(x0));
            end
                
            % change arguments
            xv(i) = xpnh(i); 

            % evaluate function for x-h
            Lf0 = fun_handle(xv);
                        
            % compute column in functional matrix
            dFdx(:,i) = (Lf1-Lf0)/(2*h(i));
            
            % reset arguments
            xv(i)=xsav(i);
        end               
end

end