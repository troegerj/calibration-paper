function [un,q,stress,strain,yieldval,reactions] = gsolve(model,un,qn,coord,inz,rhs_nodalForce,outConv)
% DESCRIPTION
%  function for the iterative solution of the system of (non-)linear
%  equations g(u,q) = 0 representing the global level of the finite element
%  computation
%
% INPUT
%  model -- structure containing information about the specified model
%  un -- displacement array (ndof x 1)
%  qn -- internal variable state at tn (nq*nint x 1)
%  coord -- node coordinates
%  inz -- incidence matrix
%  rhs_nodalForce -- applied nodal forces
%  outConv -- switch for output of convergence information
%
% OUTPUT
%  un -- converged displacement solution at tn+1
%  q -- updated internal variable state at tn+1
%  stress -- stresses at integration points
%  strain -- strains at integration points
%  yieldval -- yielding indicators at integration points
%  reactions -- reaction force at prescribed displacement degrees of freedom

% extract quantities from model-structure
maxni = model.maxni;
tolg  = model.tolg;
toldu = model.toldu;
nelem = model.eset(1).nelem;
nintp = model.eset(1).nintp;
neq   = model.neq;

% initialize arrays for stresses, strains, and yield indicator
stress   = zeros(6,nintp,nelem);
strain   = zeros(6,nintp,nelem);
yieldval = false(1,nintp,nelem);

% iterations of Newton method
for k = 1 : maxni
    
    % build global system of linear equations
    [q,stress,strain,yieldval,stiff,rhs] = seteqs(model,un,qn,coord,inz,rhs_nodalForce);    
            
    % solve system of linear equations
    du = stiff(1:neq,1:neq)\rhs(1:neq);

    % compute norm of right-hand side
    normg = norm(rhs(1:neq));
    % compute norm of displacement update
    normdu = norm(du);
    
    % output of convergence information
    if outConv
        fprintf('Newton iteration: k = %d\n',k);
        fprintf('k = %d, \t conv. criteria: ||du|| < toldu & ||g|| < tolg \n',k);
        fprintf('k = %d, \t conv. tolerances: toldu = %8.4E, \t tolg = %8.4E \n',k,toldu,tolg);
        fprintf('k = %d, \t conv. values: ||du|| = %8.4E, \t ||g|| = %8.4E \n',k,normdu,normg);
    end
    
    % check termination criteria
    if (normg <= tolg) && (normdu <= toldu)
        % get reaction forces at prescribed displacement degrees of freedom
        reactions = -rhs(neq+1:end); 
        % exit
        break
    end

    % update displacements
    un(1:neq) = un(1:neq) + du;

end

if (normg > tolg) || (normdu > toldu)
    error('%s: Newton method did not converge within maximum number of iterations',mfilename);
end

end