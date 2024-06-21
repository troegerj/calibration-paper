function [result] = integrate(model,bcond,load,lfun,coord,inz,ID,outConv)
% DESCRIPTION
%  function to perform the time-stepping for solving the initial
%  boundary-value problem
%
% INPUT
%  model -- structure containing information about the specified model
%  bcond -- array with boundary condition information
%  load -- applied nodal loads
%  lfun -- load functions
%  coord -- node coordinates
%  inz -- incidence matrix
%  ID -- ID-array
%  outConv -- switch for output of convergence information
%
% OUTPUT
%  result -- results structure

% extract quantities from model-structure
nintp = model.eset(1).nintp;
nelem = model.eset(1).nelem;
maxsteps = model.maxsteps;
ndof = model.ndof;
ndim = model.ndim;
nq = model.nq;
nbc = model.nbc;
nloadbc = model.nloadbc;
neq = model.neq;
ndf = model.ndf;
dt = model.dt;
out = model.out;

% initialize quantities computed in each time-step
stress = zeros(6,nintp,nelem);
strain = zeros(6,nintp,nelem);
yieldval = false(1,nintp,nelem);
un = zeros(ndof,1);
qn = zeros(nq,nintp,nelem);

% initialize output quantities
nt_out = 0;
if out.disp
    result.disp = zeros(ndof,size(out.time,1));
end
if out.stress
    result.stress = zeros(6,nintp,nelem,size(out.time,1));
end
if out.strain
    result.strain = zeros(6,nintp,nelem,size(out.time,1));
end
if out.yieldval
    result.yieldval = zeros(1,nintp,nelem,size(out.time,1));
end
if out.innerVar 
    result.innerVar = zeros(nq,nintp,nelem,size(out.time,1));
end
if out.reactionForce
    result.reactionForce = zeros(ndof-neq,size(out.time,1));
end

% initialize time information
time = zeros(maxsteps,1);
time(1) = model.t0;

% check for nodal forces
if nloadbc > 0
    rhs_nodalForce = sparse(ndof,1);
else
    rhs_nodalForce = [];
end

% loop over all time-steps
for n = 1 : maxsteps

    % output
    if outConv
        fprintf('\n');
        fprintf('integrate.m: tn = %8.8E, \t dt = %8.8E \n',time(n),dt);
    end

    % compute tn+1
    time(n+1) = time(n) + dt;

    % set boundary conditions
    %
    % Dirichlet (displacement) boundary conditions
    for i = 1 : nbc
        % global node number
        node = bcond(i,1);

        for j = 1 : ndf
            % load function number
            function_number = bcond(i,j+1);
            % assign boundary condition
            if function_number ~= 0
                % check entry in ID-array
                if ID(j,node) <= neq
                    error('%s: degree of freedom for boundary condition is lower than neq',mfilename);
                end
                %
                fac            = bcond(i,ndf+1+j);
                un(ID(j,node)) = fac*lfun{function_number}(time(n+1));
            end
        end
    end
    %
    % Neumann (force) boundary conditions
    for i = 1 : nloadbc
        % global node number
        node = load(i,1);

        for j = 1 : ndim
            % load function number
            function_number = load(i,j+1);
            % assign boundary condition
            if function_number ~= 0
                % check entry in ID-array
                if ID(j,node) > neq
                    error('%s: Load can not be set for known variable',mfilename);
                end
                %
                fac = load(i,ndf+1+j);
                rhs_nodalForce(ID(j,node)) = fac*lfun{function_number}(time(n+1));
            end
        end
    end

    % solve system of non-linear equations g(u,q) = 0 for u
    [un,q,stress,strain,yieldval,reactions] = gsolve(model,un,qn,coord,inz,rhs_nodalForce,outConv);

    % assign output quantities
    if ismembertol(time(n+1),out.time,1.e-6)
        nt_out = nt_out + 1;
        if out.disp
            result.disp(:,nt_out) = un(:,1);
        end
        if out.stress
            result.stress(:,:,:,nt_out) = stress;
        end
        if out.strain
            result.strain(:,:,:,nt_out) = strain;
        end
        if out.yieldval
            result.yieldval(:,:,:,nt_out) = yieldval;
        end
        if out.innerVar
            result.innerVar(:,:,:,nt_out) = q;
        end
        if out.reactionForce
            result.reactionForce(:,nt_out) = reactions;
        end
        result.time(nt_out,1) = time(n+1);
    end

    % exit loop if time-step size becomes to small
    if (time(n+1)-model.tend) > -1.e-10
        break;
    end

    % adapt time-step size to end of time interval
    if (time(n+1)+dt) >= model.tend
        dt = model.tend - time(n+1);
    end

    % update internal variables for next time-step
    qn = q;

end

% reduce time array
time(n+1:end) = [];

end