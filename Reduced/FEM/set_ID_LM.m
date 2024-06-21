function [ID,LM,neq,ncs] = set_ID_LM(model,bcond,inz)
% DESCRIPTION
%  function to create the ID- and LM-array (see Hughes - The finite element
%  method as reference)
%
% INPUT
%  model -- structure containing information about the specified model
%  bcond -- array with boundary condition information
%  inz -- incidence matrix
%
% OUTPUT
%  ID -- ID-array
%  LM -- localisation matrix
%  neq -- number of unknown degrees of freedom
%  ncs -- number of constrained degrees of freedom

% extract information from model-structure
nnode = model.nnode;
nelem = model.eset(1).nelem;
npe   = model.eset(1).npe;
ndof  = model.ndof;
ndf   = model.ndf;
nbc   = model.nbc;

%% ID-array

% initialize with -1
ID = -ones(ndf,nnode);

% consider boundary conditions, prescribed DOF with 0
for i = 1 : nbc
    for dof = 1 : ndf
        if (bcond(i,1+dof) ~= 0)
            ID(dof,bcond(i,1))= 0;
        end
    end
end

% determine number of constrained degrees of freedom
ncs  = 0;
if (nbc > 0)
    ncs = sum(sum(bcond(:,2:1+ndf)~=0));
end

% number of unknown degrees of freedom
neq = ndof-ncs;

% set entries in ID-array
dof_free = 1;         % counter unknown degrees of freedom
dof_pres = neq + 1;   % counter prescribed degrees of freedom
for node = 1 : nnode
    for dof = 1 : ndf
        if (ID(dof,node) == 0) % degree of freedom is prescribed
            ID(dof,node) = dof_pres;
            % update counter
            dof_pres     = dof_pres + 1;
        else                   % unknown degree of freedom
            ID(dof,node) = dof_free;
            % update counter
            dof_free = dof_free + 1;
        end
    end
end

%% LM-array

LM = zeros(ndf*npe,nelem);
for el = 1 : nelem
    % reset counter
    dof = 0;
    for node = 1 : npe
        for j = 1 : ndf
            dof = dof + 1;
            LM(dof,el) = ID(j,inz(el,node));
        end
    end
end

end