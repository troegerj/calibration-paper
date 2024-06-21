function [q,stress,strain,yieldval,stiff,rhs] = seteqs(model,un,qn,coord,inz,rhs_nodalForce)
% DESCRIPTION
%  function to build the global system of linear equations
%
% INPUT
%  model -- structure containing information about the specified model
%  un -- displacement array (ndof x 1)
%  qn -- internal variable state at tn (nq*nint x 1)
%  coord -- node coordinates
%  inz -- incidence matrix
%  rhs_nodalForce -- applied nodal forces
%
% OUTPUT
%  q -- updated internal variable state at tn+1
%  stress -- stresses at integration points
%  strain -- strains at integration points
%  yieldval -- yielding indicators at integration points
%  stiff -- assembled stiffness matrix for global Newton method
%  rhs -- assembled right-hand side for global Newton method

% extract quantities from model-structure
ndim   = model.ndim;
ndof   = model.ndof;
nelem  = model.eset(1).nelem;
elform = model.eset(1).elform;
npe    = model.eset(1).npe;
wght   = model.eset(1).weights;
shp    = model.eset(1).shp;
nintp  = model.eset(1).nintp;
LM     = model.LM;
matfun = model.matfun;
matpar = model.matpar;
dt     = model.dt;
nq     = model.nq;

% initialize stresses, strains, and internal variables
if ndim == 2
    switch model.ptype
        case 'plane_strain'
            stress = zeros(4,nintp,nelem);
            strain = zeros(3,nintp,nelem);
        case 'plane_stress'
            stress = zeros(3,nintp,nelem);
            strain = zeros(4,nintp,nelem);
        otherwise
            error('%s: two-dimensional problemtype not implemented',mfilename);
    end
elseif ndim == 3
    stress = zeros(6,nintp,nelem);
    strain = zeros(6,nintp,nelem);
end
yieldval = false(1,nintp,nelem);
q = zeros(nq,nintp,nelem);

% initialize/reset global stiffness matrix (reset to remove entries from
% previous iteration)
stiff = sparse(ndof,ndof);
% initialize/reset right-hand side (reset to remove entries from previous
% iteration)
rhs   = sparse(ndof,1);

% consider prescribed nodal forces, if present
if ~isempty(rhs_nodalForce)
    rhs(1:ndof,1) = rhs_nodalForce;
end

% set function handle for element routine
switch elform

    case 'quad' % quadrilateral element
        if model.nlgeom % nonlinear geometry -- finite strains
            elem = @elem2D_largeDef;
        else % small strain case
            elem = @elem2D;
        end

    case 'hex' % hexahedral element
        if model.nlgeom
            error('%s: nonlinear geometry switch not allowed for 3D-elements')
        else
            elem = @elem3D;
        end

    otherwise
        error('%s: element type %s not implemented',mfilename,eltype)

end

%% computation of element contributions

for el = 1 : nelem

    % node coordinates of element el
    xel = coord(inz(el,:),:);

    % store displacements of current element
    ue = un(LM(:,el));

    % computation of element contributions to stiffness matrix and force
    % vector
    [kelem,felem,stress(:,:,el),strain(:,:,el),yieldval(:,:,el),q(:,:,el)] = ...
        elem(el,ndim,xel,shp,wght,npe,dt,ue,matfun,matpar,qn(:,:,el),model.ptype);

    % assembly of element contributions into global matrices
    stiff(LM(:,el),LM(:,el)) = stiff(LM(:,el),LM(:,el)) + kelem;
    rhs(LM(:,el)) = rhs(LM(:,el)) + felem;

end

end