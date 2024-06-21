function [model,coord,inz,bcond,load,lfun] = hyperelas_plate_with_hole_HW(matpar)
% DESCRIPTION
%  input file for hyperelastic plate with a hole using Haines-Wilson model
%  containing required informations of the spatial discretization and 
%  applied loads
%
% INPUT
%  matpar -- material parameters
%
% OUTPUT
%  model -- structure containing information about the specified model
%  coord -- node coordinates
%  inz -- incidence matrix
%  bcond -- array with boundary condition information
%  load -- applied nodal loads
%  lfun -- load functions

%% model data

ndim   = 2; % two-dimensional problem
npe    = 8; % nodes per element: quadrilateral elements with quadratic ansatz
ndf    = 2; % number of degrees of freedom per node
ptype  = 'plane_strain'; % problemtype 2D
elform = 'quad'; % element type: quadrilateral

% path to spatial discretization data
path = '../FEM/inputfiles/modelData_hyperelas_plate-with-hole/';

% node coordinates
coord = readmatrix(strcat(path,'hyperelas_plate_nodeCoordinates.csv'));
nnode = size(coord,1);

% number of degrees of freedom
ndof = nnode*ndf;

% incidence matrix
inz = readmatrix(strcat(path,'hyperelas_plate_elementIncidences.csv'));
nelem = size(inz,1);

%% boundary conditions

% boundary condition information stored in array
%
% rows: information of i-th boundary condition
% columns: node number,
%          load function (displacement/force x_1-direction),
%          load function (displacement/force x_2-direction),
%          load factor (x_1-direction),
%          load factor (x_2-direction)

% initialize
bcond = zeros(nnode,2*ndf+1);
% create index array
idx = (1:nnode)';

% edge length
L = 1; % [mm]

% exclude nodes at corners
exclude_nodes(1) = idx(all(abs(coord-repmat([0,1],size(coord,1),1))<eps,2)); % x_1 = 0, x_2 = 1
exclude_nodes(2) = idx(all(abs(coord-repmat([1,0],size(coord,1),1))<eps,2)); % x_1 = 1, x_2 = 0
exclude_nodes(3) = idx(all(abs(coord-repmat([1,1],size(coord,1),1))<eps,2)); % x_1 = 1, x_2 = 1

% symmetry boundary condition at left edge (x_1 == 0) 
global_nodeID      = idx(abs(coord(:,1))<=10*eps);
global_nodeID      = global_nodeID(~ismember(global_nodeID,exclude_nodes));
idx1               = 1;
idx2               = idx1-1+length(global_nodeID);
bcond(idx1:idx2,1) = global_nodeID;
bcond(idx1:idx2,2) = 1; % degrees of freedom are fixed in x_1-direction
bcond(idx1:idx2,4) = 0; % degrees of freedom are fixed in x_1-direction

% symmetry boundary condition at bottom edge (x_2 == 0)
global_nodeID      = idx(abs(coord(:,2))<=10*eps);
global_nodeID      = global_nodeID(~ismember(global_nodeID,exclude_nodes));
idx1               = idx2+1;
idx2               = idx1-1+length(global_nodeID);
bcond(idx1:idx2,1) = global_nodeID;
bcond(idx1:idx2,3) = 1; % degrees of freedom are fixed in x_2-direction
bcond(idx1:idx2,5) = 0; % degrees of freedom are fixed in x_2-direction

% prescribed displacement boundary condition at right edge (x_1 == 1)
global_nodeID      = idx(abs(coord(:,1)-L)<=10*eps);
global_nodeID      = global_nodeID(~ismember(global_nodeID,exclude_nodes));
idx1               = idx2+1;
idx2               = idx1-1+length(global_nodeID);
bcond(idx1:idx2,1) = global_nodeID;
bcond(idx1:idx2,2) = 2; % degrees of freedom are prescribed in x_1-direction
bcond(idx1:idx2,4) = 1; % degrees of freedom are prescribed in x_1-direction

% prescribed displacement boundary condition at top edge (x_2 == 1)
global_nodeID      = idx(abs(coord(:,2)-L)<=10*eps);
global_nodeID      = global_nodeID(~ismember(global_nodeID,exclude_nodes));
idx1               = idx2+1;
idx2               = idx1-1+length(global_nodeID);
bcond(idx1:idx2,1) = global_nodeID;
bcond(idx1:idx2,3) = 3; % degrees of freedom are prescribed in x_2-direction
bcond(idx1:idx2,5) = 1; % degrees of freedom are prescribed in x_2-direction

% set boundary condition for corner nodes
global_nodeID      = exclude_nodes(1); % x_1 = 0, x_2 = 1
idx1               = idx2+1;
idx2               = idx1-1+length(global_nodeID);
bcond(idx1:idx2,1) = global_nodeID;
bcond(idx1:idx2,2) = 1; % degrees of freedom are fixed in x_1-direction
bcond(idx1:idx2,3) = 3; % degrees of freedom are prescribed in x_2-direction
bcond(idx1:idx2,4) = 0; % degrees of freedom are fixed in x_1-direction
bcond(idx1:idx2,5) = 1; % degrees of freedom are prescribed in x_2-direction
%
global_nodeID      = exclude_nodes(2); % x_1 = 1, x_2 = 0
idx1               = idx2+1;
idx2               = idx1-1+length(global_nodeID);
bcond(idx1:idx2,1) = global_nodeID;
bcond(idx1:idx2,2) = 2; % degrees of freedom are prescribed in x_1-direction
bcond(idx1:idx2,3) = 1; % degrees of freedom are fixed in x_2-direction
bcond(idx1:idx2,4) = 1; % degrees of freedom are prescribed in x_1-direction
bcond(idx1:idx2,5) = 0; % degrees of freedom are fixed in x_2-direction
%
global_nodeID      = exclude_nodes(3); % x_1 = 1, x_2 = 1
idx1               = idx2+1;
idx2               = idx1-1+length(global_nodeID);
bcond(idx1:idx2,1) = global_nodeID;
bcond(idx1:idx2,2) = 2; % degrees of freedom are prescribed in x_1-direction
bcond(idx1:idx2,3) = 3; % degrees of freedom are prescribed in x_2-direction
bcond(idx1:idx2,4) = 1; % degrees of freedom are prescribed in x_1-direction
bcond(idx1:idx2,5) = 1; % degrees of freedom are prescribed in x_2-direction

% reduction
bcond(idx2+1:end,:) = [];

% determine number of nodes with Dirichlet boundary condition
nbc = size(bcond,1);

% initialize -- no nodal loads are applied
load = [];

% determine number of nodes with Neumann boundary condition
nloadbc = size(load,1);

% cell-array for load function
lfun    = cell(3,1);
lfun{1} = @(t)(0); % degrees of freedom are fixed
lfun{2} = @(t)(0.4*t); % prescribed displacement in x_1-direction [mm]
lfun{3} = @(t)(0.8*t); % prescribed displacement in x_2-direction [mm]

%% element sets

% number of element sets
neset = 1;
% starting element number for set 1
eset(1).start_el = 1;
% number of elements of set 1
eset(1).nelem = nelem;
% number of nodes per element in set 1
eset(1).npe = npe;
% incidence matrix of element set 1
eset(1).incidence = inz;
% element form in element set 1
eset(1).elform = elform;

%% constitutive model for material behavior

% Haines-Wilson hyperelasticity with material parameters
%  matpar(1) - bulk modulus [N/mm^2]
%  matpar(2) - c_10 [N/mm^2]
%  matpar(3) - c_01 [N/mm^2]
%  matpar(4) - c_11 [N/mm^2]
%  matpar(5) - c_30 [N/mm^2]
model.matpar = matpar;
model.matfun = @stct_HainesWilson;

% hyperelasticity has no internal variables
model.nq = 0;

% nonlinear geometry switch
model.nlgeom = true;

%% settings for solution of systems of linear equations

% time-integration settings (here, only step-wise application of loads)
t0 = 0; % [s]
tend = 1.0; % [s]
dt = 0.0125; % [s] - time-step size
maxsteps = 100; % maximum number of time-steps

% termination criteria for Newton method 
tolg = 1.e-9; % tolerance in discretized equilibrium equations
toldu = 1.e-9; % tolerance for displacement update
maxni = 25; % maximum number of Newton iterations

%% settings for output

% output quantities
out.disp          = true;
out.stress        = false;
out.strain        = false;
out.yieldval      = false;
out.innerVar      = false;
out.reactionForce = true;

% output time
out.time = linspace(0.125,tend,tend/0.125)'; % every 0.125 s 

% visualization of nodal quantities
out.visualizeDisp = false;

%% store data in model-structure

model.ndim    = ndim;
model.ptype   = ptype;
model.nnode   = nnode;
model.nbc     = nbc;
model.nloadbc = nloadbc;
model.ndof    = ndof;
model.ndf     = ndf;
model.neset   = neset;
model.eset    = eset;
%
model.t0       = t0;
model.tend     = tend;
model.dt       = dt;
model.maxsteps = maxsteps;
model.tolg     = tolg;
model.toldu    = toldu;
model.maxni    = maxni;
%
model.out = out;

end