clc
clear
close all
rng(0)
addpath(genpath('../../..'));

%% Strings
str_noise = 'withoutNoise';
% str_noise = 'withNoise2e-04';
% str_noise = 'withNoise4e-04';

str_data = ['displacements_' str_noise '.csv'];
str_mesh = ['mesh_plate.mat'];
str_processed = ['displacements_processed_' str_noise '.mat'];

%% Full-Field Data
data_given = readmatrix(str_data);
x1_given = data_given(:,1);
x2_given = data_given(:,2);
ux_given = data_given(:,3);
uy_given = data_given(:,4);
u_global_FOV = zeros(2*length(ux_given),1);
u_global_FOV(1:2:(end-1)) = ux_given;
u_global_FOV(2:2:end) = uy_given;
mesh_given = load(str_mesh,'mesh');
mesh_given = mesh_given.('mesh');
mesh.node = mesh_given.node_coord;
mesh.element = mesh_given.elements;
mesh_FOV = mesh_info(mesh);
mesh_FOV = bc_plate_w_hole_tension(mesh_FOV);

%% Force Data
reaction = -1500;

%% True Material Parameters
E_true = 210000;
nu_true = 0.3;
CPlaneStrain_true = E_true/((1+nu_true)*(1-2*nu_true)) * [ ...
    1-nu_true, nu_true, 0;...
    nu_true, 1-nu_true, 0;...
    0, 0, 1/2-nu_true;...
    ];
CPlaneStress_true = E_true/(1-nu_true^2) * [ ...
    1, nu_true, 0;...
    nu_true, 1, 0;...
    0, 0, 1/2*(1-nu_true);...
    ];
theta_true = CPlaneStress_true(1:2,1);

%% Save
para_true.E = E_true;
para_true.nu = nu_true;
para_true.theta = theta_true;
para_true.CPlaneStrain = CPlaneStrain_true;
para_true.CPlaneStress = CPlaneStress_true;
mesh_total.node = mesh_given.node_coord;
mesh_total.element = mesh_given.elements;
mesh = mesh_FOV;
u_global = u_global_FOV;

save(str_processed,'mesh_total','mesh','u_global','reaction','para_true')

%% Plot
plot_mesh(mesh_total.node,mesh_total.element)
plot_scalar_field_2D(mesh_FOV.node,mesh_FOV.element,ux_given,'newfigure',true,'mesh_overlay',false,'clabel','$u_1$')
xlim([min(mesh_total.node(:,1)),max(mesh_total.node(:,1))])
ylim([min(mesh_total.node(:,2)),max(mesh_total.node(:,2))])

