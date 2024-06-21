function [expData,weights] = eval_exp_EUCLID(const_mod,nodeCoord)
% DESCRIPTION
%  function to perform the linear interpolation of the synthetical
%  experimental data onto the finite element node positions. It is assumed
%  that the data is present in ./EUCLID_data/ -- adapt path, if necessary
%
% INPUT
%  const_mod -- structure with information about choosen model and noise level
%  nodeCoord -- finite element node positions
%
% OUTPUT
%  expData -- (experimental) displacement data interpolated to finite element nodes
%             and reaction force information
%  weights -- weighting factors for displacements and forces (always max.
%             value in experimental data over time)
%             weights(1) -- max. horz. displacement [mm]
%             weights(2) -- max. vert. displacement [mm]
%             weights(3) -- max. horz. reaction force [N]
%             weights(4) -- max. horz. reaction force [N]

% main path
EUCLIDpath = './EUCLID_Data';

% build experimental data path depending on noise level and constitutive
% model
if const_mod.noiseLevel == 0 % no noise
    noiseStr = 'plate_hole_60k';
elseif const_mod.noiseLevel == 1 % low noise
    noiseStr = 'denoised_60k_to_60k_noise=0.0001';
elseif const_mod.noiseLevel == 2 % high noise
    noiseStr = 'denoised_60k_to_60k_noise=0.001';
end
%
if strcmp(const_mod.constitutiveModel,'NH') % Neo-Hooke
    matStr = 'plate_hole_60k_NeoHookeanJ2';
    ntime = 4; % number of time points
elseif strcmp(const_mod.constitutiveModel,'IH') % Isihara
    matStr = 'plate_hole_60k_Isihara';
    ntime = 8; % number of time points
elseif strcmp(const_mod.constitutiveModel,'HW') % Haines-Wilson
    matStr = 'plate_hole_60k_HainesWilson';
    ntime = 8; % number of time points
end
% 
exp_path = strcat(EUCLIDpath,'/',noiseStr,'/',matStr,'/');

% get experimental information for each considered time point
for ii = 1 : ntime
    % build paths
    pathDIC = strcat(exp_path,num2str(ii),'0/output_nodes.csv');
    pathForce = strcat(exp_path,num2str(ii),'0/output_reactions.csv');

    % read synthetical displacement data and point coordinates in reference configuration
    tmp = readmatrix(pathDIC);
    if ii == 1
        coord = tmp(:,2:3); % x y [mm]
    end
    expDisp = tmp(:,4:5); % ux uy [mm]
    clear tmp

    % perform interpolation of synthetic data to finite element node positions
    %
    % build interpolants for ux (horizontal disp.) and uy (vertical disp.)
    F_ux = scatteredInterpolant(coord(:,1),coord(:,2),expDisp(:,1));
    F_uy = scatteredInterpolant(coord(:,1),coord(:,2),expDisp(:,2));
    % evaluate interpolants at finite element node coordinates
    exp_mod.disp(1,:,ii) = F_ux(nodeCoord(:,1),nodeCoord(:,2));
    exp_mod.disp(2,:,ii) = F_uy(nodeCoord(:,1),nodeCoord(:,2));

    % get reaction forces 
    tmp = readmatrix(pathForce);
    exp_mod.force(ii,1) = tmp(2); % horizontal force in N
    exp_mod.force(ii,2) = tmp(4); % vertical force in N
    clear tmp
end

% determine weights
weights(1) = max(max(exp_mod.disp(1,:,:))); % max. horizontal disp. [mm]
weights(2) = max(max(exp_mod.disp(2,:,:))); % max. vertical disp. [mm]
weights(3) = max(exp_mod.force(:,1)); % max. horizontal reaction force [N]
weights(4) = max(exp_mod.force(:,2)); % max. vertical reaction force [N]

% build experimental data array consisting of 
% [horizontal displacements; vertical displacments; horizontal reaction forces;
% vertical reaction forces]
expData = [reshape(exp_mod.disp(1,:,:),[],1); reshape(exp_mod.disp(2,:,:),[],1); ...
    exp_mod.force(:,1); exp_mod.force(:,2)];

end