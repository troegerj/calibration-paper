function [statistics,res_optimized] = run_statistics(expData,sim_inpFile,opt_para,const_mod)
% function to evaluate different statistical quantities after performing
% the non-linear least-squares calibration
%
% INPUT
%  expData -- experimental data
%  sim_inpFile -- inputfile name for finite element code
%  opt_para -- parameter structure for calibration
%  const_mod -- structure containing different quantities such as switches or settings
%
% OUTPUT
%  statistics -- structure with statistical quantities
%  res_optimized -- model response when using optimized parameters

% compute unweighted residuals in the found solution
if strcmp(sim_inpFile,'plate_linElas')
    % compute model response with optimized parameters and finite elements
    res_optimized = mainFE(sim_inpFile,[opt_para(:).value]');
    %
    modelData = [res_optimized.dispCalib{1}(1,:)'; res_optimized.dispCalib{1}(2,:)']; % only one timestep is considered
    statistics.residual = modelData - expData;
elseif strcmp(sim_inpFile,'cube')
    if const_mod.matpar == 1 % calibrate Young's modulus using stress-strain data
        axStrain = linspace(const_mod.maxStrainElas/const_mod.ndata,...
            const_mod.maxStrainElas,const_mod.ndata)'; % axial strain array
        % evaluate stress-strain relation for linear elasticity
        modelData = [opt_para(1).value]*axStrain; % sigma = E*epsilon
        res_optimized = modelData;
    elseif const_mod.matpar == 2 % calibrate Poisson's ratio using lateral strain data
        axStrain = linspace(const_mod.maxStrainElas/const_mod.ndata,...
            const_mod.maxStrainElas,const_mod.ndata)'; % axial strain array
        % evaluate lateral strain relation for linear elasticity
        modelData = -[opt_para(2).value]*axStrain; % epsilonQ = -nu*epsilon
        res_optimized = modelData;
    elseif const_mod.matpar == 3 % calibration of plasticity -> FEM evaluation
        % compute model response with optimized parameters and finite elements
        idx = [1; 2; 3; 6; 7]; % material parameters for plasticity
        res_optimized = mainFE(sim_inpFile,[opt_para(idx).value]');
        modelData = reshape(res_optimized.stress(1,1,1,:),[],1);
    end
    statistics.residual = modelData - expData;
end

% size of residuals
nres = size(statistics.residual,1);

% build weight matrix (stored as vector) and compute value of objective
% function with weights
if const_mod.weightRes
    if strcmp(sim_inpFile,'plate_linElas')
        weightFactors = [const_mod.weightDisp_x1*ones(nres/2,1); const_mod.weightDisp_x2*ones(nres/2,1)];
        statistics.objFun_weighted = 0.5*(diag(weightFactors)*statistics.residual)'*(diag(weightFactors)*statistics.residual);
    elseif strcmp(sim_inpFile,'cube')
        % const_mod.weights contains the covariance of experimental
        % observations Sigma -- for calibration with lsqnonlin, we need to
        % compute sqrt(Sigma^-1) as weight matrix
        Sigma = diag(const_mod.weights);
        W = sqrt(inv(Sigma));
        weightFactors = diag(W);
        %
        statistics.objFun_weighted = 0.5*(W*statistics.residual)'*(W*statistics.residual);
    end
end

% compute value of objective function in solution without weights
statistics.objFun = 0.5*statistics.residual'*statistics.residual;

% determine number of optimized parameters
nopt = sum([opt_para(:).opt]);

% save optimized parameters 
optvec_flags(:) = [opt_para(:).opt];
optvec_optimized(:) = [opt_para(optvec_flags).value];
statistics.optvec = optvec_optimized;

%%
% compute empiric mean values -- only reasonable when same physical quantities are present
statistics.mean_res = mean(statistics.residual); % unweighted residual
statistics.mean_res_weighted = mean(diag(weightFactors)*statistics.residual); % weighted residual
statistics.mean_exp = mean(expData); 

%%
% compute variance of residuals (squared standard deviation)
statistics.sigSquare_res = (statistics.residual'*statistics.residual)/(length(statistics.residual)-1);

%%
% compute Jacobian at optimized parameters using numerical differentiation
%
% temporary copy of parameters structure
opt_para_tmp = opt_para;
% Jacobian without normalization
statistics.jacobian = zeros(length(statistics.residual),nopt);
statistics.jacobian(:,:) = numerical_differentiation(optvec_optimized,@compute_residuum,const_mod.finite_diff_method);

% compute jacobian with weights
statistics.jacobian_weighted(:,:) = diag(weightFactors)*statistics.jacobian;

%%
% compute approximation of Hessian
statistics.H_approx(:,:) = statistics.jacobian'*statistics.jacobian;
statistics.H_approx_weighted(:,:) = (diag(weightFactors)*statistics.jacobian)'*(diag(weightFactors)*statistics.jacobian);
% compute determinant of approx. Hessian
statistics.detH_approx = det(statistics.H_approx);
statistics.detH_approx_weighted = det(statistics.H_approx_weighted);

%%
% compute identifiability quantities - ratio of eigenvalues and
% dimensionless xi from Beck and Arnold (1977)

% compute ratio between smallest and largest eigenvalue
statistics.ratioEig_H_approx = min(eig(statistics.H_approx))/max(eig(statistics.H_approx));
statistics.ratioEig_H_weighted = min(eig(statistics.H_approx_weighted))/max(eig(statistics.H_approx_weighted));

% compute criterion from Beck and Arnold
statistics.xi_H_approx = statistics.detH_approx/(nopt^(-1)*trace(statistics.H_approx))^nopt;
statistics.xi_H_approx_weighted = statistics.detH_approx_weighted/(nopt^(-1)*trace(statistics.H_approx))^nopt;

%%
% set up covariance matrix (with approximated Hessian)
statistics.P_approx = statistics.sigSquare_res*inv(statistics.H_approx);
statistics.P_approx_weighted = statistics.sigSquare_res*inv(statistics.H_approx_weighted);

%%
% set up correlation matrix (based on approximated Hessian)
statistics.correlation_approx = zeros(nopt,nopt);
%
for j = 1:size(statistics.P_approx,2)
    for i = 1:size(statistics.P_approx,1)
        statistics.correlation_approx(i,j) = statistics.P_approx(i,j)/(sqrt(statistics.P_approx(i,i)*statistics.P_approx(j,j)));
        statistics.correlation_approx_weighted(i,j) = statistics.P_approx_weighted(i,j)/(sqrt(statistics.P_approx_weighted(i,i)*statistics.P_approx_weighted(j,j)));
    end
end

%%
% compute confidence interval (based on approximated Hessian)
statistics.Delta_kappa_approx = sqrt(diag(statistics.P_approx));
statistics.Delta_kappa_approx_weighted = sqrt(diag(statistics.P_approx_weighted));

%%
% coefficient of determination (R^2-value)
statistics.R_Square = 1 - (statistics.residual'*statistics.residual)/((expData-statistics.mean_exp)'*(expData-statistics.mean_exp));


    function res = compute_residuum(optvec)
        % here, no normalization should be done! 
        optvec_normalization_tmp    = zeros(nopt,1);
        optvec_normalization_tmp(:) = 1;
        %
        % update parameters + perform simulation + compute residuals
        res = compute_residuum_main(opt_para_tmp,optvec,...
            optvec_normalization_tmp,optvec_flags,expData,sim_inpFile,const_mod);
    end

end


%% subfunction

function res = compute_residuum_main(opt_para,optvec_norm,optvec_normalization,...
    optvec_flags,expData,sim_inpFile,const_mod)

% update structure of optimized parameters
opt_para = update_opt_para(opt_para,optvec_norm,optvec_normalization,optvec_flags);

% perform simulation to obtain model response using finite element code or
% evaluate analytical function (for calibration of elasticity parameters 
% using stress-strain or lateral strain information)
if any(contains(fieldnames(const_mod),'matpar')) % calibrate elasto-plasticity
    if const_mod.matpar == 1 % calibrate Young's modulus using stress-strain data
        axStrain = linspace(const_mod.maxStrainElas/const_mod.ndata,...
            const_mod.maxStrainElas,const_mod.ndata)'; % axial strain array
        % evaluate stress-strain relation for linear elasticity
        modelResponse = [opt_para(1).value]*axStrain; % sigma = E*epsilon
    elseif const_mod.matpar == 2 % calibrate Poisson's ratio using lateral strain data
        axStrain = linspace(const_mod.maxStrainElas/const_mod.ndata,...
            const_mod.maxStrainElas,const_mod.ndata)'; % axial strain array
        % evaluate lateral strain relation for linear elasticity
        modelResponse = -[opt_para(2).value]*axStrain; % epsilonQ = -nu*epsilon
    elseif const_mod.matpar == 3 % calibration of plasticity -> FEM evaluation
        idx = [1; 2; 3; 6; 7]; % material parameters for plasticity
        res_model = mainFE(sim_inpFile,[opt_para(idx).value]);
    end
else % perform finite element simulation 
    res_model = mainFE(sim_inpFile,[opt_para(:).value]);
    % perform simulation using parametric PINN
    %%% adapt this line for model evaluation using a pre-trained parametric PINN
end

% compute residuals -- without weights (considered above in calling function, if necessary)
if strcmp(sim_inpFile,'plate_linElas') % plate with hole (linear elasticity)
    % get model response [displacements in x_1; displacements in x_2]
    nodalDisp = [res_model.dispCalib{1}(1,:)'; res_model.dispCalib{1}(2,:)']; % only one timestep is considered
    % compute unweighted residuals
    res = nodalDisp-expData;
elseif strcmp(sim_inpFile,'cube') % cube (elasto-plasticity)
    % build model response in case of plasticity
    if const_mod.matpar == 3
        modelResponse = reshape(res_model.stress(1,1,1,:),[],1);
    end
    res = modelResponse - expData;
else
    error('%s: computation of residuals not implemented for input file %s',mfilename,sim_inpFile);
end

end