function [opt_para,t_elapsed,resnorm,residual,opt_exitflag,opt_output,opt_lambda,...
    opt_jacobian] = perform_optimization(expData,sim_inpFile,opt_para,opt_options,const_mod)
% function to perform the calibration using the non-linear least-squares
% method with the lsqnon-lin function
%
% INPUT
%  expData -- experimental data
%  sim_inpFile -- inputfile name for finite element code
%  opt_para -- parameter structure for calibration
%  opt_options -- option-structure for lsqnonlin-optimizer
%  const_mod -- structure containing different quantities such as switches or settings
%
% OUTPUT
%  opt_para -- parameter structure after optimization
%  t_elapsed -- time required for optimization
%  resnorm -- L2-norm of the residuals in the solution
%  residual -- residuals vector
%  opt_exitflag -- indicator for termination of optimizer (should be 3)
%  opt_output -- various quantities about the optimization procedure
%  opt_lambda -- informations about the lower and upper bounds
%  opt_jacobian -- normalized Jacobian in the solution

% get data from opt_para and normalize with initial values
optvec_flags(:)         = [opt_para(:).opt];
optvec_normalization(:) = [opt_para(optvec_flags).initial_value];
optvec_initial(:)       = [opt_para(optvec_flags).initial_value]./optvec_normalization;
lb(:)                   = [opt_para(optvec_flags).lb];
ub(:)                   = [opt_para(optvec_flags).ub];

%% solve non-linear least-squares

% start timer
t_opt = tic;

% do optimization
[optvec_fit,resnorm,residual,opt_exitflag,opt_output,opt_lambda,opt_jacobian] =...
    lsqnonlin(@compute_residuum,optvec_initial,lb,ub,opt_options);

% resnorm is the squared L2-norm of the residual, compute L2-norm of the residual
resnorm = sqrt(resnorm);

% save fitted parameters
opt_para = update_opt_para(opt_para,optvec_fit,optvec_normalization,optvec_flags);

% stop timer
t_elapsed = toc(t_opt);


    function [res,J] = compute_residuum(optvec_norm)
        
        % compute residuals (vector required within lsqnonlin-function)
        res = compute_residuum_main(opt_para,optvec_norm,optvec_normalization,...
            optvec_flags,expData,sim_inpFile,const_mod);
        
        % compute jacobian, if requested
        if nargout > 1   % Two output arguments
            J = numerical_differentiation(optvec_norm,@compute_residuum_main,const_mod.finite_diff_method);
        else
            J = [];
        end
        
    end

end

%% subfunction to compute residuum between model response and experimental data

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

% compute residuals
if strcmp(sim_inpFile,'plate_linElas') % plate with hole (linear elasticity)
    % get model response [displacements in x_1; displacements in x_2]
    nodalDisp = [res_model.dispCalib{1}(1,:)'; res_model.dispCalib{1}(2,:)']; % only one timestep is considered
    if const_mod.weightRes
        % number of experimental data
        numd = size(expData,1);
        % build weight matrix (stored as vector)
        weightFactors = [const_mod.weightDisp_x1*ones(numd/2,1); const_mod.weightDisp_x2*ones(numd/2,1)];
        % weight residuals
        res = diag(weightFactors)*(nodalDisp - expData);
    else
        % unweighted residuals
        res = nodalDisp - expData;
    end
elseif strcmp(sim_inpFile,'cube') % cube (elasto-plasticity)
    % build model response in case of plasticity
    if const_mod.matpar == 3
        modelResponse = reshape(res_model.stress(1,1,1,:),[],1);
    end
    % compute residuals
    if const_mod.weightRes
        % const_mod.weights contains the covariance of experimental
        % observations Sigma -- for calibration with lsqnonlin, we need to
        % compute sqrt(Sigma^-1) as weight matrix
        Sigma = diag(const_mod.weights);
        W = sqrt(inv(Sigma));
        %
        res = W*(modelResponse - expData);
    else
        res = modelResponse - expData;
    end
elseif strcmp(sim_inpFile,'plate_hyperelas_NH') || ...
       strcmp(sim_inpFile,'plate_hyperelas_IH') || ...
       strcmp(sim_inpFile,'plate_hyperelas_HW')
    % build model response similar to experimental data, see eval_exp_EUCLID.m
    ntime = size(res_model.dispCalib,2);
    for i = 1 : ntime
        nodalDisp(1,:,i) = res_model.dispCalib{i}(1,:); % horizontal displacements
        nodalDisp(2,:,i) = res_model.dispCalib{i}(2,:); % vertical displacements
    end
    modelResponse = [reshape(nodalDisp(1,:,:),[],1); reshape(nodalDisp(2,:,:),[],1); ...
        res_model.reactionCalib(:,1); res_model.reactionCalib(:,2)];
    % consider weighting coefficients
    if const_mod.weightRes
        % number of experimental displacement data
        numd = size(reshape(nodalDisp(1,:,:),[],1),1); 
        % build weight matrix (stored as vector)
        weightFactors = [const_mod.weights(1)*ones(numd,1); ...
            const_mod.weights(2)*ones(numd,1); ...
            const_mod.weights(3)*ones(ntime,1); ...
            const_mod.weights(4)*ones(ntime,1)];
        % weight residuals
        res = diag(weightFactors)*(modelResponse - expData);
    else
        res = modelResponse - expData;
    end    
else
    error('%s: computation of residuals not implemented for input file %s',mfilename,sim_inpFile);
end

end