% clc
clear
close all
format longg
rng(0)
addpath(genpath(pwd));

plot_results = true;
save_results = true;

%% Data
for i=[6,5,4,3,2,1]
    tic
    switch i
        case 1
            dataname = 'displacements_processed_withoutNoise';
            norm_type = 'I'; % Euclidean norm
            lambda_r = 1e4; % influence of the reaction force
            lambda_d = 1e-5; % influence of the data
        case 2
            dataname = 'displacements_processed_withNoise2e-04';
            norm_type = 'I'; % Euclidean norm
            lambda_r = 1e4; % influence of the reaction force
            lambda_d = 1e-5; % influence of the data
        case 3
            dataname = 'displacements_processed_withNoise4e-04';
            norm_type = 'I'; % Euclidean norm
            lambda_r = 1e4; % influence of the reaction force
            lambda_d = 1e-5; % influence of the data
        case 4
            dataname = 'displacements_processed_withoutNoise';
            norm_type = 'K'; % K-norm
            lambda_r = 1e4; % influence of the reaction force
            lambda_d = 1e-10; % influence of the data
        case 5
            dataname = 'displacements_processed_withNoise2e-04';
            norm_type = 'K'; % K-norm
            lambda_r = 1e4; % influence of the reaction force
            lambda_d = 1e-10; % influence of the data
        case 6
            dataname = 'displacements_processed_withNoise4e-04';
            norm_type = 'K'; % K-norm
            lambda_r = 1e4; % influence of the reaction force
            lambda_d = 1e-10; % influence of the data
    end
    load(['data/' dataname '.mat'])

    E_true = para_true.E;
    nu_true = para_true.nu;
    theta_true = para_true.theta;
    E_true_test = theta_true(1)*(1-(theta_true(2)/theta_true(1))^2);
    nu_true_test = theta_true(2)/theta_true(1);

    %% Hyperparameters
    
    % note: for lambda_d --> inf, the results of the all-at-once approach
    % converges to the result of the virtual field method
    % note: for lambda_d --> 0, the nonlinear solver takes a lot of time

    %% FEM
    [K_Dirichlet, b_Dirichlet] = linEla_assemble_K_b_Dirichlet(theta_true,u_global,mesh,lambda_r);
    b_Neumann = assemble_b(reaction,mesh,lambda_r);
    b = b_Dirichlet + b_Neumann;
%     u_fem_unknown = linsolve(K_Dirichlet,b); % only possible if mesh.dof_known has sufficiently many entries
% 
%     u_fem = zeros(size(u_global));
%     u_fem(mesh.dof_known) = u_global(mesh.dof_known);
%     u_fem(mesh.dof_unknown) = u_fem_unknown;
% 
%     disp('========== FEM ==========')
%     disp('Maximum error:')
%     disp(num2str(maxabs(u_global-u_fem)));
%     disp(' ')

    %% VFM
    disp('========== VFM ==========')
    A = linEla_assemble_A(u_global,mesh,lambda_r);
    theta_vfm = linsolve(A,b_Neumann);
    E_vfm = theta_vfm(1)*(1-(theta_vfm(2)/theta_vfm(1))^2);
    nu_vfm = theta_vfm(2)/theta_vfm(1);

    disp('Maximum error:')
    disp(num2str(maxabs(A*theta_vfm-b_Neumann)));
    disp(' ')
    % note: if norm(A*theta_vfm-b_Neumann) is very small, it does not make
    % sense to apply the all-at-once approach, because assuming the VFM
    % solution for theta and the measured displacement for u will be a minimum
    % of the AAO cost function

    % disp('True theta:')
    % disp(theta_true);
    % disp('Calibrated theta (VFM):')
    % disp(theta_vfm);
    % disp('True E:')
    % disp(E_true);
    % disp('Calibrated E (VFM):')
    % disp(E_vfm);
    % disp('True nu:')
    % disp(nu_true);
    % disp('Calibrated nu (VFM):')
    % disp(nu_vfm);
    % disp(' ')

    %% AAO
    disp('========== AAO ==========')
    theta_init = [225000;65000];
    E_init = theta_init(1)*(1-(theta_init(2)/theta_init(1))^2);
    nu_init = theta_init(2)/theta_init(1);
    u_init = u_global(mesh.dof_unknown);
    utheta_init = [theta_init;u_init];
    cost_vec = @(utheta) linEla_cost_vec_aao(utheta(1:2),utheta(3:end),reaction,u_global,mesh,lambda_r,lambda_d,'norm_type',norm_type);
    cost_dcost_vec_ANA = @(utheta) linEla_cost_vec_aao(utheta(1:2),utheta(3:end),reaction,u_global,mesh,lambda_r,lambda_d,'norm_type',norm_type,'derivative_type','ANA');
    cost_dcost_vec_FD = @(utheta) linEla_cost_vec_aao(utheta(1:2),utheta(3:end),reaction,u_global,mesh,lambda_r,lambda_d,'norm_type',norm_type,'derivative_type','FD');

    [utheta_aao, options] = solve_aao(cost_vec,utheta_init,'method','lsqnonlin','dcost',cost_dcost_vec_ANA,'theta_lb',[0; 0]);
    %utheta_aao = solve_aao(cost_vec,utheta_init,'method','lsqnonlin','theta_lb',[0; 0]);

    %% Check Derivatives
    % theta_check = 0.5*theta_vfm;
    % u_check = 0.5*u_global(mesh.dof_unknown);
    % utheta_check = [theta_check;u_check];
    % [~, dcost_vec_check_ANA] = cost_dcost_vec_ANA(utheta_check);
    % [~, dcost_vec_check_FD] = cost_dcost_vec_FD(utheta_check);
    % disp('Derivative error:')
    % derror = dcost_vec_check_ANA(:,1:7)-dcost_vec_check_FD(:,1:7);
    % disp(num2str(maxabs(derror)));

    %% Postprocessing
    theta_aao = utheta_aao(1:2);
    E_aao = theta_aao(1)*(1-(theta_aao(2)/theta_aao(1))^2);
    nu_aao = theta_aao(2)/theta_aao(1);
    u_aao = zeros(size(u_global));
    u_aao(mesh.dof_known) = u_global(mesh.dof_known);
    u_aao(mesh.dof_unknown) = utheta_aao(3:end);

    disp('True theta:')
    disp(num2str(theta_true));
    disp('Calibrated theta (VFM):')
    disp(num2str(theta_vfm));
    disp('Initial guess theta (AAO):')
    disp(num2str(theta_init));
    disp('Calibrated theta (AAO):')
    disp(num2str(theta_aao));
    disp('True E:')
    disp(E_true);
    disp('Calibrated E (VFM):')
    disp(E_vfm);
    disp('Calibrated E (AAO):')
    disp(E_aao);
    disp('True nu:')
    disp(nu_true);
    disp('Calibrated nu (VFM):')
    disp(nu_vfm);
    disp('Calibrated nu (AAO):')
    disp(nu_aao);
    disp(' ')

    if plot_results
        figure
        set(gcf,'Position',[100 100 800 700])
        subplot(2,2,1)
        plot_scalar_field_2D(mesh.node,mesh.element,u_global(1:2:end),'newfigure',false,'mesh_overlay',false,'clabel','$u_1$ (EXP)')
        xlim([min(mesh_total.node(:,1)),max(mesh_total.node(:,1))])
        ylim([min(mesh_total.node(:,2)),max(mesh_total.node(:,2))])
        subplot(2,2,2)
        plot_scalar_field_2D(mesh.node,mesh.element,u_aao(1:2:end),'newfigure',false,'mesh_overlay',false,'clabel','$u_1$ (AAO)')
        xlim([min(mesh_total.node(:,1)),max(mesh_total.node(:,1))])
        ylim([min(mesh_total.node(:,2)),max(mesh_total.node(:,2))])
        subplot(2,2,3)
        plot_scalar_field_2D(mesh.node,mesh.element,u_global(2:2:end),'newfigure',false,'mesh_overlay',false,'clabel','$u_2$ (EXP)')
        xlim([min(mesh_total.node(:,1)),max(mesh_total.node(:,1))])
        ylim([min(mesh_total.node(:,2)),max(mesh_total.node(:,2))])
        subplot(2,2,4)
        plot_scalar_field_2D(mesh.node,mesh.element,u_aao(2:2:end),'newfigure',false,'mesh_overlay',false,'clabel','$u_2$ (AAO)')
        xlim([min(mesh_total.node(:,1)),max(mesh_total.node(:,1))])
        ylim([min(mesh_total.node(:,2)),max(mesh_total.node(:,2))])
        sgtitle([ ...
            '$E = $' num2str(E_aao) ...
            '$,~\nu = $' num2str(nu_aao) ...
            '$,~\lambda_r = $' num2str(lambda_r) ...
            '$,~\lambda_d = $' num2str(lambda_d) ...
            ])
        save_path = [datestr(now,'yyyymmddTHHMMSS') '_AAO_' norm_type '_' dataname '.jpg'];
        save_path = ['plot\' save_path];
        saveas(gcf,save_path)
    end

    if save_results
        %% Save Data
        results.dataname = dataname;
        results.norm_type = norm_type;
        results.lambda_r = lambda_r;
        results.lambda_d = lambda_d;
        results.options = options;
        results.u_global = u_global;
        results.u_aao = u_aao;
        results.theta_init = theta_init;
        results.theta_true = theta_true;
        results.theta_vfm = theta_vfm;
        results.theta_aao = theta_aao;
        results.E_true = E_true;
        results.E_vfm = E_vfm;
        results.E_aao = E_aao;
        results.nu_true = nu_true;
        results.nu_vfm = nu_vfm;
        results.nu_aao = nu_aao;
        results.time = toc;

        save_path = [datestr(now,'yyyymmddTHHMMSS') '_AAO_' norm_type '_' dataname];
        save_path = ['results\' save_path];
        save(save_path,'results')
    else
        disp(['Time needed: ' num2str(toc)])
    end

end

