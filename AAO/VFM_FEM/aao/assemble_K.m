function K = assemble_K(theta,mesh,lambda_r)

%% Material Parameters
if length(theta) == 2
    CPlaneStress = [ ...
        theta(1), theta(2), 0; ...
        theta(2), theta(1), 0; ...
        0, 0, (theta(1)-theta(2))/2; ...
        ];
elseif any(size(theta) == [3, 3])
    CPlaneStress = theta;
else
    error('Material parameters are given in the wrong format.')
end

%% Misc
n_dof = mesh.n_dof;
n_element = mesh.n_element;
n_dof_per_element = mesh.n_dof_per_element;
element_dof = mesh.element_dof;
Gauss_weights = mesh.Gauss_weights;
n_Gauss_per_dim = mesh.n_Gauss_per_dim;
detJ_GP = mesh.detJ_GP;
B_GP = mesh.B_GP;
% dof_fix = bc.dof_fix;
% dof_displacement = bc.dof_displacement;
% dof_reaction = bc.dof_reaction;

%% BC
% displacement = bc.displacement;
% dof_Dirichlet = [dof_fix, dof_displacement];
% value_Dirichlet = [zeros(size(dof_fix)), displacement];

%% Solve
% f_global = zeros(n_dof,1);
K_global = zeros(n_dof);
for idx_ele = 1:n_element
    
    K_ele = zeros(n_dof_per_element);
    dof_ele = element_dof(idx_ele,:);
    
    for idx_Gauss_x = 1:n_Gauss_per_dim
        for idx_Gauss_y = 1:n_Gauss_per_dim
            
            detJ = detJ_GP(idx_ele,idx_Gauss_x,idx_Gauss_y);
            B = squeeze(B_GP(idx_ele,idx_Gauss_x,idx_Gauss_y,:,:));
            
            K_ele = K_ele + ...
                B' * CPlaneStress * B * Gauss_weights(idx_Gauss_x) * Gauss_weights(idx_Gauss_y) * detJ;
            
        end
    end
    
    % assembly
    K_global(dof_ele,dof_ele) = K_global(dof_ele,dof_ele) + K_ele;
    
end

% % Dirichlet boundary conditions
% K_global_BC = K_global;
% for d = 1:length(dof_Dirichlet)
%     K_global_BC(dof_Dirichlet(d),:) = 0;
%     K_global_BC(dof_Dirichlet(d),dof_Dirichlet(d)) = 1;
%     % no external force (f_ext_global)
%     f_global(dof_Dirichlet(d)) = value_Dirichlet(d);
% end

% rearrange
K_free = K_global(mesh.dof_free,:);
K_reaction_x_top = sum(K_global(mesh.dof_top_x,:),1);
K_reaction_y_top = sum(K_global(mesh.dof_top_y,:),1);
K_reaction_top = [K_reaction_x_top;K_reaction_y_top];
K_underdet = [K_free; sqrt(lambda_r)*K_reaction_top];
% we assume that the displacement is zero at the bottom boundary and
% constant at the top boundary
K = [K_underdet(:,mesh.dof_free), sum(K_underdet(:,mesh.dof_top_x),2), sum(K_underdet(:,mesh.dof_top_y),2)];

end