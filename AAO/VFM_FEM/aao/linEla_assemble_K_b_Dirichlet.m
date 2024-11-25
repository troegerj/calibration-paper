function [K_Dirichlet, b_Dirichlet] = linEla_assemble_K_b_Dirichlet(theta,u_global,mesh,lambda_r)
% linEla_assemble_K_b_Dirichlet assembles a matrix K_Dirichlet and a vector
% b_Dirichlet such that K_Dirichlet*u_unknown=b_Dirichlet, where u_unknown
% is a vector of unknown displacements
%
% ## Comments
% 
% Linear isotropic elasticity.
% 
% ## Input Arguments
% 
% 
% ## Output Arguments
% 
% 

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

%% Stiffness Matrix
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

%% Stiffness Matrix Rearrangement
% equations corresponding to the free degrees of freedom
K_free = K_global(mesh.dof_free,:);
% sum of equations corresponding to the reaction force degrees of freedom
K_reaction_sum = zeros(mesh.n_reaction,size(K_free,2));
for idx = 1:mesh.n_reaction
    K_reaction_sum(idx,:) = sum(K_global(mesh.dof_reaction(idx,:),:),1);
end
% combine equations
K_free_reaction = [K_free; sqrt(lambda_r)*K_reaction_sum];

% compose a system K*u=b, such that u is a vector containing only the
% unknown displacments
if isempty(mesh.dof_known)
    b_Dirichlet = zeros(size(K_free_reaction,1),1);
    K_Dirichlet = K_free_reaction;
else
    b_Dirichlet = - K_free_reaction(:,mesh.dof_known)*u_global(mesh.dof_known);
    K_Dirichlet = K_free_reaction(:,mesh.dof_unknown);
end

%%% assume that u is constant along the reaction force degrees of freedom
%%% K_Dirichlet = [K_free_reaction(:,mesh.dof_free), sum(K_free_reaction(:,mesh.dof_reaction),2)];

%% Check
% f_int_global = K_global * u_global; % == f_int_global
% f_int_free = K_free * u_global; % == 0
% reaction_horizontal = K_reaction_horizontal_sum * u_global; % == reaction_horizontal_true
% f_int_free_reaction = K_free_reaction * u_global; % == [0; reaction_horizontal_true]
% f_int_Dirichlet = K_Dirichlet * u_global(mesh.dof_unknown) - b_Dirichlet; % == [0; reaction_horizontal_true]

end