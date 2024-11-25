function A = linEla_assemble_A(u,mesh,lambda_r)
% linEla_assemble_A assembles a matrix A such that A*theta=b
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

%% Check Input
if length(u) == mesh.n_dof
    u_global = u;
else
    error('Not implemented.')
end

%% Assemble A_global
n_unknown = 2;
A_global = zeros(mesh.n_dof,n_unknown);
for idx_ele = 1:mesh.n_element
    
    dof_ele = mesh.element_dof(idx_ele,:);
    A_ele = zeros(mesh.n_dof_per_element,n_unknown);
    u_ele = u_global(dof_ele);
    
    for idx_Gauss_x = 1:mesh.n_Gauss_per_dim
        for idx_Gauss_y = 1:mesh.n_Gauss_per_dim
            
            detJ = mesh.detJ_GP(idx_ele,idx_Gauss_x,idx_Gauss_y);
            B = squeeze(mesh.B_GP(idx_ele,idx_Gauss_x,idx_Gauss_y,:,:));
            epsilonV = B*u_ele;
            EPSILON = [ ...
                epsilonV(1), epsilonV(2); ...
                epsilonV(2), epsilonV(1); ...
                0.5*epsilonV(3), -0.5*epsilonV(3); ...
                ];
            A_ele = A_ele + B' * EPSILON * mesh.Gauss_weights(idx_Gauss_x) * mesh.Gauss_weights(idx_Gauss_y) * detJ;
            
        end
    end
    
    % assembly
    A_global(dof_ele,:) = A_global(dof_ele,:) + A_ele;
    
end

%% Matrix Rearrangement
A_free = A_global(mesh.dof_free,:);
A_reaction = zeros(mesh.n_reaction,size(A_free,2));
for idx = 1:mesh.n_reaction
    A_reaction(idx,:) = sum(A_global(mesh.dof_reaction(idx,:),:),1);
end
A = [A_free; sqrt(lambda_r)*A_reaction];

end