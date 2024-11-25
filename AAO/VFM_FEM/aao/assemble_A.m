function A = assemble_A(u,mesh,lambda_r)

if length(u) == mesh.n_dof
    u_global = u;
else
    % a reduced displacement vector is given
    u_global = zeros(mesh.n_dof,1);
    u_global(mesh.dof_free) = u(1:end-2);
    u_global(mesh.dof_top_x) = u(end-1);
    u_global(mesh.dof_top_y) = u(end-0);
end

n_unknown = 2;
A_global = zeros(mesh.n_dof,n_unknown);
% f_int_global = zeros(mesh.n_dof,1);

for idx_ele = 1:mesh.n_element
    
    dof_ele = mesh.element_dof(idx_ele,:);
    A_ele = zeros(mesh.n_dof_per_element,n_unknown);
    u_ele = u_global(dof_ele);
%     f_int_ele = zeros(mesh.n_dof_per_element,1);
    
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
    %     f_int_global(dof_ele) = f_int_global(dof_ele) + f_int_ele;
    
end

A_free = A_global(mesh.dof_free,:);
A_reaction_x_top = sum(A_global(mesh.dof_top_x,:),1);
A_reaction_y_top = sum(A_global(mesh.dof_top_y,:),1);
A_reaction_top = [A_reaction_x_top;A_reaction_y_top];
A = [A_free; sqrt(lambda_r)*A_reaction_top];

end