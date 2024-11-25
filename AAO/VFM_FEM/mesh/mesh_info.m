function mesh = mesh_info(mesh)
% mesh_info generates important information for a given finite element mesh
%
% ## Comments
% 
% Given a finite element mesh, this function computes information such as
% the number of degrees of freedom, the number of elements, the Jacobian of
% the mapping between physical and reference element, the shape function
% derivatives etc..
% 
% ## Input Arguments
% 
% `mesh` (_struct_) - structural array containing information about the
% finite element mesh
% 
% ## Output Arguments
% 
% `mesh` (_struct_) - structural array containing information about the
% finite element mesh such as for example the determinant of the Jacobian
% `mesh.detJ_GP` and the shape function derivatives `mesh.B_GP` at each
% Gauss point
% 

mesh.n_dof_per_node = 2;
mesh.n_node = size(mesh.node,1);
mesh.n_dof = mesh.n_dof_per_node*mesh.n_node;
mesh.dof = (1:1:mesh.n_dof);
mesh.n_element = size(mesh.element,1);
mesh.n_node_per_element = size(mesh.element,2);
mesh.n_dof_per_element = mesh.n_dof_per_node*mesh.n_node_per_element;
mesh.element_dof = zip_array(2*mesh.element-1,2*mesh.element);

%% Gaussian quadrature information
mesh.Gauss_points = [-1/sqrt(3); 1/sqrt(3)];
mesh.Gauss_weights = [1; 1];
mesh.n_Gauss_per_dim = length(mesh.Gauss_points);
mesh.n_Gauss_per_element = mesh.n_Gauss_per_dim^size(mesh.node,2);
mesh.n_Gauss = mesh.n_Gauss_per_element*mesh.n_element;

%% Gauss point quantities
mesh.detJ_GP = zeros(mesh.n_element,mesh.n_Gauss_per_dim,mesh.n_Gauss_per_dim);
mesh.B_GP = zeros(mesh.n_element,mesh.n_Gauss_per_dim,mesh.n_Gauss_per_dim,3,mesh.n_dof_per_element);
for idx_ele = 1:mesh.n_element
    X = mesh.node(mesh.element(idx_ele,:),:);
    for idx_Gauss_x = 1:mesh.n_Gauss_per_dim
        for idx_Gauss_y = 1:mesh.n_Gauss_per_dim
            xi = mesh.Gauss_points(idx_Gauss_x);
            eta = mesh.Gauss_points(idx_Gauss_y);
            [N,dNdxi] = lagrange_basis('Q4',[xi eta]);
            J = X'*dNdxi;
            detJ = det(J);
            dNdx = dNdxi / J;
            B = zeros(3,mesh.n_dof_per_element);
            B(1,1:2:end-1) = dNdx(:,1)';
            B(2,2:2:end) = dNdx(:,2)';
            B(3,1:2:end-1) = dNdx(:,2)';
            B(3,2:2:end) = dNdx(:,1)';
            
            % save Gauss point quantities
            mesh.detJ_GP(idx_ele,idx_Gauss_x,idx_Gauss_y) = detJ;
            mesh.B_GP(idx_ele,idx_Gauss_x,idx_Gauss_y,:,:) = B;
        end
    end
end

end