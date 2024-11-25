function b = assemble_b(reaction,mesh,lambda_r)

b = [zeros(length(mesh.dof_free),1); sqrt(lambda_r)*reaction];

end