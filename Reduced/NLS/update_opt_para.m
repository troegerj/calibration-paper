function opt_para = update_opt_para(opt_para,optvec_norm,optvec_normalized,optvec_flags)
% function to update the parameters during optimization. Lsqnonlin
% normalizes parameters during optimization, here the normalization is
% undone to perform the model evaluation afterwards.
%
% INPUT
%  opt_para -- parameter structure for calibration
%  optvec_norm -- current values for normalized optimization parameters
%  optvec_normalized -- normalization values for optimization parameters
%  optvec_flags -- logical values indicating optimized parameters
%
% OUTPUT
%  opt_para -- parameter structure for calibration

j = 0;
for k = 1 : length(opt_para)

    % check if parameter is optimized
    if ~(optvec_flags(k))
        continue
    end
    
    % update counter
    j = j + 1;
    % save fitted parameter (undo normalization)
    opt_para(k).value = optvec_norm(j)*optvec_normalized(j);

end

end