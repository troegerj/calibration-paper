function stop = History_Output_lsqnonlin(x,optimValues,state)
% function to track the optimization process using lsqnonlin-function
%
% INPUT
%  x -- parameter vector
%  optimValues -- current values of optimization procedure (number of
%                 iterations, gradient, first-order optimality, etc.
%  state -- state switch
%
% OUTPUT
%  stop -- flag to indicate whether optimization should stop (true) or
%          continue (false)


% function is called at end of 0th iteration, after n function
% calls --> does not represent the exact starting point and would
% give the simulation results of the last varied optimization
% parameter

% Iteration number — starts at 0. Final value equals optimization function output output.iterations.
%
% First-order optimality (depends on algorithm).
% Final value equals optimization function output output.firstorderopt.
%
% Cumulative number of function evaluations.
% Final value equals optimization function output output.funcCount.

% initialize flag
stop = false;

% determine counter
i0 = optimValues.iteration + 1;

% update history
switch state
    case 'init' % the algorithm is in the initial state before the first iteration.
        save_infos()
    case 'iter' % the algorithm is at the end of an iteration.
        save_infos()
    case 'done' % the algorithm is in the final state after the last iteration.
        save_infos()
    otherwise
        % do nothing
end

    function save_infos()
        % save parameter vector
        history.opt_parVec(i0,:)           = x';
        % save iteration numnber
        history.iteration(i0)              = optimValues.iteration;
        % save cumulative number of function evaluations
        history.funccount(i0)              = optimValues.funccount;
        % save First-order optimality
        % in the case of unconstrained optmization problems
        % the first-order optimality measure is the infinity norm
        % (meaning maximum absolute value) of ∇f(x)
        if (isfield(optimValues,'firstorderopt') && ~isempty(optimValues.firstorderopt))
            history.firstorderopt(i0)      = optimValues.firstorderopt;
        end
        % compute own measure for first derivative (should be identical to first-order optimality measure of MATLAB)
        % optimValues.gradient: Current gradient of objective function — either analytic gradient
        % if you provide it or finite-differencing approximation.
        % Final value equals optimization function output grad.
        if (isfield(optimValues,'gradient') && ~isempty(optimValues.gradient))
            history.firstorderopt_user(i0) = max(optimValues.gradient);
        end
        % optimValues.positivedefinite: 0 if algorithm detects negative curvature while computing Newton step; 1 otherwise.
        if (isfield(optimValues,'positivedefinite') && ~isempty(optimValues.positivedefinite))
            history.positivedefinite(i0)   = optimValues.positivedefinite;
        end
        % save ratio of change in the objective function to change in the quadratic approximation
        if (isfield(optimValues,'ratio') && ~isempty(optimValues.ratio))
            history.ratio(i0)              = optimValues.ratio;
        end
        % save 2-norm of the residual squared.
        history.resnorm(i0)                = optimValues.resnorm;
        history.resnorm_user(i0)           = optimValues.residual'*optimValues.residual;
        %             % save search direction
        %             history.searchdirection(i0)      = optimValues.searchdirection;
        % save current step size (displacement in x). Final value equals optimization function output output.stepsize.
        if (isfield(optimValues,'stepsize') && ~isempty(optimValues.stepsize))
            history.stepsize(i0)           = optimValues.stepsize;
        end
        % save radius of trust region
        if (isfield(optimValues,'trustregionradius') && ~isempty(optimValues.trustregionradius))
            history.trustregionradius(i0)  = optimValues.trustregionradius;
        end
    end

end