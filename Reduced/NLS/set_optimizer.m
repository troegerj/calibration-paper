function opt_options = set_optimizer
% function providing the settings for the optimization using
% lsqnonlin-function
%
% INPUT
%  none
%
% OUTPUT
%  opt_options -- option-structure for lsqnonlin-optimizer

SolverName                  = 'lsqnonlin';
%
% choose between 'trust-region-reflective' (default) and 'levenberg-marquardt'
algorithm                   = 'trust-region-reflective';
%
% Display diagnostic information about the function to be minimized
% or solved. Choices are 'off' (default) or 'on'.
Diagnostics                 = 'on';
% Level of display (see Iterative Display):
%**  'off' or 'none' displays no output.
%**  'iter' displays output at each iteration, and gives the default exit message.
%**  'iter-detailed' displays output at each iteration, and gives the technical exit message.
%**  'final' (default) displays just the final output, and gives the default exit message.
%**  'final-detailed' displays just the final output, and gives the technical exit message.
display_level               = 'iter-detailed';
%
% check whether function values are valid.
% 'on' displays an error when the function returns a value that is
% complex, Inf, or NaN. The default 'off' displays no error.
FunValCheck                 = 'on';
%
% Maximum number of function evaluations allowed, a positive integer.
% The default is 100*numberOfVariables.
% MaxFunctionEvaluations is a bound on the number of function evaluations.
MaxFunEvals                = 10000;
%
% MaxIterations: Maximum number of iterations allowed, a positive integer.
% The default is 400.
% MaxIterations is a bound on the number of solver iterations.
MaxIterations              = 100;
%
% StepTolerance: Termination tolerance on x, a positive scalar.
% The default is 1e-6.
TolX                       = 1e-6;
%
% FunctionTolerance: Termination tolerance on the function value,
% a positive scalar. The default is 1e-6.
TolFun                     = 1e-6;
%
% OptimalityTolerance: Termination tolerance on
% the first-order optimality measure, a positive scalar.
% The default is 1e-6.
OptimalityTolerance        = 1e-6;
%
% PlotFcn: plots various measures of progress while the algorithm executes;
% select from predefined plots or write your own.
% Pass a function handle or a cell array of function handles.
% The default is none ([]):
% We use predfined functions for this task.
PlotFcn                    = {...
    @optimplotx,...
    @optimplotfunccount,...
    @optimplotfval,...
    @optimplotresnorm,...
    @optimplotstepsize,...
    @optimplotfirstorderopt};
%
% OutputFcn: specify one or more user-defined functions that an
% optimization function calls at each iteration,
% either as a function handle or as a cell array of function handles.
% The default is none ([]).
OutputFcn                  = {@History_Output_lsqnonlin};
%
% SpecifyObjectiveGradient:
% If false (default), the solver approximates
% the Jacobian using finite differences.
% If true, the solver uses a user-defined Jacobian (defined in fun),
% or Jacobian information (when using JacobMult),
% for the objective function.
SpecifyObjectiveGradient   = false;
%
% Compare user-supplied derivatives (gradients of objective or constraints)
% to finite-differencing derivatives. Choices are false (default) or true.
CheckGradients            = true;
%
% FiniteDifferenceType: Finite differences, used to estimate gradients,
% are either 'forward' (default), or 'central' (centered).
% 'central' takes twice as many function evaluations,
% but should be more accurate. The algorithm is careful to obey bounds
% when estimating both types of finite differences.
% So, for example, it could take a backward, rather
% than a forward, difference to avoid evaluating at a point outside bounds.
FiniteDifferenceType       = 'central';
%
% UseParallel: When true, the solver estimates gradients in parallel.
% Disable by setting to the default, false.
UseParallel                = false;

% set options of optimizer
opt_options=optimoptions(SolverName,...
    'Algorithm',algorithm,...
    'Diagnostics',Diagnostics,...
    'Display',display_level,...
    'FunValCheck',FunValCheck,...
    'MaxIterations',MaxIterations,...
    'MaxFunEvals',MaxFunEvals,...
    'StepTolerance',TolX,...
    'FunctionTolerance',TolFun,...
    'OptimalityTolerance',OptimalityTolerance,...
    'PlotFcn',PlotFcn,...
    'OutputFcn',OutputFcn,...
    'SpecifyObjectiveGradient',SpecifyObjectiveGradient,...
    'FiniteDifferenceType',FiniteDifferenceType,....
    'CheckGradients',CheckGradients,...
    'UseParallel',UseParallel);

end