function [chain,acceptanceRate] = MCMC(stepSize,chainLength,ensembleInit,logLikelihood,logPrior)
% function to carry out Markov-Chain Monte-Carlo simulations using
% affine-invariant ensemble sampler from Goodman/Weare (2010)
%
% INPUT
%  stepSize -- prescribed stretch scale
%  chainLength -- prescribed chain length
%  ensembleInit -- initial ensemble (numberParameter x numberWalker)
%  logLikelihood -- function handle for evaluation of log-likelihood
%  logPrior -- function handle for prior evaluation
%
% OUTPUT
%  chain -- Markov chains
%  acceptanceRate -- acceptance rate, see Goodman/Weare for details

% determine number of parameters
numberParameter = size(ensembleInit,1);

acceptanceRate = 0;
numberWalker = size(ensembleInit,2);
chainDim = size(ensembleInit,1)*numberWalker;
chain = zeros(chainDim,chainLength);
chain(:,1) = reshape(ensembleInit,[chainDim,1]);

% loop over chain length
for k = 2:chainLength
    [chain(:,k),acceptanceRateStep] = stretch_move(chain(:,k-1),numberWalker,stepSize,logLikelihood,logPrior);
    acceptanceRate = acceptanceRate + acceptanceRateStep;
end

% calculate acceptance rate w.r.t. chain length
acceptanceRate = acceptanceRate/(chainLength-1);

% remove burn-in
chain = chain(:,floor(end/2):end);

% combine all walkers into one chain
chainLengthAfterBurnin = size(chain,2);
chain_tmp = chain;
chain = [];
for i = 1 : numberParameter
    chain = [chain; reshape(chain_tmp(i:numberParameter:end,:),[1,chainLengthAfterBurnin*numberWalker])];
end

end
