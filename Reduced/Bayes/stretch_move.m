function [ensembleNew,acceptanceRate] = stretch_move(ensembleCurrent,numberWalker,stepSize,logLikelihood,logPrior)
% function performing the stretch move within the affine-invariant ensemble
% sampler
%
% INPUT
%  ensembleCurrent -- current ensemble
%  numberWalker -- number of walkers
%  stepSize -- prescribed stretch scale
%  logLikelihood -- function handle for evaluation of log-likelihood
%  logPrior -- function handle for prior evaluation
%
% OUTPUT
%  ensembleNew -- updated ensemble after stretch move
%  acceptanceRate -- acceptance rate, see Goodman/Weare for details

numberParameter = length(ensembleCurrent)/numberWalker;
ensembleCurrent = reshape(ensembleCurrent,[numberParameter,numberWalker]);
ensembleNew = ensembleCurrent;
acceptanceNumber = 0;

for k = 1:numberWalker

    % randomly select partner chain
    indsWalker = 1:numberWalker;
    indsWalker(k) = [];
    partnerChain = indsWalker(randi(numberWalker-1));
    
    % generate candidate
    Z = ((stepSize - 1)*rand(1) + 1)^2 / stepSize;
    candidate = ensembleCurrent(:,partnerChain) + ...
        Z*(ensembleCurrent(:,k) - ensembleCurrent(:,partnerChain));
    
    % accept/reject
    if any(candidate < 0) % prevent from negative parameters
        logAcceptanceProbability = -Inf;
    elseif any(candidate > 1) % parameter should not exceed 1
        logAcceptanceProbability = -Inf;
    else
        logAcceptanceProbability = (numberParameter-1)*log(Z) + logLikelihood(candidate) ...
        + logPrior(candidate) - logLikelihood(ensembleCurrent(:,k)) - logPrior(ensembleCurrent(:,k));
    end
    
    acceptanceProbability = min(1,exp(logAcceptanceProbability));
    
    if rand(1) < acceptanceProbability
        ensembleNew(:,k) = candidate;
        acceptanceNumber = acceptanceNumber + 1;
    end
end

acceptanceRate = acceptanceNumber/numberWalker;
ensembleNew = reshape(ensembleNew,[numberParameter*numberWalker,1]);

end