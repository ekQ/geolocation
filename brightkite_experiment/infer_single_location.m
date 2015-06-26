function [coord_mle, log_likelihoods] = infer_single_location(...
    candidate_locations, neighbor_distributions, neighbor_types, ...
    kernel_functions, do_hard_iteration)
%%Find maximum likelihood estimate for location.
%
% Input:
%   candidate_locations     Coordinates of all n candidate locations (nx2).
%   neighbor_distributions  Cell array of location distributions of the
%                           neighboring objects. cell{i}(j,:) = [lat lon
%                           prob] of the jth location of the ith neighbor.
%   neighbor_types          Integer array indicating which kernel is used
%                           for evaluating edge probability.
%   kernel_functions        List of all kernel functions that give the 
%                           probability of an edge given the distance. If
%                           neighbor_types is not provided, then this is
%                           assumed to be just a single function handle.
%   do_hard_iteration       Whether we hard-assign the neighbors to their
%                           most likely location after each iteration.
%
% Output:
%   coord_mle       Coordinates of the MLE estimate.
%   likelihoods     Likelihoods of all candidate locations.
%
% (c) Eric Malmi

if nargin >= 5 && do_hard_iteration
    hard_iteration = true;
else
    hard_iteration = false;
end

nc = size(candidate_locations,1);
if nc == 1
    coord_mle = candidate_locations(1,:);
    log_likelihoods = 1;
    return
end

log_likelihoods = zeros(nc,1);
% Go through candidate locations
for i = 1:nc
    loc = candidate_locations(i,:);
    log_lh = 0;
    % Go through neighbors
    for j = 1:length(neighbor_distributions)
        neigh_locs = neighbor_distributions{j}(:,1:2);
        neigh_log_probs = neighbor_distributions{j}(:,3);

        dists = coord_dist(loc, neigh_locs);
        if isempty(neighbor_types)
            kernel = kernel_functions;
        else
            kernel = kernel_functions(neighbor_types(j));
        end
        if ~hard_iteration
            log_lh_sum = logsumexp(neigh_log_probs + log(kernel(dists)));
            log_lh = log_lh + log_lh_sum;
        else
            [~,max_idx] = max(neigh_log_probs);
            log_lh = log_lh + log(kernel(dists(max_idx)));
        end
    end
    log_likelihoods(i) = log_lh;
end
[~,loc_idx] = max(log_likelihoods);
coord_mle = candidate_locations(loc_idx,:);
log_likelihoods = log_likelihoods - logsumexp(log_likelihoods);