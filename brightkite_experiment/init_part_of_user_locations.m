function [location_distributions, uncertain_locs, exact_locs] = ...
    init_part_of_user_locations(true_coords, fraction, max_uncertains)
%%INIT_PART_OF_USER_LOCATIONS initializes a location distribution data
% structure with a given fraction of users correctly located.

n = size(true_coords,1);
location_distributions = cell(n,1);
users = randperm(n);
exact_locs = users(1:floor(fraction*n));
uncertain_locs = users(floor(fraction*n)+1:end);
if nargin >= 3
    uncertain_locs(max_uncertains+1:end) = [];
end
for i = 1:length(exact_locs)
    uid = exact_locs(i);
    location_distributions{uid} = [true_coords(uid,:) 1];
end