function centroid = select_centroid(coords, candidate_locations)
%%SELECT_CENTROID finds the geometric median of given locations.

if nargin < 2
    candidate_locations = coords;
end

n = size(candidate_locations,1);
centroid = [];
centroid_cost = Inf;
for i = 1:n
    loc = candidate_locations(i,:);
    dists = coord_dist(loc, coords);
    cost = sum(dists);
    if cost < centroid_cost
        centroid_cost = cost;
        centroid = candidate_locations(i,:);
    end
end
