function d = coord_dist(coord1, coord2, type)
%%COORD_DIST Distance between geographic coordinates.
%
% Input:
%   coord1  Source coordinates (1x2).
%   coord2  Target coordinates (kx2).
%   type    'latlon' (default) computes Haversine distance, 'euclidean'
%           computes the Euclidean distance.
%
% Output:
%   Distance(s) from the source to the target(s).
%
% (c) Eric Malmi

if nargin < 3
    type = 'latlon';
end

if strcmp(type, 'latlon')
    d = lldistkm(coord1, coord2);
elseif strcmp(type, 'euclidean')
    d = transpose(pdist2(coord1, coord2))/1000;
end