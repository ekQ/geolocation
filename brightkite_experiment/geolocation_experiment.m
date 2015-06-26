%%GEOLOCATION_EXPERIMENT Runs a home location prediction experiment.
%
% The experiment has been described in Section 4.1 of our paper:
%
% Eric Malmi, Arno Solin, Aristides Gionis, "The blind leading the
% blind: Network-based location estimation under uncertainty". In Proc.
% ECML PKDD 2015.
%
% (c) Eric Malmi

%% Load data
% The dataset we use has been derived from a SNAP dataset. See the
% following link for further information, including a citation you should
% include if you use this data:
% http://snap.stanford.edu/data/loc-brightkite.html
edges = load('data/Brightkite_edges.txt');
m = size(edges,1);
edges = edges + 1; % Make the indexing start from 1 instead of 0
G = sparse(edges(:,1), edges(:,2), ones(m,1));

% Groundtruth locations
true_locs = load('data/Brightkite_userLocations.txt');
% Make the indexing start from 1 instead of 0
true_locs(:,1) = true_locs(:,1) + 1;
n = max(max(edges(:)), max(true_locs(:,1)));
coords = zeros(n,2);
coords(true_locs(:,1),:) = true_locs(:,2:3);

missing_users = find(coords(:,1)==0 | coords(:,2)==0);
coords(missing_users,:) = [];
G(missing_users,:) = [];
G(:,missing_users) = [];
n = size(G,1)

%% Split to train and test data
n_train = floor(0.5*n);
all_idxs = randperm(n);
train_idxs = sort(all_idxs(1:n_train));
test_idxs = sort(all_idxs(n_train+1:end));

coords_tr = coords(train_idxs,:);
G_tr = G(train_idxs, train_idxs);
coords_te = coords(test_idxs,:);
G_te = G(test_idxs, test_idxs);

%% Study distance distribution for friends vs. strangers
friend_dists = [];
for i = 1:n_train
    loc = coords_tr(i,:);
    friend_locs = coords_tr(find(G_tr(:,i)),:);
    d = coord_dist(loc,friend_locs);
    friend_dists = [friend_dists; d];
end

% Compute distances between randomly selected user pairs
frac_other = 0.2;
n_other = floor(n_train * frac_other);
order = randperm(n_train);
other_dists = ones(n_other*(n_other-1),1);
tic
for i = 1:n_other
    idx = order(i);
    loc = coords_tr(idx,:);
    other_locs = coords_tr;
    other_locs(idx,:) = [];
    d = coord_dist(loc,other_locs);
    other_dists((i-1)*(n_train-1)+1:i*(n_train-1)) = d;
end
toc

% Compute friendship probability at a given (binned) distance
n_bins = 20;
max_bin = max(max(friend_dists), max(other_dists))*1.01;
min_bin = 0.1;

log_bins = log(min_bin):log(max_bin-min_bin)/n_bins:log(max_bin);
bins = exp(log_bins);
bin_widths = bins(2:end) - bins(1:end-1);
bin_centers = (bins(2:end) + bins(1:end-1)) / 2;

vals_f = histc(friend_dists,bins);
vals_f(end) = [];
vals_o = histc(other_dists,bins) / frac_other;
vals_o(end) = [];
probs = vals_f./vals_o;

% Plot friendship probability distribution
figure(1121), clf
loglog(bin_centers, probs, 'x'), hold on

P = polyfit(log(bin_centers)', log(probs), 1);
f = @(x) min(1, exp(polyval(P,log(x))));

xs = exp(log(min_bin):(log(max_bin)-log(min_bin))/100:log(max_bin));
ys = f(xs);
loglog(xs, ys, 'k')
xlim([1e-1 3e4])
xlabel('User-to-user distance (km)')
ylabel('Friendship probability')
legend('Observed data', sprintf('Best fit %.3fd^{%.2f}',exp(P(2)), P(1)), ...
    'Location','NorthEast')

%% Remove users without enough friends
bad_users = sum(G_te)<2';
coords_te(bad_users,:) = [];
G_te(bad_users,:) = [];
G_te(:,bad_users) = [];
n = size(G_te,1)

% Remove still the isolated vertices
bad_users = sum(G_te)==0';
coords_te(bad_users,:) = [];
G_te(bad_users,:) = [];
G_te(:,bad_users) = [];
n = size(G_te,1)

%% Settings
n_iters = 6;
% Vary the fraction of users whose locations are known exactly
frac_known = [0.1:0.1:0.9];
% Repeat with random initialization of known user locations
n_reps = 5;

max_error = 40; % km

%% Start the geolocation (running this may take several hours)
% MLE results
all_accs = zeros(length(frac_known),n_iters,n_reps);
all_errs = zeros(length(frac_known),n_iters,n_reps);
all_errors = cell(length(frac_known),n_reps);

% Backstrom results
all_accs_bl = zeros(length(frac_known),n_iters,n_reps);
all_errs_bl = zeros(length(frac_known),n_iters,n_reps);
all_errors_bl = cell(length(frac_known),n_reps);

% Jurgens results
all_accs_bl2 = zeros(length(frac_known),n_iters,n_reps);
all_errs_bl2 = zeros(length(frac_known),n_iters,n_reps);
all_errors_bl2 = cell(length(frac_known),n_reps);

tstart = tic;
for rep = 1:n_reps
    fprintf('\n\nRepetition number: %d\n\n', rep)
    parfor k = 1:length(frac_known)
        frac = frac_known(k);
        fprintf('\n\n------ Fraction known: %.2f ------\n\n', frac)
        [initial_location_distributions, uncertain_locs, exact_locs] = ...
            init_part_of_user_locations(coords_te, frac);

        location_distributions = initial_location_distributions;
        n_pred = length(uncertain_locs);
        fprintf('%d locations to predict.\n', n_pred);

        %% The proposed MLE method
        errors = zeros(n_pred,n_iters);
        all_accs_slice = zeros(n_iters,1);
        all_errs_slice = zeros(n_iters,1);
        for r = 1:n_iters
            new_loc_dists = location_distributions;
            % Iterate over users
            for i = 1:n_pred
                user = uncertain_locs(i);
                % Take neighbor locations as candidate locations
                neigh_idxs = find(G_te(:,user));
                neighbor_distributions = {};
                candidate_locations = [];
                for j = 1:length(neigh_idxs)
                    neigh = neigh_idxs(j);
                    if isempty(location_distributions{neigh})
                        continue
                    end
                    neighbor_distributions{end+1} = location_distributions{neigh};
                    [~,most_probable] = max(location_distributions{neigh}(:,3));
                    candidate_locations(end+1,:) = ...
                        location_distributions{neigh}(most_probable,1:2);
                end
                % Estimate
                if length(neighbor_distributions) > 0
                    [coord_mle, log_likelihoods] = infer_single_location(...
                        candidate_locations, neighbor_distributions, [], f);

                    true_coord = coords_te(user,:);
                    errors(i,r) = coord_dist(true_coord, coord_mle);
                    new_loc_dists{user} = [candidate_locations log_likelihoods];
                else
                    errors(i,r) = Inf;
                end
            end
            acc = sum(errors(:,r)<=max_error) / length(errors(:,r));
            all_accs_slice(r) = acc;
            all_errs_slice(r) = median(errors(:,r));
            fprintf('\n--- (MLE) Fraction %.2f, iteration %d, accuracy %.3f ---\n', ...
                frac, r, all_accs_slice(r));
            location_distributions = new_loc_dists;
        end
        all_accs(k,:,rep) = all_accs_slice;
        all_errs(k,:,rep) = all_errs_slice;
        all_errors{k,rep} = errors;

        %% Backstrom method (slightly simplified)
        % This method is equivalent to the proposed method with the
        % exception that we only keep track of the most probable location
        % for each individual
        errors_bl = zeros(n_pred,n_iters);
        location_distributions = initial_location_distributions;
        all_accs_bl_slice = zeros(n_iters,1);
        all_errs_bl_slice = zeros(n_iters,1);
        for r = 1:n_iters
            new_loc_dists = location_distributions;
            for i = 1:n_pred
                user = uncertain_locs(i);
                neigh_idxs = find(G_te(:,user));
                neighbor_locations = {};
                candidate_locations = [];
                for j = 1:length(neigh_idxs)
                    neigh = neigh_idxs(j);
                    if isempty(location_distributions{neigh})
                        continue
                    end
                    assert(numel(location_distributions{neigh}) == 3);
                    neighbor_locations{end+1} = location_distributions{neigh};
                    candidate_locations(end+1,:) = ...
                        location_distributions{neigh}(1:2);
                end
                if length(neighbor_locations) > 0
                    [coord_mle, log_likelihoods] = infer_single_location(...
                        candidate_locations, neighbor_locations, [], f);

                    true_coord = coords_te(user,:);
                    errors_bl(i,r) = coord_dist(true_coord, coord_mle);
                    % Store only the most probable location
                    new_loc_dists{user} = [coord_mle 0];
                else
                    errors_bl(i,r) = Inf;
                end
            end
            acc = sum(errors_bl(:,r)<=max_error) / length(errors_bl(:,r));
            all_accs_bl_slice(r) = acc;
            all_errs_bl_slice(r) = median(errors_bl(:,r));
            fprintf('\n--- (Backstrom) Fraction %.2f, iteration %d, accuracy %.3f ---\n', ...
                frac, r, all_accs_bl_slice(r));
            location_distributions = new_loc_dists;
        end
        all_accs_bl(k,:,rep) = all_accs_bl_slice;
        all_errs_bl(k,:,rep) = all_errs_bl_slice;
        all_errors_bl{k,rep} = errors_bl;

        %% Jurgens method
        errors_bl2 = zeros(n_pred,n_iters);
        location_distributions = initial_location_distributions;
        all_accs_bl2_slice = zeros(n_iters,1);
        all_errs_bl2_slice = zeros(n_iters,1);
        for r = 1:n_iters
            new_loc_dists = location_distributions;
            for i = 1:n_pred
                user = uncertain_locs(i);
                neigh_idxs = find(G_te(:,user));
                neighbor_locations = [];
                for j = 1:length(neigh_idxs)
                    neigh = neigh_idxs(j);
                    if isempty(location_distributions{neigh})
                        continue
                    end
                    assert(numel(location_distributions{neigh}) == 3);
                    neighbor_locations(end+1,:) = location_distributions{neigh}(1:2);
                end
                if size(neighbor_locations,1) > 0
                    coord_centroid = select_centroid(neighbor_locations);
                    true_coord = coords_te(user,:);
                    errors_bl2(i,r) = coord_dist(true_coord, coord_centroid);
                    new_loc_dists{user} = [coord_centroid 0];
                else
                    errors_bl2(i,r) = Inf;
                end
            end
            acc = sum(errors_bl2(:,r)<=max_error)/length(errors_bl2(:,r));
            all_accs_bl2_slice(r) = acc;
            all_errs_bl2_slice(r) = median(errors_bl2(:,r));
            fprintf('\n--- (Jurgens) Fraction %.2f, iteration %d, accuracy %.3f ---\n', ...
                frac, r, all_accs_bl2_slice(r));
            location_distributions = new_loc_dists;
        end
        all_accs_bl2(k,:,rep) = all_accs_bl2_slice;
        all_errs_bl2(k,:,rep) = all_errs_bl2_slice;
        all_errors_bl2{k,rep} = errors_bl2;
    end
end
%
telapsed = toc(tstart);
fprintf('Total time: %f hours\n', telapsed/3600);

mean_all_accs = mean(all_accs,3)
mean_all_accs_bl = mean(all_accs_bl,3)
mean_all_accs_bl2 = mean(all_accs_bl2,3)
mean_all_errs = mean(all_errs,3)
mean_all_errs_bl = mean(all_errs_bl,3)
mean_all_errs_bl2 = mean(all_errs_bl2,3)
improvement = mean_all_accs(:,end) - mean_all_accs_bl(:,end)

%% Plot performance curves
mean_all_accs = mean(all_accs,3);
mean_all_accs_bl = mean(all_accs_bl,3);
mean_all_accs_bl2 = mean(all_accs_bl2,3);
mean_all_errs = mean(all_errs,3);
mean_all_errs_bl = mean(all_errs_bl,3);
mean_all_errs_bl2 = mean(all_errs_bl2,3);

chosen_iter = 4;

figure(23), clf
subplot(1,2,1)
plot(frac_known, mean_all_accs(:,chosen_iter), '-o', ...
    'Color',[     0    0.4470    0.7410], 'LineWidth',1), hold on
plot(frac_known, mean_all_accs_bl(:,chosen_iter), '-x', ...
    'Color',[0.8500    0.3250    0.0980], 'LineWidth',1)
plot(frac_known, mean_all_accs_bl2(:,chosen_iter), '-v', ...
    'Color',[0.9290    0.6940    0.1250], 'LineWidth',1)
xlim([0.1 0.9])
xlabel('Fraction of user locations known exactly')
ylabel(sprintf('Fraction within %d km', max_error))
legend('MLE', 'Backstrom*', 'Jurgens', 'Location', 'SouthEast')
box on

subplot(1,2,2)
plot(frac_known, mean_all_errs(:,chosen_iter), '-o', ...
    'Color',[     0    0.4470    0.7410], 'LineWidth',1), hold on
plot(frac_known, mean_all_errs_bl(:,chosen_iter), '-x', ...
    'Color',[0.8500    0.3250    0.0980], 'LineWidth',1)
plot(frac_known, mean_all_errs_bl2(:,chosen_iter), '-v', ...
    'Color',[0.9290    0.6940    0.1250], 'LineWidth',1)
xlim([0.1 0.9])
xlabel('Fraction of user locations known exactly')
ylabel('Median error (km)')
legend('MLE', 'Backstrom*', 'Jurgens', 'Location', 'NorthEast')
box on, xlim([0 1]);
set(gca,'XTick',[0:.2:1])

%% Significance testing
test_mcnemar(all_errors, all_errors_bl, max_error, 0.001, chosen_iter)