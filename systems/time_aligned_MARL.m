function results = time_aligned_MARL(minSteps)
% TIME_ALIGNED_MARL  Time-aligned formation-error and collision stats
% for all MARL episodes in the current directory.
%
% Usage:
%   cd MARL
%   results = time_aligned_MARL;       % uses default 300 steps
%   results = time_aligned_MARL(400);  % custom minSteps

    if nargin < 1
        minSteps = 300;   % default cutoff
    end

    fprintf("\n=== MARL: using minimum %d steps ===\n", minSteps);

    files = dir("*.csv");
    nFiles = numel(files);
    if nFiles == 0
        error("No .csv files found in current directory.");
    end

    errList = {};
    colList = {};

    for k = 1:nFiles
        fname = fullfile(files(k).folder, files(k).name);
        T = readtable(fname);

        if height(T) < minSteps
            fprintf("  Skipping %s (only %d steps)\n", files(k).name, height(T));
            continue;
        end

        vn = T.Properties.VariableNames;

        % --- formation error column ---
        if ismember("mean_form_error", vn)
            fe = T.mean_form_error(1:minSteps);
        elseif ismember("followers_mean_pos_err_m", vn)
            fe = T.followers_mean_pos_err_m(1:minSteps);
        else
            warning("No formation error column found in %s; skipping.", files(k).name);
            continue;
        end

        % --- collision column (optional) ---
        if ismember("collision", vn)
            col = T.collision(1:minSteps);
        else
            col = zeros(minSteps,1);
        end

        errList{end+1} = fe(:);
        colList{end+1} = col(:);
    end

    if isempty(errList)
        error("No usable episodes (>= %d steps) in MARL folder.", minSteps);
    end

    errMat = cell2mat(errList);
    colMat = cell2mat(colList);

    nEpisodes = size(errMat, 2);
    fprintf("MARL episodes used: %d\n", nEpisodes);

    steps = (1:minSteps)';
    mean_err       = mean(errMat, 2);
    std_err        = std(errMat, 0, 2);
    collision_prob = mean(colMat, 2);

    results.steps           = steps;
    results.mean_err        = mean_err;
    results.std_err         = std_err;
    results.collision_prob  = collision_prob;
    results.num_episodes    = nEpisodes;

    % --- plots ---
    % Formation error
    figure; hold on;
    fill([steps; flipud(steps)], ...
         [mean_err - std_err; flipud(mean_err + std_err)], ...
         [0.7 0.7 1], "EdgeColor","none", "FaceAlpha",0.3);
    plot(steps, mean_err, "b", "LineWidth", 2);
    xlabel("Time step");
    ylabel("Mean formation error (m)");
    title(sprintf("MARL formation tracking (N = %d episodes)", nEpisodes));
    legend("MARL \pm std","MARL mean");
    grid on;

    % Collision probability
    figure;
    plot(steps, collision_prob, "b", "LineWidth", 2);
    xlabel("Time step");
    ylabel("P(collision)");
    title(sprintf("MARL collision probability (N = %d episodes)", nEpisodes));
    grid on;
end
