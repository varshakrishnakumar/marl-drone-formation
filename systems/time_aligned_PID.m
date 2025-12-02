function results = time_aligned_PID(minSteps)
% TIME_ALIGNED_PID  Time-aligned formation-error and collision stats
% for all PID episodes in the current directory.
%
% Usage:
%   cd PID
%   results = time_aligned_PID;       % uses default 300 steps
%   results = time_aligned_PID(400);  % custom minSteps
%

    if nargin < 1
        minSteps = 300;   % default cutoff
    end

    fprintf("\n=== PID: using minimum %d steps ===\n", minSteps);

    files = dir("*.csv");
    nFiles = numel(files);
    if nFiles == 0
        error("No .csv files found in current directory.");
    end

    errList = {};    % each cell: [minSteps x 1] formation error
    colList = {};    % each cell: [minSteps x 1] collision flag

    for k = 1:nFiles
        fname = fullfile(files(k).folder, files(k).name);

        % Skip any obvious summary files by name
        if contains(files(k).name, "summary", "IgnoreCase", true)
            fprintf("  Skipping %s (summary file)\n", files(k).name);
            continue;
        end

        T = readtable(fname);
        vn = T.Properties.VariableNames;

        % --- Clean out NaN-only / header-like first row if needed ---
        if ismember("step", vn)
            % Drop rows with NaN step
            valid = ~isnan(T.step);
            T = T(valid, :);
        end

        N = height(T);
        if N < minSteps
            fprintf("  Skipping %s (only %d valid steps)\n", files(k).name, N);
            continue;
        end

        vn = T.Properties.VariableNames;  % refresh after filtering

        % ---------- 1) Formation error ----------
        fe = [];

        % Preferred: your example PID log
        if ismember("mean_form_error_m", vn)
            fe = T.mean_form_error_m(1:minSteps);

        % Fallbacks if you reuse this script elsewhere
        elseif ismember("mean_form_error", vn)
            fe = T.mean_form_error(1:minSteps);
        elseif ismember("followers_mean_pos_err_m", vn)
            fe = T.followers_mean_pos_err_m(1:minSteps);
        end

        if isempty(fe)
            warning("No usable formation error column in %s; skipping.", files(k).name);
            continue;
        end

        % ---------- 2) Collision flag ----------
        if ismember("collision_flag", vn)
            col = T.collision_flag(1:minSteps);
        elseif ismember("collision", vn)
            col = T.collision(1:minSteps);
        else
            % No collision info: assume 0
            col = zeros(minSteps,1);
        end

        errList{end+1} = fe(:);
        colList{end+1} = col(:);
    end

    if isempty(errList)
        error("No usable PID episodes (>= %d steps) with valid formation error.", minSteps);
    end

    % Stack into matrices [steps x episodes]
    errMat = cell2mat(errList);
    colMat = cell2mat(colList);

    nEpisodes = size(errMat, 2);
    fprintf("PID episodes used: %d\n", nEpisodes);

    % ---------- time-aligned statistics ----------
    steps          = (1:minSteps)';
    mean_err       = mean(errMat, 2);
    std_err        = std(errMat, 0, 2);
    collision_prob = mean(colMat, 2);

    % Pack results
    results.steps          = steps;
    results.mean_err       = mean_err;
    results.std_err        = std_err;
    results.collision_prob = collision_prob;
    results.num_episodes   = nEpisodes;

    % ---------- plots ----------
    % Formation error
    figure; hold on;
    fill([steps; flipud(steps)], ...
         [mean_err - std_err; flipud(mean_err + std_err)], ...
         [1 0.7 0.7], "EdgeColor","none", "FaceAlpha",0.3);
    plot(steps, mean_err, "r", "LineWidth", 2);
    xlabel("Time step");
    ylabel("Mean formation error (m)");
    title(sprintf("PID formation tracking (N = %d episodes)", nEpisodes));
    legend("PID \\pm std","PID mean");
    grid on;

    % Collision probability
    figure;
    plot(steps, collision_prob, "r", "LineWidth", 2);
    xlabel("Time step");
    ylabel("P(collision)");
    title(sprintf("PID collision probability (N = %d episodes)", nEpisodes));
    grid on;
end
