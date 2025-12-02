function results = analyze_MARL_advanced(minSteps)
% ANALYZE_MARL_ADVANCED
%   Advanced analysis of MARL formation-control logs in current folder.
%
%   Computes:
%     1) Episode survival curve
%     2) Cumulative collision probability
%     3) Mean +/- std min_dyn_distance vs time
%     4) Accuracy vs safety scatter
%     5) Time-aligned formation error vs time
%
% Usage:
%   cd MARL
%   results = analyze_MARL_advanced;         % default 300 steps
%   results = analyze_MARL_advanced(400);    % custom horizon

    if nargin < 1
        minSteps = 300;
    end
    fprintf("\n=== MARL advanced analysis (minSteps = %d) ===\n", minSteps);

    episodes = load_marl_episodes();
    Ne = numel(episodes);
    if Ne == 0
        error("No usable MARL episodes found.");
    end
    fprintf("Loaded %d MARL episodes.\n", Ne);
    lens = [episodes.len]';

    % ===== 1) SURVIVAL CURVE =====
    maxLen = max(lens);
    t_surv = (1:maxLen)';
    surv   = zeros(maxLen,1);
    for ii = 1:maxLen
        surv(ii) = mean(lens >= ii);
    end

    figure;
    plot(t_surv, surv, "b", "LineWidth", 2);
    xlabel("Time step");
    ylabel("P(episode still running)");
    title(sprintf("MARL survival curve (N = %d episodes)", Ne));
    grid on;

    results.survival.t    = t_surv;
    results.survival.prob = surv;

    % ===== 2) CUMULATIVE COLLISION PROBABILITY =====
    T = min(minSteps, maxLen);
    cumMat = false(T, Ne);
    for k = 1:Ne
        c = episodes(k).coll ~= 0;
        if isempty(c), continue; end
        cum = cumsum(c) > 0;
        if numel(cum) >= T
            cumMat(:,k) = cum(1:T);
        else
            cumMat(1:numel(cum),k) = cum;
            cumMat(numel(cum)+1:end,k) = cum(end);
        end
    end
    cumProb = mean(cumMat, 2);

    figure;
    plot((1:T)', cumProb, "b", "LineWidth", 2);
    xlabel("Time step");
    ylabel("P(collision has occurred by t)");
    title(sprintf("MARL cumulative collision probability (N = %d episodes)", Ne));
    grid on;

    results.cumCollision.t    = (1:T)';
    results.cumCollision.prob = cumProb;

    % ===== 3) MIN DYNAMIC DISTANCE VS TIME =====
    D = nan(T, Ne);
    for k = 1:Ne
        d = episodes(k).mindist;
        if isempty(d), continue; end
        if numel(d) >= T
            D(:,k) = d(1:T);
        else
            D(1:numel(d),k) = d(:);
        end
    end
    meanD = mean(D, 2, "omitnan");
    stdD  = std(D, 0, 2, "omitnan");

    figure; hold on;
    tt = (1:T)';
    fill([tt; flipud(tt)], ...
         [meanD - stdD; flipud(meanD + stdD)], ...
         [0.8 0.8 1], "EdgeColor","none", "FaceAlpha",0.4);
    plot(tt, meanD, "b", "LineWidth", 2);
    xlabel("Time step");
    ylabel("Min dynamic distance (m)");
    title(sprintf("MARL min dynamic distance vs time (N = %d)", Ne));
    legend("MARL \pm std","MARL mean","Location","best");
    grid on;

    results.mindist.t    = tt;
    results.mindist.mean = meanD;
    results.mindist.std  = stdD;

    % ===== 4) ACCURACY VS SAFETY SCATTER =====
    meanErr_ep  = nan(Ne,1);
    minDist_ep  = nan(Ne,1);
    collided_ep = false(Ne,1);

    for k = 1:Ne
        e  = episodes(k).err;
        d  = episodes(k).mindist;  % already cleaned in loader
        c  = episodes(k).coll;
        Lk = episodes(k).len;

        Tacc = min(minSteps, Lk);

        meanErr_ep(k)  = mean(e(1:Tacc), "omitnan");
        minDist_ep(k)  = min(d, [], "omitnan");
        collided_ep(k) = any(c ~= 0);
    end

    figure; hold on;
    idxColl = collided_ep;
    idxSafe = ~collided_ep;

    scatter(minDist_ep(idxSafe),  meanErr_ep(idxSafe), 40, "g", "filled");
    scatter(minDist_ep(idxColl), meanErr_ep(idxColl), 40, "r", "filled");
    xlabel("Min dyn distance over episode (m)");
    ylabel(sprintf("Mean formation error (first %d steps) (m)", minSteps));
    title("MARL accuracy vs safety");
    legend("No collision","Collision","Location","best");
    grid on;

    results.scatter.mean_err = meanErr_ep;
    results.scatter.min_dist = minDist_ep;
    results.scatter.collided = collided_ep;

    % ===== 5) TIME-ALIGNED FORMATION ERROR VS TIME =====
    E = nan(T, Ne);
    for k = 1:Ne
        e = episodes(k).err;
        if numel(e) >= T
            E(:,k) = e(1:T);
        else
            E(1:numel(e),k) = e(:);
        end
    end
    meanE = mean(E, 2, "omitnan");
    stdE  = std(E, 0, 2, "omitnan");

    figure; hold on;
    fill([tt; flipud(tt)], ...
         [meanE - stdE; flipud(meanE + stdE)], ...
         [0.7 0.7 1], "EdgeColor","none", "FaceAlpha",0.3);
    plot(tt, meanE, "b", "LineWidth", 2);
    xlabel("Time step");
    ylabel("Mean formation error (m)");
    title(sprintf("MARL formation tracking (N = %d episodes)", Ne));
    legend("MARL \pm std","MARL mean","Location","best");
    grid on;

    results.formErr.t    = tt;
    results.formErr.mean = meanE;
    results.formErr.std  = stdE;

end


% =====================================================================
%  HELPER: load all MARL episodes into a struct array
% =====================================================================
function episodes = load_marl_episodes()

    files = dir("*.csv");
    episodes = struct('len',{}, 'err',{}, 'mindist',{}, 'coll',{});

    for k = 1:numel(files)
        fname = fullfile(files(k).folder, files(k).name);

        % Skip obvious summaries, if any
        if contains(files(k).name, "summary", "IgnoreCase", true)
            continue;
        end

        T = readtable(fname);
        vn = T.Properties.VariableNames;

        % Remove NaN step row if present
        if ismember("step", vn)
            valid = ~isnan(T.step);
            T = T(valid,:);
        end

        vn = T.Properties.VariableNames;

        % Formation error column (depending on how you logged MARL)
        if ismember("followers_mean_pos_err_m", vn)
            e = T.followers_mean_pos_err_m;
        elseif ismember("mean_form_error_m", vn)
            e = T.mean_form_error_m;
        elseif ismember("mean_form_error", vn)
            e = T.mean_form_error;
        else
            warning("Skipping %s (no formation error column).", files(k).name);
            continue;
        end

        % Min dynamic distance
        if ismember("min_dyn_distance_m", vn)
            d = T.min_dyn_distance_m;
        elseif ismember("min_dyn_distance", vn)
            d = T.min_dyn_distance;
        else
            d = nan(height(T),1);
        end
        d(isinf(d)) = NaN;   % ignore Inf values in later stats/plots

        % Collision flag
        if ismember("collision", vn)
            c = T.collision;
        elseif ismember("collision_flag", vn)
            c = T.collision_flag;
        else
            c = zeros(height(T),1);
        end

        ep.len     = height(T);
        ep.err     = e(:);
        ep.mindist = d(:);
        ep.coll    = c(:);

        episodes(end+1) = ep; %#ok<AGROW>
    end
end
