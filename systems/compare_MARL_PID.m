function compare_MARL_PID(M, P)

figure('Position',[100 100 1600 1000]);

%% 1. Formation error vs time
subplot(2,3,1); hold on;
plot(M.formErr.t, M.formErr.mean, 'b', 'LineWidth', 2);
plot(P.formErr.t, P.formErr.mean, 'r', 'LineWidth', 2);

fill([M.formErr.t; flipud(M.formErr.t)], ...
     [M.formErr.mean-M.formErr.std; flipud(M.formErr.mean+M.formErr.std)], ...
     [0.8 0.85 1], 'EdgeColor','none','FaceAlpha',0.3);

fill([P.formErr.t; flipud(P.formErr.t)], ...
     [P.formErr.mean-P.formErr.std; flipud(P.formErr.mean+P.formErr.std)], ...
     [1 0.8 0.8], 'EdgeColor','none','FaceAlpha',0.3);

title("Formation error vs time");
xlabel("Time step"); ylabel("Error (m)");
legend("MARL","PID","Location","northwest");
grid on;

%% 2. Min dynamic distance vs time
subplot(2,3,2); hold on;
plot(M.mindist.t, M.mindist.mean, 'b', 'LineWidth', 2);
plot(P.mindist.t, P.mindist.mean, 'r', 'LineWidth', 2);

fill([M.mindist.t; flipud(M.mindist.t)], ...
     [M.mindist.mean-M.mindist.std; flipud(M.mindist.mean+M.mindist.std)], ...
     [0.8 0.85 1], 'EdgeColor','none','FaceAlpha',0.3);

fill([P.mindist.t; flipud(P.mindist.t)], ...
     [P.mindist.mean-P.mindist.std; flipud(P.mindist.mean+P.mindist.std)], ...
     [1 0.8 0.8], 'EdgeColor','none','FaceAlpha',0.3);

title("Min dynamic distance vs time");
xlabel("Time step"); ylabel("Distance (m)");
legend("MARL","PID","Location","southwest");
grid on;

%% 3. Cumulative collision probability
subplot(2,3,3);
plot(M.cumCollision.t, M.cumCollision.prob, 'b', 'LineWidth', 2); hold on;
plot(P.cumCollision.t, P.cumCollision.prob, 'r', 'LineWidth', 2);
title("Cumulative collision probability");
xlabel("Time step"); ylabel("P(collision by t)");
legend("MARL","PID","Location","northwest");
grid on;

%% 4. Survival curves
subplot(2,3,4);
plot(M.survival.t, M.survival.prob, 'b', 'LineWidth', 2); hold on;
plot(P.survival.t, P.survival.prob, 'r', 'LineWidth', 2);
title("Survival curve");
xlabel("Time step"); ylabel("P(episode still running)");
legend("MARL","PID","Location","southwest");
grid on;

%% 5. Accuracy vs safety scatter
subplot(2,3,5); hold on;

scatter(M.scatter.min_dist(~M.scatter.collided), ...
        M.scatter.mean_err(~M.scatter.collided), 40, 'b', 'filled');
scatter(M.scatter.min_dist(M.scatter.collided), ...
        M.scatter.mean_err(M.scatter.collided), 40, 'c', 'filled');

scatter(P.scatter.min_dist(~P.scatter.collided), ...
        P.scatter.mean_err(~P.scatter.collided), 40, 'r', 'filled');
scatter(P.scatter.min_dist(P.scatter.collided), ...
        P.scatter.mean_err(P.scatter.collided), 40, 'm', 'filled');

xlabel("Min dynamic dist over episode (m)");
ylabel("Mean formation error (first 300 steps, m)");
title("Accuracy vs safety");
legend("MARL (safe)","MARL (collision)", ...
       "PID (safe)","PID (collision)","Location","best");
grid on;

sgtitle("MARL vs PID Comparative Analysis","FontSize",18,"FontWeight","bold");
end
