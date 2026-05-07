clc
close all

load("preprocessed_data.mat")

%% Stepwise Regression 

omega = 2 * pi / 24;

W_tr = w_avg_trainval;
T_tr = data_trainval.TIMESTAMP;

X_temp_tr = [W_tr, W_tr.^2, W_tr.^3, W_tr.^4, W_tr.^5];

X_time_tr = [];
for k = 1:5
    X_time_tr = [X_time_tr, sin(k*omega*T_tr), cos(k*omega*T_tr)];
end

X_inter_tr = [W_tr.*sin(omega*T_tr), W_tr.*cos(omega*T_tr), ...
              (W_tr.^2).*sin(omega*T_tr), (W_tr.^2).*cos(omega*T_tr)];

X_full_tr = [X_temp_tr, X_time_tr, X_inter_tr];

[~, ~, ~, in, ~] = stepwisefit(X_full_tr, data_trainval.LOAD, 'penter', 0.01, 'premove', 0.05, 'display', 'off');
X_train_final = [ones(size(X_full_tr,1),1), X_full_tr(:, in)];
theta = lscov(X_train_final, data_trainval.LOAD);

W_ts = w_avg_test;
T_ts = data_test.TIMESTAMP;

X_temp_ts = [W_ts, W_ts.^2, W_ts.^3, W_ts.^4, W_ts.^5];
X_time_ts = [];
for k = 1:5
    X_time_ts = [X_time_ts, sin(k*omega*T_ts), cos(k*omega*T_ts)];
end
X_inter_ts = [W_ts.*sin(omega*T_ts), W_ts.*cos(omega*T_ts), ...
              (W_ts.^2).*sin(omega*T_ts), (W_ts.^2).*cos(omega*T_ts)];

X_full_ts = [X_temp_ts, X_time_ts, X_inter_ts];
X_test_final = [ones(size(X_full_ts,1),1), X_full_ts(:, in)];

y_pred_step = X_test_final * theta;
err_step = data_test.LOAD - y_pred_step;

rmse_step = sqrt(mean(err_step.^2));
mape_step = mean(abs(err_step./data_test.LOAD))*100;
r2_step = 1 - sum(err_step.^2)/sum((data_test.LOAD - mean(data_test.LOAD)).^2);

fprintf('--- PERFORMANCE STEPWISE (Fourier 6 + Temp 5) ---\n');
fprintf('RMSE: %.2f MW | MAPE: %.2f%% | R2: %.4f\n', rmse_step, mape_step, r2_step);

%% surfacing stepwise + Fourier

x_range = linspace(min(W_tr), max(W_tr), 60);
y_range = linspace(0, 24, 100); 
[Xm, Ym] = meshgrid(x_range, y_range);
Xf = Xm(:); Yf = Ym(:);

X_temp_m = [Xf, Xf.^2, Xf.^3, Xf.^4, Xf.^5];
X_time_m = [];
for k = 1:6
    X_time_m = [X_time_m, sin(k*omega*Yf), cos(k*omega*Yf)];
end
X_inter_m = [Xf.*sin(omega*Yf), Xf.*cos(omega*Yf), ...
             (Xf.^2).*sin(omega*Yf), (Xf.^2).*cos(omega*Yf)];

X_grid_full = [X_temp_m, X_time_m, X_inter_m];
X_grid_final = [ones(length(Xf),1), X_grid_full(:, in)];
Zm = reshape(X_grid_final * theta, size(Xm));

figure;
mesh(Xm, Ym, Zm, 'FaceAlpha', 0.7);
hold on;
scatter3(W_tr, T_tr, data_trainval.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.1);
title(['Superficie Stepwise (R^2: ', num2str(r2_step,3), ')']);
xlabel('Temperatura (°F)'); ylabel('Ora del giorno'); zlabel('Carico (MW)');
grid on; 

%% goodness of fit

figure(28)
scatter(y_pred_step, data_test.LOAD, 20, 'filled', 'MarkerFaceAlpha', 0.4);
hold on;

% bisettrice
limiti = [min([y_pred_step, data_test.LOAD]), max([y_pred_step, data_test.LOAD])];
plot(limiti, limiti, 'r', 'LineWidth', 2);

grid on;
xlabel('Carico Predetto (MW)');
ylabel('Carico Reale (MW)');
title('Goodness of Fit - stepwise + Fourier');
legend('Previsioni', 'Bisettrice', 'Location', 'NorthWest');

%% subplot polinomio 5° + Fourier vs stepwise + Fourier

figure(29);
set(gcf, 'Position', [100, 100, 1200, 500]); 

% Subplot 1: Polinomio 5° grado + Fourier 
ax1 = subplot(1, 2, 1);
mesh(X_surf, Y_surf, Z_surf)
hold on
scatter3(w_avg_trainval, data_trainval.TIMESTAMP, data_trainval.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.05);
xlabel('Temperatura Media (°F)'); ylabel('Ora del giorno'); zlabel('Carico (MW)');
title(sprintf('Polinomio 5° grado + Fourier\nRMSE: %.2f - MAPE: %.2f%% - R^2: %.4f', RMSE_arm_test, mape_arm, R2_polif));
grid on; view([-135, 30]);

% Subplot 2: Stepwise + Fourier 
ax2 = subplot(1, 2, 2);
mesh(Xm, Ym, Zm)
hold on
scatter3(w_avg_trainval, data_trainval.TIMESTAMP, data_trainval.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.05);
xlabel('Temperatura Media (°F)'); ylabel('Ora del giorno'); zlabel('Carico (MW)');
title(sprintf('Stepwise + Fourier\nRMSE: %.2f - MAPE: %.2f%% - R^2: %.4f', rmse_step, mape_step, r2_step));
grid on; view([-135, 30]);

z_min = min([Z_surf(:); Zm(:)]);
z_max = max([Z_surf(:); Zm(:)]);
set([ax1, ax2], 'ZLim', [z_min z_max]);

h = linkprop([ax1, ax2], {'XLim', 'YLim', 'ZLim'});
setappdata(gcf, 'StoreTheLink', h);
sgtitle('Confronto superfici'); 

figure(30); % Nuova figura per GoF
set(gcf, 'Position', [150, 150, 1200, 500]);

% Subplot 1: GoF Stepwise Puro
subplot(1, 2, 1);
scatter(load_cap_T_arm, data_test.LOAD, 20, 'filled', 'MarkerFaceAlpha', 0.3); 
hold on;
plot(limiti, limiti, 'r', 'LineWidth', 2);
grid on; xlabel('Carico Predetto (MW)'); ylabel('Carico Reale (MW)');
title('GoF: Polinomio 5° grado +  Fourier');

% Subplot 2: GoF Stepwise + Fourier 
subplot(1, 2, 2);
scatter(y_pred_step, data_test.LOAD, 20, 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
plot(limiti, limiti, 'r', 'LineWidth', 2);
grid on; xlabel('Carico Predetto (MW)'); ylabel('Carico Reale (MW)');
title('GoF: Stepwise + Fourier');

sgtitle('Confronto GOF');

