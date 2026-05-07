clc
close all

load('preprocessed_data.mat');

%% scatter carico-ora del giorno

scatter(data_trainval.TIMESTAMP, data_trainval.LOAD);
%% fitting modello con tutte le 25 temperature

temp_matrix_trainval = temp_matrix(1:n+n_v, :);
phi_25_train = [ones(n + n_v, 1), temp_matrix_trainval];

[thetaLS_25, std_error_25, var_cap_25, var_thetaLS_25] = lscov(phi_25_train, data_trainval.LOAD);

% valori altalenanti dei parametri stimati
figure(12)
bar(thetaLS_25(2:end)); 
grid on;
title('Valori dei coefficienti \theta_1...\theta_{25}');

%% prestazioni sul test set

temp_matrix_test = table2array(data_test(:, 3:27));
load_test = data_test.LOAD;

phi_25_test = [ones(height(data_test), 1), temp_matrix_test];

% Predizione del carico
load_cap_25_test = phi_25_test * thetaLS_25;

epsilon_25_test = load_test - load_cap_25_test;


RMSE_25_test = sqrt(mean(epsilon_25_test.^2));
MAPE_25_test = mean(abs(epsilon_25_test ./ load_test)) * 100;

cond_25 = cond(phi_25_train);
scatter(w_avg_test, load_cap_25_test, '.'), grid on;
fprintf('\n--- RISULTATI MODELLO 25 TEMPERATURE ---\n');
fprintf('RMSE sul Test Set: %.4f MW\n', RMSE_25_test);
fprintf('MAPE sul Test Set: %.2f%%\n', MAPE_25_test);
fprintf('Condition Number della matrice: %.2e\n', cond_25);


%% Modello LOAD vs TIMESTAMP (Armonico vs Cubico)
t_train = data_trainval.TIMESTAMP(:);
load_train = data_trainval.LOAD(:);
t_test = data_test.TIMESTAMP(:);
load_test = data_test.LOAD(:);

phi_cubica = [ones(size(t_train)), t_train, t_train.^2, t_train.^3];
theta_cubica = lscov(phi_cubica, load_train);
phi_cubica_test = [ones(size(t_test)), t_test, t_test.^2, t_test.^3];
y_pred_cubica = phi_cubica_test * theta_cubica;

T = 24; 

phi_arm = [ones(size(t_train)), sin(2*pi*t_train/T), cos(2*pi*t_train/T), ...
    sin(4*pi*t_train/24), cos(4*pi*t_train/24)];
theta_arm = lscov(phi_arm, load_train);
phi_arm_test = [ones(size(t_test)), sin(2*pi*t_test/T), cos(2*pi*t_test/T), ...
    sin(4*pi*t_test/24), cos(4*pi*t_test/24)];
y_pred_arm = phi_arm_test * theta_arm;

rmse_cubica = sqrt(mean((load_test - y_pred_cubica).^2));
rmse_arm = sqrt(mean((load_test - y_pred_arm).^2));

fprintf('--- CONFRONTO MODELLI SOLO ORA ---\n');
fprintf('CUBICO   | RMSE: %.2f MW\n', rmse_cubica);
fprintf('ARMONICO | RMSE: %.2f MW\n', rmse_arm);

%% Fitting e Visualizzazione
T_grid = linspace(0, 24, 200)'; % 24 incluso per vedere la chiusura

% Curve per il plot
Phi_grid_cub = [ones(size(T_grid)), T_grid, T_grid.^2, T_grid.^3];
curva_cubica = Phi_grid_cub * theta_cubica;

Phi_grid_arm = [ones(size(T_grid)), sin(2*pi*T_grid/T), cos(2*pi*T_grid/T), sin(4*pi*T_grid/T), cos(4*pi*T_grid/T)];
curva_armonica = Phi_grid_arm * theta_arm;

figure(9)
scatter(t_train, load_train, '.', 'MarkerEdgeColor', [0.8 0.8 0.8]), hold on;
plot(T_grid, curva_cubica, 'r', 'LineWidth', 2)
plot(T_grid, curva_armonica, 'b', 'LineWidth', 2)
grid on;
xlabel('Ora del Giorno (h)')
ylabel('Carico Elettrico (MW)')
legend('Dati training', 'Modello Cubico', 'Modello Armonico (Sin/Cos)')
title('Confronto Fitting: Polinomiale vs Ciclico')

% Verifica chiusura mezzanotte
fprintf('\n--- VERIFICA CONTINUITÀ (h 0 vs h 24) ---\n');
fprintf('Cubico   h0: %.2f | h24: %.2f (Delta: %.2f)\n', curva_cubica(1), curva_cubica(end), abs(curva_cubica(1)-curva_cubica(end)));
fprintf('Armonico h0: %.2f | h24: %.2f (Delta: %.2f)\n', curva_armonica(1), curva_armonica(end), abs(curva_armonica(1)-curva_armonica(end)));




