clc
close all

%% fitting di modelli polinomiali ai minimi quadrati del carico elettrico in funzione di temp_media e ora del giorno

load('preprocessed_data.mat');

%% scatter dei dati
figure(14)
scatter3(w_avg_train, data_train.TIMESTAMP, data_train.LOAD, 'b'); xlabel('temp media'), ylabel('ora del giorno'), zlabel('LOAD'), grid on;
title('scatter identificazione');

figure(15)
scatter3(w_avg_val, data_val.TIMESTAMP, data_val.LOAD, '*', 'r');
title('scatter validazione');

%% modello quadratico +

q_quadratico = 5;
phi_quadratico = [ones(n, 1), w_avg_train, data_train.TIMESTAMP, w_avg_train.^2, data_train.TIMESTAMP.^2];

[thetaLS_quadratico, std_error_quadratico, var_cap_quadratico, var_thetaLS_quadratico] = lscov(phi_quadratico, data_train.LOAD);

load_cap_quadratico = phi_quadratico * thetaLS_quadratico;
epsilon_quadratico = data_train.LOAD - load_cap_quadratico;
SSR_quadratico = epsilon_quadratico' * epsilon_quadratico;

%% surfacing modello quadratico

x = linspace(min(w_avg_train), max(w_avg_train), 50); 
y = linspace(0, 24, 100);

[X, Y] = meshgrid(x, y);
x_vec = X(:);
y_vec = Y(:);
phi_vec = [ones(size(x_vec, 1), 1), x_vec, y_vec, x_vec.^2, y_vec.^2];
z_cap = phi_vec * thetaLS_quadratico;

Z = reshape(z_cap, size(X));
figure(16)
mesh(X, Y, Z), xlabel('temp media'), ylabel('ora del giorno'), zlabel('carico elettrico'), grid on;
hold on
scatter3(w_avg_train, data_train.TIMESTAMP, data_train.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.1);
title('superficie con modello quadratico')
%% modello cubico

q_cubico = 7;

phi_cubico = [ones(n, 1), w_avg_train, data_train.TIMESTAMP, w_avg_train.^2, data_train.TIMESTAMP.^2, w_avg_train.^3, data_train.TIMESTAMP.^3];

[thetaLS_cubico, std_error_cubico, var_cap_cubico, var_thetaLS_cubico] = lscov(phi_cubico, data_train.LOAD);

load_cap_cubico = phi_cubico * thetaLS_cubico;
epsilon_cubico = data_train.LOAD - load_cap_cubico;
SSR_cubico = epsilon_cubico' * epsilon_cubico;
%% surfacing modello cubico

x = linspace(min(w_avg_train), max(w_avg_train), 50); 
y = linspace(0, 24, 100);

[X, Y] = meshgrid(x, y);
x_vec = X(:);
y_vec = Y(:);
phi_vec = [ones(size(x_vec, 1), 1), x_vec, y_vec, x_vec.^2, y_vec.^2, x_vec.^3, y_vec.^3];
z_cap = phi_vec * thetaLS_cubico;

Z = reshape(z_cap, size(X));
figure(17)
mesh(X, Y, Z), xlabel('temp media'), ylabel('ora del giorno'), zlabel('carico elettrico'), grid on;
hold on
scatter3(w_avg_train, data_train.TIMESTAMP, data_train.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.1);
title('superficie con modello cubico')

%% modello quarto grado

q_quarto = 9;
phi_quarto = [ones(n, 1), w_avg_train, data_train.TIMESTAMP, w_avg_train.^2, data_train.TIMESTAMP.^2, ...
   w_avg_train.^3, data_train.TIMESTAMP.^3, w_avg_train.^4, data_train.TIMESTAMP.^4];

[thetaLS_quarto, std_error_quarto, var_cap_quarto, var_thetaLS_quarto] = lscov(phi_quarto, data_train.LOAD);

load_cap_quarto = phi_quarto * thetaLS_quarto;
epsilon_quarto = data_train.LOAD - load_cap_quarto;
SSR_quarto = epsilon_quarto' * epsilon_quarto;
%% surfacing modello quarto grado

x = linspace(min(w_avg_train), max(w_avg_train), 50); 
y = linspace(0, 24, 100);

[X, Y] = meshgrid(x, y);
x_vec = X(:);
y_vec = Y(:);
phi_vec = [ones(size(x_vec, 1), 1), x_vec, y_vec, x_vec.^2, y_vec.^2, x_vec.^3, y_vec.^3, ...
    x_vec.^4, y_vec.^4];
z_cap = phi_vec * thetaLS_quarto;

Z = reshape(z_cap, size(X));
figure(19)
mesh(X, Y, Z), xlabel('temp media'), ylabel('ora del giorno'), zlabel('carico elettrico'), grid on;
hold on
scatter3(w_avg_train, data_train.TIMESTAMP, data_train.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.1);
title('superficie con modello quarto grado')

%% modello quinto grado

q_quinto = 11;
phi_quinto = [ones(n, 1), w_avg_train, data_train.TIMESTAMP, w_avg_train.^2, ...
    data_train.TIMESTAMP.^2, w_avg_train.^3, data_train.TIMESTAMP.^3, ...
   w_avg_train.^4, data_train.TIMESTAMP.^4, w_avg_train.^5, data_train.TIMESTAMP.^5];

[thetaLS_quinto, std_error_quinto, var_cap_quinto, var_thetaLS_quinto] = lscov(phi_quinto, data_train.LOAD);

load_cap_quinto = phi_quinto * thetaLS_quinto;
epsilon_quinto = data_train.LOAD - load_cap_quinto;
SSR_quinto = epsilon_quinto' * epsilon_quinto;

%% surfacing modello quinto grado

x = linspace(min(w_avg_train), max(w_avg_train), 50); 
y = linspace(0, 24, 100);

[X, Y] = meshgrid(x, y);
x_vec = X(:);
y_vec = Y(:);
phi_vec = [ones(size(x_vec, 1), 1), x_vec, y_vec, x_vec.^2, y_vec.^2, x_vec.^3, ...
    y_vec.^3, x_vec.^4, y_vec.^4, x_vec.^5, y_vec.^5];
z_cap = phi_vec * thetaLS_quinto;

Z = reshape(z_cap, size(X));
figure(20)
mesh(X, Y, Z), xlabel('temp media'), ylabel('ora del giorno'), zlabel('carico elettrico'), grid on;
hold on
scatter3(w_avg_train, data_train.TIMESTAMP, data_train.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.1);
title('superficie con modello quinto grado')

%% modello sesto grado

q_sesto = 13;
phi_sesto = [ones(n, 1), w_avg_train, data_train.TIMESTAMP, w_avg_train.^2, ...
    data_train.TIMESTAMP.^2, w_avg_train.^3, data_train.TIMESTAMP.^3, ...
    w_avg_train.^4, data_train.TIMESTAMP.^4, w_avg_train.^5, data_train.TIMESTAMP.^5, ...
    w_avg_train.^6, data_train.TIMESTAMP.^6];

thetaLS_sesto= lscov(phi_sesto, data_train.LOAD);

load_cap_sesto = phi_sesto * thetaLS_sesto;
epsilon_sesto = data_train.LOAD - load_cap_sesto;
SSR_sesto = epsilon_sesto' * epsilon_sesto;
%% cross-validazione

phi_V_quadratico = [ones(n_v, 1), w_avg_val, data_val.TIMESTAMP, w_avg_val.^2, ...
    data_val.TIMESTAMP.^2];
load_cap_V_quadratico = phi_V_quadratico * thetaLS_quadratico;
epsilon_V_quadratico = data_val.LOAD - load_cap_V_quadratico;
SSR_V_quadratico = epsilon_V_quadratico' * epsilon_V_quadratico;

phi_V_cubico = [ones(n_v, 1), w_avg_val, data_val.TIMESTAMP, w_avg_val.^2, data_val.TIMESTAMP.^2, ...
   w_avg_val.^3, data_val.TIMESTAMP.^3];
load_cap_V_cubico = phi_V_cubico * thetaLS_cubico;
epsilon_V_cubico = data_val.LOAD - load_cap_V_cubico;
SSR_V_cubico = epsilon_V_cubico' * epsilon_V_cubico;

phi_V_quarto = [ones(n_v, 1), w_avg_val, data_val.TIMESTAMP, w_avg_val.^2, data_val.TIMESTAMP.^2, ...
    w_avg_val.^3, data_val.TIMESTAMP.^3, w_avg_val.^4, data_val.TIMESTAMP.^4];
load_cap_V_quarto = phi_V_quarto * thetaLS_quarto;
epsilon_V_quarto = data_val.LOAD - load_cap_V_quarto;
SSR_V_quarto = epsilon_V_quarto' * epsilon_V_quarto;

phi_V_quinto = [ones(n_v, 1), w_avg_val, data_val.TIMESTAMP, w_avg_val.^2, data_val.TIMESTAMP.^2, ...
    w_avg_val.^3, data_val.TIMESTAMP.^3, w_avg_val.^4, data_val.TIMESTAMP.^4, ...
    w_avg_val.^5, data_val.TIMESTAMP.^5];
load_cap_V_quinto = phi_V_quinto * thetaLS_quinto;
epsilon_V_quinto = data_val.LOAD - load_cap_V_quinto;
SSR_V_quinto = epsilon_V_quinto' * epsilon_V_quinto;

phi_V_sesto = [ones(n_v, 1), w_avg_val, data_val.TIMESTAMP, w_avg_val.^2, data_val.TIMESTAMP.^2, ...
   w_avg_val.^3, data_val.TIMESTAMP.^3, w_avg_val.^4, data_val.TIMESTAMP.^4, ...
   w_avg_val.^5, data_val.TIMESTAMP.^5, w_avg_val.^6, data_val.TIMESTAMP.^6];
load_cap_V_sesto = phi_V_sesto * thetaLS_sesto;
epsilon_V_sesto = data_val.LOAD - load_cap_V_sesto;
SSR_V_sesto = epsilon_V_sesto' * epsilon_V_sesto;

SSRs = [SSR_V_quadratico, SSR_V_cubico, SSR_V_quarto, SSR_V_quinto, SSR_V_sesto];
nomi = {'Quadratico', 'Cubico', 'Quarto Grado', 'Quinto Grado', 'Sesto grado'};

[min_SSR, idx] = min(SSRs);

fprintf('\nRISULTATO CROSS-VALIDAZIONE:\n')
fprintf('Il modello migliore è: %s (SSR: %.2e)\n', nomi{idx}, min_SSR);


%% confronto complessità vs RMSE

RMSE_quadratico_train = sqrt(SSR_quadratico / n);
RMSE_quadratico_val = sqrt(SSR_V_quadratico / n_v);
RMSE_cubico_train = sqrt(SSR_cubico / n);
RMSE_cubico_val = sqrt(SSR_V_cubico / n_v);
RMSE_quarto_train = sqrt(SSR_quarto / n);
RMSE_quarto_val = sqrt(SSR_V_quarto / n_v);
RMSE_quinto_train = sqrt(SSR_quinto / n);
RMSE_quinto_val = sqrt(SSR_V_quinto / n_v);
RMSE_sesto_train = sqrt(SSR_sesto/ n);
RMSE_sesto_val = sqrt(SSR_V_sesto / n_v);

rmse_train_vals = [RMSE_quadratico_train, RMSE_cubico_train, RMSE_quarto_train, RMSE_quinto_train, RMSE_sesto_train ];
rmse_val_vals  = [RMSE_quadratico_val, RMSE_cubico_val, RMSE_quarto_val, RMSE_quinto_val, RMSE_sesto_val];
x_axis = 1:5; 

figure(21);
plot(x_axis, rmse_train_vals, '-o', 'MarkerFaceColor', 'b');
hold on
plot(x_axis, rmse_val_vals, '-s', 'MarkerFaceColor', 'r');

grid on
xticks(x_axis)
xticklabels({'Quadratico (q=5)', 'Cubico (q=7)', 'Quarto(q=9)', ...
    'Quinto(q=11)', 'Sesto(q=13)'})
ylabel('RMSE')
xlabel('Complessità del Modello (Grado Polinomio)')
title('complessità vs RMSE')
legend('Train RMSE', 'Validazione RMSE', 'Location', 'northeast')

%% prestazioni sul test set

phi_finale = [ones(n + n_v, 1), w_avg_trainval, data_trainval.TIMESTAMP, w_avg_trainval.^2, ... 
    data_trainval.TIMESTAMP.^2, w_avg_trainval.^3, data_trainval.TIMESTAMP.^3, ...
    w_avg_trainval.^4, data_trainval.TIMESTAMP.^4, ...
    w_avg_trainval.^5, data_trainval.TIMESTAMP.^5];

[thetaLS_finale, std_error_finale, var_cap_finale, var_thetaLS_finale] = lscov(phi_finale, data_trainval.LOAD);

load_cap_finale = phi_finale * thetaLS_finale;
epsilon_finale = data_trainval.LOAD - load_cap_finale;
SSR_finale = epsilon_finale' * epsilon_finale;

phi_T_finale = [ones(n_t, 1), w_avg_test, data_test.TIMESTAMP, w_avg_test.^2, data_test.TIMESTAMP.^2, ...
    w_avg_test.^3, data_test.TIMESTAMP.^3, w_avg_test.^4, data_test.TIMESTAMP.^4, ...
    w_avg_test.^5, data_test.TIMESTAMP.^5];

load_cap_T_finale = phi_T_finale * thetaLS_finale;
epsilon_T_finale = data_test.LOAD - load_cap_T_finale;
SSR_T_finale = epsilon_T_finale' * epsilon_T_finale;

RMSE_finale_train = sqrt(SSR_finale / (n + n_v));
RMSE_finale_test = sqrt(SSR_T_finale / n_t);
mape_poly = mean(abs((data_test.LOAD - load_cap_T_finale) ./ data_test.LOAD)) * 100;

SSR_res_poli = sum(epsilon_T_finale.^2);
SSR_tot_poli = sum((data_test.LOAD - (mean(data_test.LOAD))).^2);
R2_poli = 1 - (SSR_res_poli / SSR_tot_poli);

fprintf('MAPE Polinomio 5° Grado: %.2f%%\n', mape_poly);

%% surfacing finale


x = linspace(min(w_avg_trainval), max(w_avg_trainval), 50); 
y = linspace(0, 24, 100);

[X, Y] = meshgrid(x, y);
x_vec = X(:);
y_vec = Y(:);
phi_vec = [ones(size(x_vec, 1), 1), x_vec, y_vec, x_vec.^2, y_vec.^2, x_vec.^3, y_vec.^3 ...
   x_vec.^4, y_vec.^4, x_vec.^5, y_vec.^5];
z_cap = phi_vec * thetaLS_finale;

Z = reshape(z_cap, size(X));
figure(22)
mesh(X, Y, Z), grid on;
hold on
scatter3(w_avg_trainval, data_trainval.TIMESTAMP, data_trainval.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.1);
xlabel('Temperatura Media (°F)')
ylabel('Ora del Giorno')
zlabel('Carico Elettrico (MW)')
title('Superficie con modello quinto grado')
%% goodness of fit

figure(23)
scatter(load_cap_T_finale, data_test.LOAD, 20, 'filled', 'MarkerFaceAlpha', 0.4);
hold on;

% bisettrice
limiti = [min([load_cap_T_finale, data_test.LOAD]), max([load_cap_T_finale, data_test.LOAD])];
plot(limiti, limiti, 'r', 'LineWidth', 2);

grid on;
xlabel('Carico Predetto (MW)');
ylabel('Carico Reale (MW)');
title('Goodness of Fit - Polinomio 5° grado');
legend('Previsioni 5° grado', 'Bisettrice', 'Location', 'NorthWest');

%% polinomio quinto grado + Fourier

omega = 2 * pi / 24;

phi_temp_train = [ones(n + n_v, 1), w_avg_trainval, w_avg_trainval.^2, ... 
                  w_avg_trainval.^3, w_avg_trainval.^4, w_avg_trainval.^5];

phi_temp_test = [ones(n_t, 1), w_avg_test, w_avg_test.^2, ... 
                 w_avg_test.^3, w_avg_test.^4, w_avg_test.^5];

phi_fourier_train = [sin(omega * data_trainval.TIMESTAMP), cos(omega * data_trainval.TIMESTAMP), ... 
                     sin(2*omega * data_trainval.TIMESTAMP), cos(2*omega * data_trainval.TIMESTAMP), ...
                     sin(3*omega * data_trainval.TIMESTAMP), cos(3*omega * data_trainval.TIMESTAMP), ...
                     sin(4*omega * data_trainval.TIMESTAMP), cos(4*omega * data_trainval.TIMESTAMP)];

phi_fourier_test = [sin(omega * data_test.TIMESTAMP), cos(omega * data_test.TIMESTAMP), ... 
                    sin(2*omega * data_test.TIMESTAMP), cos(2*omega * data_test.TIMESTAMP), ...
                    sin(3*omega * data_test.TIMESTAMP), cos(3*omega * data_test.TIMESTAMP), ...
                    sin(4*omega * data_test.TIMESTAMP), cos(4*omega * data_test.TIMESTAMP)];

phi_finale_arm = [phi_temp_train, phi_fourier_train];
phi_T_finale_arm = [phi_temp_test, phi_fourier_test];

[thetaLS_arm, std_error_arm] = lscov(phi_finale_arm, data_trainval.LOAD);

load_cap_T_arm = phi_T_finale_arm * thetaLS_arm;
epsilon_T_arm = data_test.LOAD - load_cap_T_arm;

RMSE_arm_test = sqrt(mean(epsilon_T_arm.^2));
mape_arm = mean(abs(epsilon_T_arm ./ data_test.LOAD)) * 100;
R2_polif = 1 - (sum(epsilon_T_arm.^2) / sum((data_test.LOAD - mean(data_test.LOAD)).^2));

fprintf('--- Confronto Modelli ---\n');
fprintf('RMSE Polinomio: %.2f%%\n', RMSE_finale_test);
fprintf('RMSE Polinomio + Fourier: %.2f%%\n', RMSE_arm_test);
fprintf('MAPE Polinomio: %.2f%%\n', mape_poly);
fprintf('MAPE Polinomio + Fourier: %.2f%%\n', mape_arm);
fprintf('R2 Polinomio: %.4f\n', R2_poli);
fprintf('R2 Polinomio + Fourier: %.4f\n', R2_polif);


%% Surfacing polinomio quinto grado + Fourier 

x_surf = linspace(min(w_avg_trainval), max(w_avg_trainval), 50); 
y_surf = linspace(0, 24, 100); 
[X_surf, Y_surf] = meshgrid(x_surf, y_surf);

T_period = 24;
omega_surf = 2 * pi / T_period;

phi_grid_temp = [ones(numel(X_surf), 1), X_surf(:), X_surf(:).^2, ...
                 X_surf(:).^3, X_surf(:).^4, X_surf(:).^5];

phi_grid_fourier = [sin(omega_surf * Y_surf(:)),   cos(omega_surf * Y_surf(:)), ...
                    sin(2 * omega_surf * Y_surf(:)), cos(2 * omega_surf * Y_surf(:)), ...
                    sin(3 * omega_surf * Y_surf(:)), cos(3 * omega_surf * Y_surf(:)), ...
                    sin(4 * omega_surf * Y_surf(:)), cos(4 * omega_surf * Y_surf(:))];

phi_grid_tot = [phi_grid_temp, phi_grid_fourier];

z_cap_poly = phi_grid_tot * thetaLS_arm;
Z_surf = reshape(z_cap_poly, size(X_surf));

figure(24) 
mesh(X_surf, Y_surf, Z_surf, 'FaceAlpha', 0.8)
hold on
scatter3(w_avg_trainval, data_trainval.TIMESTAMP, data_trainval.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.1);
xlabel('Temperatura Media (°F)')
ylabel('Ora del Giorno')
zlabel('Carico (MW)')
title('Superficie Polinomio 5° grado + Fourier')
grid on
view(-135, 30);
%% goodness of fit

figure(25)
scatter(load_cap_T_arm, data_test.LOAD, 20, 'filled', 'MarkerFaceAlpha', 0.4);
hold on;

% bisettrice
limiti = [min([load_cap_T_arm, data_test.LOAD]), max([load_cap_T_arm, data_test.LOAD])];
plot(limiti, limiti, 'r', 'LineWidth', 2);

grid on;
xlabel('Carico Predetto (MW)');
ylabel('Carico Reale (MW)');
title('Goodness of Fit - Polinomio 5° grado + Fourier');
legend('Previsioni', 'Bisettrice', 'Location', 'NorthWest');

%% subplot 5° grado vs 5° grado + Fourier

figure(26); 
set(gcf, 'Name', 'Confronto Superfici di Regressione', 'NumberTitle', 'off', 'Position', [100, 100, 1300, 600]);

% Subplot 1: Polinomio 5° Grado 
subplot(1, 2, 1);
mesh(X, Y, Z); hold on;
scatter3(w_avg_trainval, data_trainval.TIMESTAMP, data_trainval.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.05);
xlabel('Temperatura Media (°F)'); ylabel('Ora del giorno'); zlabel('Carico elettrico (MW)');
title(sprintf('Polinomio 5° grado\n RMSE: %.2f - MAPE: %.2f%% - R^2: %.4f', RMSE_finale_test, mape_poly, R2_poli));
grid on; view(-135, 30);

% Subplot 2: Polinomio 5° + Fourier 
subplot(1, 2, 2);
mesh(X_surf, Y_surf, Z_surf); hold on;
scatter3(w_avg_trainval, data_trainval.TIMESTAMP, data_trainval.LOAD, 5, 'b', 'filled', 'MarkerFaceAlpha', 0.05);
xlabel('Temperatura Media (°F)'); ylabel('Ora del giorno'); zlabel('Carico elettrico (MW)');
title(sprintf('Polinomio 5° grado + Fourier\n RMSE: %.2f - MAPE: %.2f%% - R^2: %.4f', RMSE_arm_test, mape_arm, R2_polif));
grid on; view(-135, 30);

sgtitle('Confronto superfici');

figure(27);
set(gcf, 'Name', 'Confronto Goodness of Fit', 'NumberTitle', 'off', 'Position', [150, 150, 1300, 600]);

% Subplot 1: GoF Polinomio 5°
subplot(1, 2, 1);
scatter(load_cap_T_finale, data_test.LOAD, 20, 'filled', 'MarkerFaceAlpha', 0.3); hold on;
limiti1 = [min([load_cap_T_finale; data_test.LOAD]), max([load_cap_T_finale; data_test.LOAD])];
plot(limiti1, limiti1, 'r', 'LineWidth', 2);
grid on; xlabel('Carico Predetto (MW)'); ylabel('Carico Reale (MW)');
title('GoF: Polinomio 5° grado'); 

% Subplot 2: GoF Polinomio 5° + Fourier 
subplot(1, 2, 2);
scatter(load_cap_T_arm, data_test.LOAD, 20, 'filled', 'MarkerFaceAlpha', 0.3); hold on;
limiti2 = [min([load_cap_T_arm; data_test.LOAD]), max([load_cap_T_arm; data_test.LOAD])];
plot(limiti2, limiti2, 'r', 'LineWidth', 2);
grid on; xlabel('Carico Predetto (MW)'); ylabel('Carico Reale (MW)');
title('GoF: Polinomio 5° grado + Fourier'); 


sgtitle('Confronto GOF');
