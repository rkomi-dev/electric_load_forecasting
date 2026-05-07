clc
close all

%% fitting di modelli polinomiali ai minimi quadrati del carico elettrico in funzione di temp_media

load('preprocessed_data.mat');
T_grid = (min(w_avg):0.1:max(w_avg))';

%% modello quadratico

q_quadratico = 3;

phi_quadratico_train = [ones(n, 1), w_avg_train, w_avg_train.^2];
[thetaLS_quadratico, std_error_quadratico, var_cap_quadratico, var_thetaLS_quadratico] = lscov(phi_quadratico_train, data_train.LOAD);

load_cap_quadratico = phi_quadratico_train * thetaLS_quadratico;
epsilon_quadratico = data_train.LOAD - load_cap_quadratico;
SSR_quadratico = epsilon_quadratico' * epsilon_quadratico;

%% fitting modello quadratico

Phi_grid = [ones(length(T_grid), 1) T_grid, T_grid.^2];
curva = Phi_grid*thetaLS_quadratico;

figure(6)
scatter(w_avg_train, data_train.LOAD, '.'), grid on;
hold on
plot(T_grid, curva, 'r', 'LineWidth', 2)
legend('Dati training', 'Modello Quadratico')
title('Fitting: Carico vs Temperatura Media (training)')

figure(7)
scatter(w_avg_val, data_val.LOAD, '.'), grid on;
hold on
plot(T_grid, curva, 'r', 'LineWidth', 2)
legend('Dati test', 'Modello Quadratico')
title('Fitting: Carico vs Temperatura Media (validazione)')

%% modello cubico

q_cubico = 4;

phi_cubico_train = [ones(n, 1), w_avg_train, w_avg_train.^2, w_avg_train.^3];
[thetaLS_cubico, std_error_cubico, var_cap_cubico, var_thetaLS_cubico] = lscov(phi_cubico_train, data_train.LOAD);

load_cap_cubico = phi_cubico_train * thetaLS_cubico;
epsilon_cubico = data_train.LOAD - load_cap_cubico;
SSR_cubico = epsilon_cubico' * epsilon_cubico;
%% fitting modello cubico

Phi_grid = [ones(length(T_grid), 1), T_grid, T_grid.^2, T_grid.^3];
curva = Phi_grid*thetaLS_cubico;

figure(8)
scatter(w_avg_train, data_train.LOAD, '.'), grid on;
hold on
plot(T_grid, curva, 'r', 'LineWidth', 2)
legend('Dati training', 'Modello Cubico')
title('Fitting: Carico vs Temperatura Media (training)')

figure(9)
scatter(w_avg_val, data_val.LOAD, '.'), grid on;
hold on
plot(T_grid, curva, 'r', 'LineWidth', 2)
legend('Dati test', 'Modello Cubico')
title('Fitting: Carico vs Temperatura Media (validazione)')

%% modello quarto grado

q_quarto = 5;

phi_quarto_train = [ones(n, 1), w_avg_train, w_avg_train.^2, w_avg_train.^3, w_avg_train.^4];
[thetaLS_quarto, std_error_quarto, var_cap_quarto, var_thetaLS_quarto] = lscov(phi_quarto_train, data_train.LOAD);

load_cap_quarto = phi_quarto_train * thetaLS_quarto;
epsilon_quarto = data_train.LOAD - load_cap_quarto;
SSR_quarto = epsilon_quarto' * epsilon_quarto;
%% fitting modello quarto grado

Phi_grid = [ones(length(T_grid), 1), T_grid, T_grid.^2, T_grid.^3, T_grid.^4];
curva = Phi_grid*thetaLS_quarto;

figure(10)
scatter(w_avg_train, data_train.LOAD, '.'), grid on;
hold on
plot(T_grid, curva, 'r', 'LineWidth', 2)
legend('Dati training', 'Modello quarto grado')
title('Fitting: Carico vs Temperatura Media (training)')

figure(11)
scatter(w_avg_val, data_val.LOAD, '.'), grid on;
hold on
plot(T_grid, curva, 'r', 'LineWidth', 2)
legend('Dati test', 'Modello quarto grado')
title('Fitting: Carico vs Temperatura Media (validazione)')

%% modello quinto grado

q_quinto = 6;

phi_quinto_train = [ones(n, 1), w_avg_train, w_avg_train.^2, w_avg_train.^3, w_avg_train.^4, w_avg_train.^5];
[thetaLS_quinto, std_error_quinto, var_cap_quinto, var_thetaLS_quinto] = lscov(phi_quinto_train, data_train.LOAD);

load_cap_quinto = phi_quinto_train * thetaLS_quinto;
epsilon_quinto = data_train.LOAD - load_cap_quinto;
SSR_quinto = epsilon_quinto' * epsilon_quinto;
%% test F

alpha = 0.05;

% quadratico vs cubico

f_alpha = finv(1 - alpha, 1, n - q_quadratico);
f = (n - q_quadratico) * ((SSR_quadratico - SSR_cubico ) / SSR_cubico);
fprintf('\nTEST F:\n')
fprintf('\nquadratico vs cubico: \n');
if(f < f_alpha) 
    disp('scelgo modello quadratico')
else
    disp('scelgo modello cubico')
end

% cubico vs quarto grado

f_alpha = finv(1 - alpha, 1, n - q_quarto);
f = (n - q_cubico) * ((SSR_cubico - SSR_quarto ) / SSR_quarto);
fprintf('\nTEST F:\n')
fprintf('\ncubico vs quarto grado: \n');
if(f < f_alpha) 
    disp('scelgo modello cubico')
else
    disp('scelgo modello quarto grado')
end


% quarto vs quinto

f_alpha = finv(1 - alpha, 1, n - q_quinto);
f = (n - q_quarto) * ((SSR_quarto - SSR_quinto ) / SSR_quinto);
fprintf('\nTEST F:\n')
fprintf('\nquarto vs quinto: \n');
if(f < f_alpha) 
    disp('scelgo modello quarto grado')
else
    disp('scelgo modello quinto grado')
end

% quinto vs sesto

f_alpha = finv(1 - alpha, 1, n - q_sesto);
f = (n - q_quinto) * ((SSR_quinto - SSR_sesto ) / SSR_sesto);
fprintf('\nTEST F:\n')
fprintf('\nquinto vs sesto grado: \n');
if(f < f_alpha) 
    disp('scelgo modello quinto grado')
else
    disp('scelgo modello sesto grado')
end
%% cross-validazione

phi_V_quadratico = [ones(n_v, 1), w_avg_val, w_avg_val.^2];
load_cap_V_quadratico = phi_V_quadratico * thetaLS_quadratico;
epsilon_V_quadratico = data_val.LOAD - load_cap_V_quadratico;
SSR_V_quadratico = epsilon_V_quadratico' * epsilon_V_quadratico;

phi_V_cubico = [ones(n_v, 1), w_avg_val, w_avg_val.^2, w_avg_val.^3];
load_cap_V_cubico = phi_V_cubico * thetaLS_cubico;
epsilon_V_cubico = data_val.LOAD - load_cap_V_cubico;
SSR_V_cubico = epsilon_V_cubico' * epsilon_V_cubico;

phi_V_quarto = [ones(n_v, 1), w_avg_val, w_avg_val.^2, w_avg_val.^3, w_avg_val.^4];
load_cap_V_quarto = phi_V_quarto * thetaLS_quarto;
epsilon_V_quarto = data_val.LOAD - load_cap_V_quarto;
SSR_V_quarto = epsilon_V_quarto' * epsilon_V_quarto;

phi_V_quinto = [ones(n_v, 1), w_avg_val, w_avg_val.^2, w_avg_val.^3, w_avg_val.^4, w_avg_val.^5];
load_cap_V_quinto = phi_V_quinto * thetaLS_quinto;
epsilon_V_quinto = data_val.LOAD - load_cap_V_quinto;
SSR_V_quinto = epsilon_V_quinto' * epsilon_V_quinto;

SSRs = [SSR_V_quadratico, SSR_V_cubico, SSR_V_quarto, SSR_V_quinto];
nomi = {'Quadratico', 'Cubico','Quarto Grado', 'Quinto Grado'};

[min_SSR, idx] = min(SSRs);

fprintf('\nRISULTATO CROSS-VALIDAZIONE:\n')
fprintf('Il modello migliore è: %s (SSR: %.4e)\n', nomi{idx}, min_SSR);

%% confronto complessità vs RMSE

RMSE_quadratico_train = sqrt(SSR_quadratico / n);
RMSE_quadratico_val = sqrt(SSR_V_quadratico / n_v);
RMSE_cubico_train = sqrt(SSR_cubico / n);
RMSE_cubico_val = sqrt(SSR_V_cubico / n_v);
RMSE_quarto_train = sqrt(SSR_quarto / n);
RMSE_quarto_val = sqrt(SSR_V_quarto / n_v);
RMSE_quinto_train = sqrt(SSR_quinto / n);
RMSE_quinto_val = sqrt(SSR_V_quinto / n_v);

rmse_train_vals = [RMSE_quadratico_train, RMSE_cubico_train, RMSE_quarto_train, RMSE_quinto_train];
rmse_val_vals  = [RMSE_quadratico_val, RMSE_cubico_val, RMSE_quarto_val, RMSE_quinto_val];
x_axis = 1:4; 

figure(12);
plot(x_axis, rmse_train_vals, '-o', 'MarkerFaceColor', 'b');
hold on
plot(x_axis, rmse_val_vals, '-s', 'MarkerFaceColor', 'r');

grid on
xticks(x_axis)
xticklabels({'Quadratico (q=3)', 'Cubico (q=4)', 'Quarto (q=5)', 'Quinto(q=6)'})
ylabel('RMSE (MW)')
xlabel('Complessità del Modello (Grado Polinomio)')
title('complessità vs RMSE')
legend('Training RMSE', 'Validazione RMSE', 'Location', 'northeast')

%% prestazioni sul test set

phi_finale = [ones((n + n_v), 1), w_avg_trainval, w_avg_trainval.^2, w_avg_trainval.^3, w_avg_trainval.^4];

[thetaLS_finale, std_error_finale, var_cap_finale, var_thetaLS_finale] = lscov(phi_finale, data_trainval.LOAD);

load_cap_finale = phi_finale * thetaLS_finale;
epsilon_finale = data_trainval.LOAD - load_cap_finale;
SSR_finale = epsilon_finale' * epsilon_finale;

phi_T_finale = [ones(n_t, 1), w_avg_test, w_avg_test.^2, w_avg_test.^3, w_avg_test.^4];

load_cap_T_finale = phi_T_finale * thetaLS_finale;
epsilon_T_finale = data_test.LOAD - load_cap_T_finale;
SSR_T_finale = epsilon_T_finale' * epsilon_T_finale;

RMSE_finale_train = sqrt(SSR_finale / (n + n_v));
RMSE_finale_test = sqrt(SSR_T_finale / n_t);
mape_poly = mean(abs((data_test.LOAD - load_cap_T_finale) ./ data_test.LOAD)) * 100;

SSR_res_poli = sum(epsilon_T_finale.^2);
SSR_tot_poli = sum((data_test.LOAD - (mean(data_test.LOAD))).^2);
R2_poli = 1 - (SSR_res_poli / SSR_tot_poli);

fprintf('RMSE Polinomio 4° Grado: %.2f\n', RMSE_finale_test);
fprintf('MAPE Polinomio 4° Grado: %.2f%%\n', mape_poly);
fprintf('R^2 Polinomio 4° Grado: %.4f\n', R2_poli);

%% fitting finale

T_grid_finale = (min(w_avg_trainval):0.1:max(w_avg_trainval))';

Phi_grid = [ones(length(T_grid_finale), 1), T_grid_finale, T_grid_finale.^2, ...
    T_grid_finale.^3, T_grid_finale.^4];
curva = Phi_grid*thetaLS_finale;

figure(10)
scatter(w_avg_trainval, data_trainval.LOAD, '.'), grid on;
hold on
plot(T_grid_finale, curva, 'r', 'LineWidth', 2)
xlabel('Temperatura Media (°F)')
ylabel('Carico elettrico (MW)')
legend('Dati training', 'Modello quarto grado')
title(sprintf('Fitting: Carico vs Temperatura Media\n RMSE: %.2f - MAPE: %.2f%% - R^2: %.4f', ...
    RMSE_finale_test, mape_poly, R2_poli))