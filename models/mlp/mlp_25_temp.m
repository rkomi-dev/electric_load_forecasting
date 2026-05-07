clc
close all

load('preprocessed_data.mat');


%% trasformazione timestamp in periodico e normalizzazione

T = 24;
time_sin_trainval = sin(2 * pi * data_trainval.TIMESTAMP / T);
time_cos_trainval = cos(2 * pi * data_trainval.TIMESTAMP / T);

X_trainval_mlp_25 = [data_trainval{:, 3:27}, time_sin_trainval, time_cos_trainval]; 

% Normalizzazione

[X_trainval_n_25, settings_x_25] = mapminmax(X_trainval_mlp_25'); 
[y_trainval_n_25, settings_y_25] = mapminmax(data_trainval.LOAD');

%% Scelta neuroni tramite K-Fold Standard e regola 1-SD
k = 5; 
N_tot = length(y_trainval_n_25);

% Creazione dei fold 
cv = cvpartition(N_tot, 'KFold', k); 

neuron_range = 1:25; 
rmse_kfold_train_medio = zeros(length(neuron_range), 1);
rmse_kfold_val_medio = zeros(length(neuron_range), 1);
rmse_kfold_val_std = zeros(length(neuron_range), 1);

fprintf('\nInizio K-Fold Standard\n');

for i = 1:length(neuron_range)
    n_hid = neuron_range(i);
    rmse_train_fold = zeros(k, 1);
    rmse_val_fold = zeros(k, 1);
    
    for f = 1:k
        % Indici per il fold corrente
        idx_train = training(cv, f);
        idx_val = test(cv, f);
        
        % Split dei dati
        X_train_fold = X_trainval_n_25(:, idx_train);
        y_train_fold = y_trainval_n_25(idx_train);
        X_val_fold = X_trainval_n_25(:, idx_val);
        y_val_fold = y_trainval_n_25(idx_val);
        
        % Training della rete
        net_fold = fitnet(n_hid, 'trainlm');
        net_fold.trainParam.showWindow = false;
        net_fold = train(net_fold, X_train_fold, y_train_fold);
        
        % Valutazione Training 
        y_tr_pred = mapminmax('reverse', net_fold(X_train_fold), settings_y_25);
        y_tr_real = mapminmax('reverse', y_train_fold, settings_y_25);
        rmse_train_fold(f) = sqrt(mean((y_tr_real - y_tr_pred).^2));
        
        % Valutazione Validazione 
        y_val_pred = mapminmax('reverse', net_fold(X_val_fold), settings_y_25);
        y_val_real = mapminmax('reverse', y_val_fold, settings_y_25);
        rmse_val_fold(f) = sqrt(mean((y_val_real - y_val_pred).^2));
    end
    
    % Medie e Deviazione Standard per il numero di neuroni corrente
    rmse_kfold_train_medio(i) = mean(rmse_train_fold);
    rmse_kfold_val_medio(i) = mean(rmse_val_fold);
    rmse_kfold_val_std(i) = std(rmse_val_fold);
    
    fprintf('Neuroni: %d | RMSE Val Medio: %.2f MW (±%.2f)\n', ...
        n_hid, rmse_kfold_val_medio(i), rmse_kfold_val_std(i));
end

% Regola 1-SD 
[min_rmse, idx_min] = min(rmse_kfold_val_medio);
sd_del_min = rmse_kfold_val_std(idx_min);
soglia_tolleranza = min_rmse + sd_del_min;

% modello più semplice che rientra nella soglia
idx_scelto = find(rmse_kfold_val_medio <= soglia_tolleranza, 1, 'first');
n_hid_ottimo = neuron_range(idx_scelto);

fprintf('\nMinimo RMSE: %.2f MW (a %d neuroni)\n', min_rmse, neuron_range(idx_min));
fprintf('Numero neuroni SCELTO (Regola 1-SD): %d\n', n_hid_ottimo);

%% confronto complessità vs RMSE

figure(21)
plot(neuron_range, rmse_kfold_train_medio, '-s', 'LineWidth', 1.5);
hold on
plot(neuron_range, rmse_kfold_val_medio, '-s', 'LineWidth', 1.5);
hold on
yline(soglia_tolleranza, '--r', 'LineWidth', 2);
plot(n_hid_ottimo, rmse_kfold_val_medio(idx_scelto), 'gs', 'MarkerSize', 12, 'LineWidth', 2);
grid on
xlabel('Numero di Neuroni (Hidden Layer)')
ylabel('RMSE (MW)')
legend('Train RMSE', 'Validazione RMSE', 'Soglia di tolleranza')
title('complessità vs RMSE: MLP')
%% prestazioni sul test set

net_finale_25 = fitnet(19, 'trainlm');
net_finale_25.trainParam.showWindow = false; 

net_finale_25 = train(net_finale_25, X_trainval_n_25, y_trainval_n_25);

time_sin_test = sin(2 * pi * data_test.TIMESTAMP / T);
time_cos_test = cos(2 * pi * data_test.TIMESTAMP / T);
X_test_25 = [data_test{:, 3:27}, time_sin_test, time_cos_test];

X_test_n_25 = mapminmax('apply', X_test_25', settings_x_25);

y_test_pred_n_25 = net_finale_25(X_test_n_25);
y_test_pred_mw_25 = mapminmax('reverse', y_test_pred_n_25, settings_y_25);
y_test_real_mw = data_test.LOAD'; 

errore_25 = y_test_real_mw - y_test_pred_mw_25;
MAPE_mlp_25 = mean(abs(errore_25 ./ y_test_real_mw)) * 100;
RMSE_mlp_25 = sqrt(mean(errore_25.^2));
SSR_res_mlp_25 = sum(errore_25.^2);
SSR_tot_mlp_25 = sum((y_test_real_mw - (mean(y_test_real_mw))).^2);
R2_mlp_25 = 1 - (SSR_res_mlp_25 / SSR_tot_mlp_25);

fprintf('--- PERFORMANCE FINALI SUL TEST SET ---\n');
fprintf('RMSE: %.4f MW\nMAPE: %.2f%%\nR2: %.4f\n', RMSE_mlp_25, MAPE_mlp_25, R2_mlp_25);

%% goodness of fit

figure(23)
scatter(y_test_pred_mw_25, y_test_real_mw, 20, 'filled', 'MarkerFaceAlpha', 0.4);
hold on;

% bisettrice
limiti = [min([y_test_pred_mw_25, y_test_real_mw]), max([y_test_pred_mw_25, y_test_real_mw])];
plot(limiti, limiti, 'r', 'LineWidth', 2);

grid on;
xlabel('Carico Predetto (MW)');
ylabel('Carico Reale (MW)');
title('Goodness of Fit - MLP');
legend('Previsioni MLP', 'Bisettrice', 'Location', 'NorthWest');

%% dispersione dei residui

scatter(y_test_pred_mw_25, errore_25, 10, 'filled', 'MarkerFaceAlpha', 0.2);
hold on;
yline(0, 'r', 'LineWidth', 2); % Linea dello zero (errore nullo)
xlabel('Carico Predetto [MW]');
ylabel('Residuo (Reale - Predetto) [MW]');
title('MLP con 19 neuroni a 25 temperature');
grid on

%% istogramma dei residui

figure;
h1 = histogram(errore_25, 50, 'Normalization', 'pdf', 'FaceColor', [0.4 0.6 0.8]);
hold on;

% curva normale teorica per confronto
x_range = linspace(min(errore_25), max(errore_25), 100);
norm_curve = normpdf(x_range, mean(errore_25), std(errore_25));
plot(x_range, norm_curve, 'r', 'LineWidth', 2);

xlabel('Errore [MW]');
title('MLP con 19 neuroni a 25 temperature');
grid on;
legend('Residui Modello', 'Distribuzione Normale');

%% grafico reale vs predetto 

y_max = max([max(data_test.LOAD), max(y_test_pred_mw_25)]) + 10;

figure;

subplot(1, 2, 1)
scatter(w_avg_test, data_test.LOAD, '.'), grid on;
ylim([0, y_max]);
xlabel('Temperatura Media (°F)')
ylabel('Carico elettrico (MW)')
title('Dati reali')

subplot(1, 2, 2)
scatter(w_avg_test, y_test_pred_mw_25, '.'), grid on;
ylim([0, y_max]);
xlabel('Temperatura Media (°F)')
ylabel('Previsioni Carico elettrico (MW)')
title('Previsioni MLP')

sgtitle('Carico in funzione della temperatura media (Reale vs Predetto)')

%% confronto mlp con 8 neuroni a temp media vs mlp con 19 neuroni a 25 temp


%% subplot gof mlp vs mlp25

figure; 
set(gcf, 'Position', [150, 150, 1200, 500]);

all_values = [y_test_pred_mw_25(:); y_test_pred_mw(:); y_test_real_mw(:)];
min_val = min(all_values);
max_val = max(all_values);
limiti = [min_val, max_val];

subplot(1, 2, 2);
scatter(y_test_pred_mw_25, data_test.LOAD, 20, 'filled', 'MarkerFaceAlpha', 0.3); 
hold on;
plot(limiti, limiti, 'r', 'LineWidth', 2);
grid on; xlabel('Carico Predetto (MW)'); ylabel('Carico Reale (MW)');
title('GoF: MLP con 19 neuroni a 25 temperature');

subplot(1, 2, 1);
scatter(y_test_pred_mw, y_test_real_mw, 20, 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
plot(limiti, limiti, 'r', 'LineWidth', 2);
grid on; xlabel('Carico Predetto (MW)'); ylabel('Carico Reale (MW)');
title('GoF: MLP con 8 neuroni a temperatura media');

sgtitle('Confronto GOF');

%% subplot dispersione dei residui

x_min = min([min(y_test_pred_mw), min(y_test_pred_mw_25)]);
x_max = max([max(y_test_pred_mw), max(y_test_pred_mw_25)]);
x_lims = [x_min, x_max];

y_min = min([min(errore), min(errore_25)]);
y_max = max([max(errore), max(errore_25)]);
y_lims = [y_min, y_max];

figure;
subplot(1,2,2);
scatter(y_test_pred_mw_25, errore_25, 10, 'filled', 'MarkerFaceAlpha', 0.2);
hold on;
yline(0, 'r', 'LineWidth', 2); % Linea dello zero (errore nullo)
xlim(x_lims)
ylim(y_lims)
xlabel('Carico Predetto [MW]');
ylabel('Residuo (Reale - Predetto) [MW]');
title('MLP con 19 neuroni a 25 temperature');
grid on;

subplot(1,2,1);

scatter(y_test_pred_mw, errore, 10, 'filled', 'MarkerFaceAlpha', 0.2);
hold on;
yline(0, 'r', 'LineWidth', 2); % Linea dello zero (errore nullo)
xlim(x_lims)
ylim(y_lims)
xlabel('Carico Predetto [MW]');
ylabel('Residuo (Reale - Predetto) [MW]');
title('MLP con 8 neuroni a temperatura media');
grid on

sgtitle('Confronto dispersione dei residui');

%% subplot istogramma dei residui

f1 = histcounts(errore_25, 50, 'Normalization', 'pdf');
f2 = histcounts(errore, 50, 'Normalization', 'pdf');
y_max_global = max([max(f1), max(f2)]) * 1.1; 
y_lims = [0, y_max_global];

figure;

subplot(1,2,2)
h1 = histogram(errore_25, 50, 'Normalization', 'pdf', 'FaceColor', [0.4 0.6 0.8]);
hold on;

% curva normale teorica per confronto
x_range = linspace(min(errore_25), max(errore_25), 100);
norm_curve = normpdf(x_range, mean(errore_25), std(errore_25));
plot(x_range, norm_curve, 'r', 'LineWidth', 2);

xlabel('Errore [MW]');
title('MLP con 19 neuroni a 25 temperature');
grid on;
ylim(y_lims);
legend('Residui Modello', 'Distribuzione Normale');

subplot(1,2,1);

h2 = histogram(errore, 50, 'Normalization', 'pdf', 'FaceColor', [0.4 0.6 0.8]);
hold on;

% curva normale teorica per confronto
x_range = linspace(min(errore), max(errore), 100);
norm_curve = normpdf(x_range, mean(errore), std(errore));
plot(x_range, norm_curve, 'r', 'LineWidth', 2);

xlabel('Errore [MW]');
title('MLP con 8 neuroni a temperatura media');
grid on;
ylim(y_lims);
legend('Residui Modello', 'Distribuzione Normale');

sgtitle('Confronto istogrammi dei residui');

%% subplot predetto mlp vs predetto mlp25

y_max = max([max(y_test_pred_mw_25), max(y_test_pred_mw)]) + 10;

figure;

subplot(1, 2, 2)
scatter(w_avg_test, y_test_pred_mw_25, '.'), grid on;
ylim([0, y_max]);
xlabel('Temperatura Media (°F)')
ylabel('Carico elettrico (MW)')
title('Previsioni MLP con 19 neuroni a 25 temperature')

subplot(1, 2, 1)
scatter(w_avg_test, y_test_pred_mw, '.'), grid on;
ylim([0, y_max]);
xlabel('Temperatura Media (°F)')
ylabel('Previsioni Carico elettrico (MW)')
title('Previsioni MLP con 8 neuroni a temperatura media')

sgtitle('Confronto predizioni')
%% grafico reale vs predetto puntuale

ore_da_visualizzare = 1:48; 

y_reale = data_test.LOAD(ore_da_visualizzare);
y_pred  = y_test_pred_mw_25(ore_da_visualizzare);
t  = 1:length(ore_da_visualizzare);

figure;

scatter(t, y_reale, 30, 'k', 'LineWidth', 1); 
hold on
scatter(t, y_pred, 30, '+', 'LineWidth', 1); 

grid on;
xlabel('Tempo in ore');
ylabel('Carico Elettrico (MW)');
title('Confronto Puntuale: Reale vs MLP (Dettaglio 48h)');
legend('Dati Reali', 'Predizioni MLP');

%% Andamento temporale

t = 1:length(data_test.LOAD);

figure('Color', 'w', 'Name', 'Analisi Ciclicità Settimanale');

% --- SETTIMANA 1 ---
subplot(2, 1, 1)
idx1 = 1:168;
plot(t(idx1), data_test.LOAD(idx1), 'k', 'LineWidth', 1.2); hold on;
plot(t(idx1), y_test_pred_mw_25(idx1), 'r--', 'LineWidth', 1.1);

xlabel('Tempo (ore)');
ylabel('Carico (MW)');
title('Confronto Temporale: Prima Settimana');

% Linee grigie ogni 24 ore per vedere i singoli giorni
for d = 24:24:144
    xline(d);
end
legend('Andamento Reale', 'Predizioni MLP', 'Location', 'northeast');

% --- SETTIMANA 2 ---
subplot(2, 1, 2)
idx2 = 169:336;

t_sett = 1:168; 
plot(t_sett, data_test.LOAD(idx2), 'k', 'LineWidth', 1.2); hold on;
plot(t_sett, y_test_pred_mw_25(idx2), 'r--', 'LineWidth', 1.1);

xlabel('Tempo (ore)');
ylabel('Carico (MW)');
title('Confronto Temporale: Seconda Settimana');
for d = 24:24:144
    xline(d);
end

legend('Andamento Reale', 'Predizioni MLP');


