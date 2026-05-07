clear
clc
close all

%% Caricamento e Pre-processing
data = readtable('L1_train.csv');
data = rmmissing(data);

temp_matrix = table2array(data(:, 3:27));
w_avg = mean(temp_matrix, 2);
data.W_AVG = w_avg; 

N_tot = height(data);
idx_rand = randperm(N_tot); 

% 70% Train, 15% Val, 15% Test
n_train = floor(0.70 * N_tot);
n_val   = floor(0.15 * N_tot);

% Split Randomizzato
data_train = data(idx_rand(1:n_train), :);
data_val   = data(idx_rand(n_train+1 : n_train+n_val), :);
data_test  = data(idx_rand(n_train+n_val+1 : end), :);
data_trainval = [data_train; data_val]; 

w_avg_train = data_train.W_AVG;
w_avg_val   = data_val.W_AVG;
w_avg_test  = data_test.W_AVG;
w_avg_trainval = data_trainval.W_AVG;

n = height(data_train);
n_v = height(data_val);
n_t = height(data_test);
%% Plot Esplorativi

% carico in funzione del tempo
figure(1)
plot(data.LOAD), grid on
xlabel('Tempo'), ylabel('Carico elettrico (MW)')
sgtitle('Carico elettrico in funzione del tempo')

% temp_media in funzione del tempo 
figure(2)
plot(w_avg), grid on
xlabel('Tempo'), ylabel('Temperatura Media (°F)') 
sgtitle('Temperatura media in funzione del tempo')

% carico in funzione della temperatura media
figure(3)
scatter(w_avg, data.LOAD, '.'), grid on;
xlabel('Temperatura Media (°F)'), ylabel('Carico elettrico (MW)')
sgtitle('Carico elettrico in funzione della temperatura media')

%% Generazione Carpet Plot Annuali (Anni 1-6)
ore_anno = 8760;
n_anni = 6;

figure('Name', 'Analisi Evoluzione Carico Annuale', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);

for i = 1:n_anni
   
    idx_inizio = (i-1) * ore_anno + 1;
    idx_fine = i * ore_anno;
    
    if idx_fine > height(data)
        break;
    end
    
    % Estrazione e reshape dei dati (24 ore x 365 giorni)
    carico_anno = data.LOAD(idx_inizio:idx_fine);
    matrice_carpet = reshape(carico_anno, 24, 365);
   
    subplot(2, 3, i); 
    imagesc(matrice_carpet);
    colormap(jet);
    colorbar;
    
    title(['Anno ', num2str(i)]);
    xlabel('Giorno dell''anno');
    ylabel('Ora');
    set(gca, 'YDir', 'normal'); 
    grid on;
    
    clim([min(data.LOAD) max(data.LOAD)]); 
end

sgtitle('Carpet Plots: Confronto del Carico Elettrico negli Anni');
%% Matrice di Correlazione
matrix_corr = corr(table2array(data(:, 3:27)));

figure;
heatmap(matrix_corr, 'ColorMap', flipud(hot));
sgtitle('Heatmap della matrice di correlazione')

%% Salvataggio per gli script successivi
save('preprocessed_data.mat', 'w_avg', 'data_train', 'data_val', 'data_test', 'data_trainval', ...
    'w_avg_train', 'w_avg_val', 'w_avg_test', 'w_avg_trainval', 'n', 'n_v', 'n_t', 'temp_matrix');