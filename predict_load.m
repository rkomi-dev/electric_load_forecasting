function yhat = predict_load(X)
    % PREDICT_LOAD Previsione del carico elettrico tramite MLP
    % Input: X - Tabella con colonne TIMESTAMP, w1, w2, ..., w25
    % Output: yhat - Vettore colonna delle previsioni in MW
    
    % Caricamento del modello salvato

    persistent net_finale_25 settings_x_25 settings_y_25
    if isempty(net_finale_25)
        data_modello = load('modello_finale.mat');
        net_finale_25 = data_modello.net_finale_25;
        settings_x_25 = data_modello.settings_x_25;
        settings_y_25 = data_modello.settings_y_25;
    end
    
    temp_matrix = table2array(X(:, 2:26));
    
    % Trasformazione Periodica del TIMESTAMP
    T_period = 24;
    time_sin = sin(2 * pi * X.TIMESTAMP / T_period);
    time_cos = cos(2 * pi * X.TIMESTAMP / T_period);
    
    % Preparazione della matrice di input 
    phi_vec = [temp_matrix, time_sin, time_cos]';
    
    % Normalizzazione Input
    phi_vec_n = mapminmax('apply', phi_vec, settings_x_25);
    
    % Predizione 
    yhat_n = net_finale_25(phi_vec_n);
    
    % Denormalizzazione Output (MW)
    yhat = mapminmax('reverse', yhat_n, settings_y_25)';
    
end