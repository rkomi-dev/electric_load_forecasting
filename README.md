# electric load forecasting

## Data exploration

<img width="2160" height="1259" alt="smile" src="https://github.com/user-attachments/assets/631093c7-436d-4475-8ef1-38e71e5fec48" />
<br>
<img width="2160" height="1259" alt="esplorativi" src="https://github.com/user-attachments/assets/75b78e98-3a35-485e-b250-bea8821b0103" />
<br>
<img width="2160" height="1259" alt="heatmap" src="https://github.com/user-attachments/assets/14e7b3ba-a295-4021-8e32-9d24fbc23a98" />
<br>
<img width="1728" height="1152" alt="carnet_plot" src="https://github.com/user-attachments/assets/41f62cd7-c31c-4902-9270-94f93ce77802" />

<br>

## Univariate analysis using the average temperature as the only regressor

* **Model complexity vs RMSE**
<img width="2160" height="1259" alt="complessità_univariata" src="https://github.com/user-attachments/assets/353a1d1d-dbfd-4364-becf-f1298182baa8" />

<br>

* **Response curve and performance**
<img width="2160" height="1259" alt="curva_quarto_grado" src="https://github.com/user-attachments/assets/e2f26ef3-ad05-4265-b2c6-2825c537fa5e" />

<br>

## Modelling with polynomial regression using average temperature and timestamp

* **Model complexity vs RMSE**

<img width="2160" height="1259" alt="complessità_poli" src="https://github.com/user-attachments/assets/fc3eb9cb-d213-47b8-b437-796ce728fb87" />

<br>

* **Response surface: 5th degree vs 5th degree + Fourier terms**

<img width="2160" height="1259" alt="confronto_sup_poli(1)" src="https://github.com/user-attachments/assets/4a1351e1-78e7-45a3-8eff-6950032c517c" />




<br>

## Modelling with Stepwise regression method

* **Response surface: 5th degree + Fourier terms vs stepwise + Fourier terms**

<img width="2160" height="1259" alt="confronto_sup_poli_vs_step(1)" src="https://github.com/user-attachments/assets/577e3c92-b26d-497e-b743-60d0c8ea3ca1" />


<br>

## Final comparison using average temperature: Stepwise + Fourier vs 8-neuron MLP

* **Neuron selection using k-fold cv and 1-SD rule**
<img width="2160" height="1259" alt="scelta_neuroni" src="https://github.com/user-attachments/assets/ce987bde-6d1e-44c3-b044-9b56c3e7180c" />


<br>

* **Response surface and performance**

<img width="2160" height="1259" alt="confronto_sup_step_vs_mlp" src="https://github.com/user-attachments/assets/cb002b8f-6fe7-425b-9540-1706aa42d555" />




 <br>
 

## Final model using all 25 temperatures

* **Neuron selection using k-fold cv and 1-SD rule**

<img width="2160" height="1259" alt="scelta_neuroni_25" src="https://github.com/user-attachments/assets/ca565e73-df8c-4342-b9db-ffa0f1fd7fc6" />

<br>

* **Goodness of Fit**

<img width="2160" height="1259" alt="confronto_gof_mlp" src="https://github.com/user-attachments/assets/bbb61482-efc2-48f3-93d7-1f7cb453a7b8" />


<br>


* **Residue histogram**

<img width="2160" height="1259" alt="isto_residui_25" src="https://github.com/user-attachments/assets/1f73c4b2-df45-4f86-b47f-c3d256ef236e" />



<br>

* **Prediction**

<img width="2160" height="1259" alt="confronto_pred_mlp" src="https://github.com/user-attachments/assets/9d10d8ef-f753-46a4-bf0a-d6af6e5c6624" />


<br>

## Performance of all models on test set

| MODELLO | RMSE | MAPE | R^2 |
| :--- | ---: | ---: | ---: |
| Polinomio 5° grado + Fourier | 17.93 | 10.14% | 0.8507 |
| Stepwise + Fourier | 15.96 | 8.75% | 0.8818 |
| MLP con 8 neuroni | 14.93 | 8.06% | 0.8968 |
| MLP con 8 neuroni (25 temp) | 10.79 | 6.03% | 0.9460 |
| <b>MLP con 19 neuroni (25 temp) | <b>10.35 | <b>5.79% | <b>0.9503 |
