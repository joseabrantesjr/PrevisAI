---
layout: default
title: Estrutura do Código
---

# Estrutura do Código

O código do PrevisAI é dividido nas seguintes partes principais:

## Preprocessamento dos Dados

**Função:** `preprocessar_dados(data, window_size=20)`

Normaliza os dados e cria janelas de 20 dias de preços para uso no modelo.

## Modelo LSTM

**Função:** `criar_modelo_lstm(input_shape)`

Define a arquitetura LSTM e compila o modelo com o otimizador Adam.

## Treinamento do Modelo

**Função:** `treinar_modelo(X, y)`

Divide os dados em treino e teste, e ajusta o modelo utilizando 300 épocas.

## Avaliação do Modelo

**Função:** `avaliar_modelo(model, X_test, y_test)`

Calcula o MSE e MAE para avaliar a precisão do modelo em dados não vistos.

## Previsão da Próxima Semana

**Função:** `prever_proxima_semana(model, last_window, scaler)`

Usa os últimos 20 dias de dados e faz previsões para os próximos 5 dias úteis.

## Recomendações

**Função:** `identificar_melhor_compra(data)`

Integra todas as etapas, faz as previsões e calcula a variação percentual, gerando a recomendação de compra, venda ou manutenção.
