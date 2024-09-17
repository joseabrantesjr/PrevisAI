import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Dropout # type: ignore
from keras.optimizers import Adam # type: ignore
import yfinance as yf
from datetime import datetime, timedelta

def preprocessar_dados(data, window_size=20):
    """
    Pré-processa os dados de preços de fechamento da ação, aplicando normalização e
    criando janelas de tempo para alimentar o modelo LSTM.

    Parâmetros:
    - data (DataFrame): Dados históricos de preços da ação, contendo ao menos a coluna 'Close'.
    - window_size (int): O tamanho da janela de dias consecutivos para usar como entrada no modelo. Padrão é 20 dias.

    Retorna:
    - X (ndarray): Conjunto de janelas de dados de entrada para o modelo, com cada janela tendo um tamanho de 'window_size'.
    - y (ndarray): Valores alvo correspondentes, que são os preços de fechamento subsequentes à janela de entrada.
    - scaler (MinMaxScaler): O escalador utilizado para normalizar os dados, que será usado para reverter a normalização nas previsões.
    """
    
    # Inicializa o MinMaxScaler para normalizar os dados na faixa de 0 a 1
    scaler = MinMaxScaler()
    
    # Normaliza os preços de fechamento ('Close') usando o MinMaxScaler
    scaled_data = scaler.fit_transform(data[['Close']])
    
    # Listas para armazenar as janelas de dados de entrada (X) e os preços alvo (y)
    X, y = [], []
    
    # Loop através dos dados normalizados, criando janelas de 'window_size' dias
    for i in range(len(scaled_data) - window_size):
        # Cria uma janela de 'window_size' dias para ser usada como entrada no modelo
        X.append(scaled_data[i:i + window_size])
        
        # O valor alvo é o preço de fechamento no dia seguinte ao final da janela
        y.append(scaled_data[i + window_size])
    
    # Converte as listas para arrays NumPy e retorna os dados processados e o scaler
    return np.array(X), np.array(y), scaler


def criar_modelo_lstm(input_shape):
    """
    Cria e compila um modelo LSTM para previsão de séries temporais de preços de ativos.

    Parâmetros:
    - input_shape (tuple): A forma dos dados de entrada para o modelo, tipicamente (window_size, 1), 
      onde 'window_size' é o número de dias usados na janela de entrada e 1 é o número de recursos (preço de fechamento).

    Retorna:
    - model (Sequential): O modelo LSTM compilado e pronto para treinamento.
    """
    
    # Define um modelo sequencial para empilhar camadas de redes neurais
    model = Sequential([
        # Primeira camada LSTM com 256 unidades, função de ativação 'tanh' e ativação da sequência de retorno
        LSTM(256, activation='tanh', return_sequences=True, input_shape=input_shape),
        
        # Camada Dropout com taxa de 0.4 para evitar overfitting, descartando 40% dos neurônios aleatoriamente
        Dropout(0.4),
        
        # Segunda camada LSTM com 256 unidades e função de ativação 'tanh', sem retorno de sequência
        LSTM(256, activation='tanh'),
        
        # Outra camada Dropout para regularização, com 40% de taxa de dropout
        Dropout(0.4),
        
        # Camada densa (totalmente conectada) final que gera a previsão de um único valor (preço futuro)
        Dense(1)
    ])
    
    # Compila o modelo com o otimizador Adam e a função de perda mean_squared_error (MSE)
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    
    # Retorna o modelo compilado, pronto para ser treinado
    return model


def treinar_modelo(X, y):
    """
    Treina o modelo LSTM usando os dados de entrada (X) e os valores-alvo (y).
    Divide os dados em conjuntos de treino e teste, e ajusta o modelo.

    Parâmetros:
    - X (ndarray): Conjunto de janelas de dados de entrada para o modelo, onde cada janela tem 'window_size' dias.
    - y (ndarray): Valores alvo correspondentes, que são os preços de fechamento subsequentes às janelas de entrada.

    Retorna:
    - model (Sequential): O modelo LSTM treinado.
    - X_test (ndarray): Dados de teste usados para avaliar o modelo.
    - y_test (ndarray): Valores alvo de teste usados para comparar com as previsões do modelo.
    """
    
    # Divide os dados em conjuntos de treino (80%) e teste (20%) de forma aleatória.
    # A variável 'random_state' garante que a divisão seja sempre a mesma para fins de reprodutibilidade.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Cria o modelo LSTM, onde a entrada tem o formato (window_size, 1).
    model = criar_modelo_lstm((X.shape[1], 1))
    
    # Treina o modelo usando os dados de treino. 
    # - epochs=300: O número de vezes que o modelo verá todos os dados de treino.
    # - batch_size=64: Quantidade de amostras processadas antes de atualizar os parâmetros do modelo.
    # - validation_split=0.2: 20% dos dados de treino serão usados para validação durante o treinamento.
    # - verbose=0: O treinamento será silencioso (sem saída de texto).
    model.fit(X_train, y_train, epochs=300, batch_size=64, validation_split=0.2, verbose=0)
    
    # Retorna o modelo treinado, além dos conjuntos de teste (X_test e y_test) para avaliação futura.
    return model, X_test, y_test


def avaliar_modelo(model, X_test, y_test):
    """
    Avalia o desempenho do modelo LSTM em dados de teste, calculando o erro quadrático
    médio (MSE) e o erro absoluto médio (MAE).

    Parâmetros:
    - model (Sequential): O modelo LSTM treinado que será avaliado.
    - X_test (ndarray): Conjunto de dados de entrada de teste (janelas de preços).
    - y_test (ndarray): Conjunto de dados de saída de teste (preços reais correspondentes aos dados de entrada).

    Retorna:
    - mse (float): O erro quadrático médio das previsões do modelo.
    - mae (float): O erro absoluto médio das previsões do modelo.
    """
    
    # Gera as previsões do modelo para os dados de teste (X_test)
    predictions = model.predict(X_test)
    
    # Calcula o erro quadrático médio (MSE) entre as previsões e os valores reais (y_test)
    mse = np.mean((predictions - y_test) ** 2)
    
    # Calcula o erro absoluto médio (MAE) entre as previsões e os valores reais
    mae = np.mean(np.abs(predictions - y_test))
    
    # Retorna o MSE e o MAE
    return mse, mae


def prever_proxima_semana(model, last_window, scaler):
    """
    Usa o modelo treinado para prever os preços de fechamento para os próximos 5 dias úteis
    com base nos últimos 20 dias de dados (last_window).

    Parâmetros:
    - model: O modelo LSTM treinado.
    - last_window (ndarray): A última janela de dados de preços de fechamento, geralmente os últimos 20 dias.
    - scaler (MinMaxScaler): O escalador usado para normalizar os dados no preprocessamento, utilizado para reverter a normalização.

    Retorna:
    - ndarray: Previsões de preços de fechamento para os próximos 5 dias úteis, desnormalizados para os valores reais.
    """
    
    # Normaliza a última janela de dados de preços usando o mesmo escalador usado durante o treino
    last_window_scaled = scaler.transform(last_window)
    
    # Lista para armazenar as previsões dos próximos 5 dias úteis
    predictions = []
    
    # Faz previsões para os próximos 5 dias úteis, utilizando as previsões anteriores como entrada para o próximo dia
    for _ in range(5):  # Prever os próximos 5 dias úteis
        # Usa o modelo para prever o próximo preço de fechamento
        next_pred = model.predict(last_window_scaled.reshape(1, -1, 1))
        
        # Adiciona a previsão do dia à lista de previsões
        predictions.append(next_pred[0, 0])
        
        # Atualiza a janela de dados deslizante:
        # Remove o primeiro valor da janela e adiciona a nova previsão no final
        last_window_scaled = np.roll(last_window_scaled, -1)
        last_window_scaled[-1] = next_pred
    
    # Desnormaliza as previsões para retornar os valores reais dos preços
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))


def identificar_melhor_compra(data):
    """
    Identifica a melhor oportunidade de compra de ações, baseado na previsão
    dos preços de fechamento para os próximos 5 dias úteis.

    Parâmetros:
    - data (DataFrame): Dados históricos de preços da ação, contendo ao menos a coluna 'Close'.

    Retorna:
    - proxima_semana_dates (DatetimeIndex): Datas dos próximos 5 dias úteis.
    - proxima_semana_prices (ndarray): Preços de fechamento previstos para os próximos 5 dias úteis.
    - mse (float): Erro Quadrático Médio do modelo ao avaliar os dados de teste.
    - mae (float): Erro Absoluto Médio do modelo ao avaliar os dados de teste.
    """
    
    # Pré-processa os dados históricos de fechamento e retorna os dados de entrada (X), 
    # os preços-alvo (y), e o escalador para normalização
    X, y, scaler = preprocessar_dados(data)
    
    # Treina o modelo LSTM usando os dados processados e divide entre treino e teste
    model, X_test, y_test = treinar_modelo(X, y)
    
    # Avalia o modelo nos dados de teste e retorna o MSE (Erro Quadrático Médio) e MAE (Erro Absoluto Médio)
    mse, mae = avaliar_modelo(model, X_test, y_test)
    
    # Seleciona os últimos 20 dias de preços de fechamento para usá-los como janela de entrada para a previsão
    last_window = data['Close'].values[-20:].reshape(-1, 1)
    
    # Faz a previsão dos preços para os próximos 5 dias úteis utilizando o modelo treinado
    proxima_semana_prices = prever_proxima_semana(model, last_window, scaler)
    
    # Gera as datas correspondentes aos próximos 5 dias úteis, começando no dia seguinte ao último dia dos dados
    proxima_semana_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=5, freq='B')
    
    # Retorna as datas previstas, os preços previstos, e os erros de avaliação (MSE e MAE)
    return proxima_semana_dates, proxima_semana_prices.flatten(), mse, mae


def main():
    """
    Função principal que executa o programa de previsão de preços de ações. 
    Solicita ao usuário o símbolo de uma empresa, baixa os dados históricos, 
    realiza a previsão dos preços para os próximos 5 dias e exibe as recomendações 
    com base nas previsões.
    """
    
    # Solicita ao usuário o símbolo da empresa para análise (por exemplo, 'AAPL' para Apple)
    simbolo_empresa = input("Digite o símbolo da empresa (por exemplo, AAPL para Apple Inc.): ")
    
    # Obtém a data atual no formato 'YYYY-MM-DD'
    data_atual = datetime.today().strftime('%Y-%m-%d')
    
    # Baixa os dados históricos da empresa a partir de 11 de setembro de 2020 até a data atual
    dados = yf.download(simbolo_empresa, start='2020-09-11', end=data_atual)
    
    # Verifica se os dados foram obtidos corretamente. Caso não haja dados, informa ao usuário.
    if dados.empty:
        print("Não foi possível obter dados para o símbolo fornecido.")
        return  # Finaliza a execução se não houver dados
    
    # Exibe o preço de fechamento mais recente do ativo
    print(f"Preço de Fechamento Atual: ${dados['Close'].iloc[-1]:.2f}")
    
    # Chama a função que identifica a melhor compra, fazendo previsões e calculando métricas
    dates, prices, mse, mae = identificar_melhor_compra(dados)
    
    # Exibe as previsões de preços para os próximos 5 dias úteis
    print("\nPrevisão de preços para a próxima semana:")
    for date, price in zip(dates, prices):
        print(f"{date.strftime('%d/%m/%Y')}: ${price:.2f}")
    
    # Exibe as métricas de desempenho do modelo: MSE e MAE
    print(f"\nErro Quadrático Médio (MSE) do modelo: {mse:.4f}")
    print(f"Erro Absoluto Médio (MAE) do modelo: {mae:.4f}")
    
    # Calcula a variação percentual prevista para o preço do ativo
    ultimo_preco_real = dados['Close'].iloc[-1]
    ultimo_preco_previsto = prices[-1]
    variacao_percentual = ((ultimo_preco_previsto - ultimo_preco_real) / ultimo_preco_real) * 100
    
    # Exibe a variação percentual prevista
    print(f"\nVariação percentual prevista: {variacao_percentual:.2f}%")
    
    # Exibe uma recomendação com base na variação percentual prevista
    if variacao_percentual > 0:
        print("Recomendação: Considere comprar. O modelo prevê uma tendência de alta.")
    elif variacao_percentual < 0:
        print("Recomendação: Considere vender ou manter. O modelo prevê uma tendência de baixa.")
    else:
        print("Recomendação: Manter. O modelo prevê estabilidade no preço.")

# Verifica se o script está sendo executado diretamente e, em caso afirmativo, chama a função main()
if __name__ == "__main__":
    main()
