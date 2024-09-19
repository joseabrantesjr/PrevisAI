## **PrevisAI**

### Visão Geral

**PrevisAI: Inteligência Avançada para Previsão de Ativos** é uma aplicação em Python desenvolvida para prever os preços de fechamento de ações utilizando técnicas de aprendizado profundo, especificamente redes LSTM (Long Short-Term Memory). A programação visa ajudar investidores a tomar decisões informadas com base em previsões de curto prazo (até uma semana), recomendando compra ou venda de ativos.

A ferramenta utiliza dados históricos obtidos através da API do Yahoo Finance para treinar um modelo que faz previsões precisas sobre o comportamento dos preços. A partir das previsões, a  programação fornece uma análise clara e uma recomendação de investimento.

---

### Funcionalidades Principais

1. **Previsão de Preços de Ações**: O script prevê o preço de fechamento de um ativo para os próximos 5 dias úteis, com base em 20 dias de dados históricos.
   
2. **Análise de Erro**: Avalia a precisão do modelo utilizando o Erro Quadrático Médio (MSE) e o Erro Absoluto Médio (MAE) para garantir previsões robustas.

3. **Recomendação de Compra/Venda**: Com base nas previsões, o PrevisAI calcula a variação percentual esperada e oferece uma recomendação de compra, venda ou manutenção do ativo.

4. **Interatividade**: Permite ao usuário inserir o símbolo de qualquer ativo listado na bolsa para análise imediata.

---

### Instalação

#### Pré-requisitos:
- Python 3.x


### Método recomendado (Docker):
  ```bash
 docker run -it previsai
 ```
### Método manual:
- Criar um ambiente vritual:
  ```bash
  python3 -m venv venv
  ```

- Ativar o ambiente virtual (MacOS, Linux):
  ```bash
  source venv/bin/activate
  ```

- Instalar as bibliotecas necessárias:
  ```bash
  pip install -r requirements.txt
  ```

#### Como instalar:
1. Clone o repositório do software (ou baixe o código):
   ```bash
   git clone https://github.com/joseabrantesjr/PrevisAI
   ```

2. Navegue até o diretório do projeto:
   ```bash
   cd PrevisAI
   ```

3. Execute o arquivo principal `Previsai.py`:
   ```bash
   python previsai.py
   ```

---

### Como Usar

1. **Executar o Software**:
   - Ao executar o arquivo principal, o sistema solicitará que você insira o símbolo de uma ação (exemplo: `AAPL` para Apple, `MSFT` para Microsoft).

2. **Obter Resultados**:
   - O PrevisAI baixa os dados históricos do Yahoo Finance e exibe o preço de fechamento atual do ativo.
   - Em seguida, o software treina o modelo LSTM e faz previsões dos preços para os próximos cinco dias úteis.
   - O resultado inclui as seguintes informações:
     - Previsões dos preços de fechamento para os próximos 5 dias úteis.
     - Erro Quadrático Médio (MSE) e Erro Absoluto Médio (MAE) das previsões.
     - Variação percentual esperada do preço do ativo.
     - Recomendações baseadas nas previsões: **comprar**, **vender** ou **manter**.

3. **Interpretação das Recomendações**:
   - **Comprar**: Se a previsão mostra um aumento percentual significativo no preço do ativo.
   - **Vender**: Se há uma previsão de queda no preço do ativo.
   - **Manter**: Se a previsão indica estabilidade.

---

### Exemplos de Uso

#### Exemplo 1:
```
Digite o símbolo da empresa (por exemplo, AAPL para Apple Inc.): MSFT
```
**Saída**:
```
Preço de Fechamento Atual: $280.15

Previsão de preços para a próxima semana:
18/09/2024: $282.35
19/09/2024: $284.00
20/09/2024: $285.50
23/09/2024: $287.25
24/09/2024: $288.90

Erro Quadrático Médio (MSE) do modelo: 0.0041
Erro Absoluto Médio (MAE) do modelo: 0.0523

Variação percentual prevista: 3.12%

Recomendação: Considere comprar. O modelo prevê uma tendência de alta.
```

#### Exemplo 2:
```
Digite o símbolo da empresa (por exemplo, AAPL para Apple Inc.): TSLA
```
**Saída**:
```
Preço de Fechamento Atual: $770.50

Previsão de preços para a próxima semana:
18/09/2024: $765.20
19/09/2024: $762.50
20/09/2024: $759.10
23/09/2024: $754.75
24/09/2024: $750.40

Erro Quadrático Médio (MSE) do modelo: 0.0037
Erro Absoluto Médio (MAE) do modelo: 0.0419

Variação percentual prevista: -2.61%

Recomendação: Considere vender ou manter. O modelo prevê uma tendência de baixa.
```

---

### Estrutura do Código

1. **Preprocessamento dos Dados**
   - Função: `preprocessar_dados(data, window_size=20)`
   - Normaliza os dados e cria janelas de 20 dias de preços para uso no modelo.

2. **Modelo LSTM**
   - Função: `criar_modelo_lstm(input_shape)`
   - Define a arquitetura LSTM e compila o modelo com o otimizador Adam.

3. **Treinamento do Modelo**
   - Função: `treinar_modelo(X, y)`
   - Divide os dados em treino e teste, e ajusta o modelo utilizando 300 épocas.

4. **Avaliação do Modelo**
   - Função: `avaliar_modelo(model, X_test, y_test)`
   - Calcula o MSE e MAE para avaliar a precisão do modelo em dados não vistos.

5. **Previsão da Próxima Semana**
   - Função: `prever_proxima_semana(model, last_window, scaler)`
   - Usa os últimos 20 dias de dados e faz previsões para os próximos 5 dias úteis.

6. **Recomendações**
   - Função: `identificar_melhor_compra(data)`
   - Integra todas as etapas, faz as previsões e calcula a variação percentual, gerando a recomendação de compra, venda ou manutenção.

---

### Melhoria e Expansão

**PrevisAI** pode ser expandido com várias funcionalidades adicionais, como:
1. **Análise Técnica e Fundamentalista**: Integrar indicadores técnicos ou dados financeiros fundamentais para enriquecer as previsões.
2. **Alocação de Portfólio**: Combinar as previsões de múltiplos ativos para otimização de portfólio, por exemplo, utilizando a teoria de portfólio de Markowitz.
3. **Aprimoramento do Modelo**: Implementar outros tipos de redes neurais ou combinar LSTM com modelos de machine learning tradicionais para melhorar a precisão.

---

### **Como Contribuir e Se Beneficiar:**
1. **Dê uma Estrala**: Se você aprecia o nosso trabalho e acha que ele pode beneficiar outros, mostre seu apoio clicando no botão "Star" no GitHub!
2. **Fork e Contribua**: Faça um fork do repositório, experimente o código e contribua com melhorias. Sua participação pode tornar o PrevisAI ainda mais poderoso!
3. **Compartilhe**: Ajude a espalhar a palavra. Compartilhe o PrevisAI com colegas, amigos e em suas redes sociais!

### **Experimente o PrevisAI Hoje!**
Junte-se à comunidade de investidores e analistas que estão transformando a maneira como analisamos o mercado financeiro. Faça parte do futuro da previsão de preços com o **PrevisAI**!

---
### ***Sobre Mim***
Sou José Calazans Abrantes Júnior, um profissional com conhecimentos avançados em , machine learning, deep learning, análise de dados. Através de uma sólida formação autodidata, desenvolvi habilidades em técnicas de previsão e análise financeira, utilizando redes neurais, como LSTM, além da utilização de modelos para otimização de portfólio.

Atualmente, trabalho na Bio-G, em Resende/RJ, aplicando meu conhecimento para otimizar processos de saneamento. Também desenvolvi o Flask Password Manager, uma aplicação web que assegura a proteção de senhas com criptografia avançada.

Minha trajetória profissional inclui a gestão de uma microempresa de artigos esportivos (proprietário) e diversas funções administrativas e logísticas em diferentes empresas, o que demonstra minha versatilidade e capacidade de adaptação. Além disso, tenho experiência internacional com trabalho voluntário na Inglaterra e na Espanha.

Sou proficiente em Python, PHP, SQL, Node e várias bibliotecas e frameworks, como TensorFlow, Keras e Flask. Estou sempre em busca de aprimorar minhas habilidades e contribuir para inovações nos campos financeiro e ambiental.

---

### Licença

O software **PrevisAI** é disponibilizado sob a licença MIT. Sinta-se livre para modificar, distribuir e usar para fins pessoais ou comerciais, desde que seja dado o devido crédito aos desenvolvedores originais.

---

**PrevisAI: Antecipe o mercado, maximize lucros.**


docker build -t previsai .
