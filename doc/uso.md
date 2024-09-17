---
layout: default
title: Como Usar
---

# Como Usar

Após a instalação, siga os passos abaixo para usar o PrevisAI:

1. **Executar o Software:**

    Ao executar o arquivo principal, o sistema solicitará que você insira o símbolo de uma ação (por exemplo, AAPL para Apple, MSFT para Microsoft).

    ```bash
    python previsai.py
    ```

2. **Obter Resultados:**

    O PrevisAI baixa os dados históricos do Yahoo Finance e exibe o preço de fechamento atual do ativo. Em seguida, o software treina o modelo LSTM e faz previsões dos preços para os próximos cinco dias úteis. O resultado inclui:

    - Previsões dos preços de fechamento para os próximos 5 dias úteis.
    - Erro Quadrático Médio (MSE) e Erro Absoluto Médio (MAE) das previsões.
    - Variação percentual esperada do preço do ativo.
    - Recomendações baseadas nas previsões: comprar, vender ou manter.

Para exemplos específicos de uso, consulte [Exemplos de Uso](exemplos.md).
