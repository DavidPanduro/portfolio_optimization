# Projeto: Portfolio Otimization through deep learning and otimization algorithms
Por David Panduro 💻<br><br>
![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/1ff6e2db-01b1-47f0-b125-a2b674c0581a)<br>

CONTEXTO:<br>
Desenvolvemos a otimização de três (03) portfólios composto por ações de três (03) empresas (que no inicio do 2023 foram sugeridos por reconhecidos Investidores Brasileiros, que não serão explícitamente mencionados 👀) da bolsa brasileira 🇧🇷, aplicando algoritmos de otimização e aprendizagem profunda em cenários de risco não sistemático, analisando os riscos e procurando maximizar os rendimentos do portfólio ao final do período. <br> <br>
BASES: <br> 
No estudo, consideramos dados de movimentações da Bolsa Brasileira, no período desde 2018-01-01 até 2023-06-30. Totalizando cinco (05) anos, mais os restantes seis (06) meses do 2023.<br><br>
Portfólios:<br> Os portfólios estão compostos pelas seguintes empresas:<br>
1. Portfólio_LB. [BB Seguridade, Banco do Brasil, Cosan].<br>
   ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/6f6e98f8-dbdd-484f-8a7e-89030a2c8ff1)<br><br>

2. Portfólio_TN. [RAPT3, RANI3, LEVE3]<br>
   ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/cc524e0e-a471-4054-b6e6-e3334131f632)<br><br>

3. Portfólio_BP. [Itausa, Vivo, Sanepa]<br>
   ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/7cf26b33-d09e-45ef-a594-4ca32014a2fa)<br><br>


PORTFÓLIO LB:<br>
Podemos observar os portfólios para ter uma visão mais clara do comportamento das ações de empresas que o conformam.<br>
   ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/62460459-8cde-4074-9b01-98fff842f959)<br>
A primeira vista, as ações da BBAS3 mostram-se com melhor rendimento ao longo do periodo.<br><br>
Mas, percebe-se que as linhas tem diferenças significativas, básicamente, porque não estão na mesma escala. Vamos tentar resolver isso, e observar que os comportamentos se mostram de manera diferente, porem, agora sim mostram a realidade e podem ser comparados.<br>
   ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/800a6b54-174e-4bf5-a4ad-5038dd354589)<br>
   Agora sim, com a apresentação na mesma escala, podemos comparar entre elas. E percebemos que a que obteve melhores resultados foi a CSAN3.<br><br>
PORTFÓLIO TN:<br>
A continuação as empresas que compõem o portfolio TN para ter uma visão mais clara do desempenho.<br>
   ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/c84adc1c-b1d7-41d2-835f-1503eeb5e130)<br>
Novamente, em diferentes escalas resulta inviável reconhecer o melhor rendimento ao longo do periodo.<br><br>
Depois de converter para a mesma escala, o gráfico representa a realidade dos rendimentos das ações das empresas do portfólio.<br>
   ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/02ad52d1-bee6-4c93-9e36-83975a2a6917)<br>
   Com a apresentação na mesma escala, percebemos que a que obteve melhores resultados foi a RANI3.<br><br>
PORTFÓLIO BP:<br>
Por fim, o histórico das ações das empresas do portfólio BP ao longo do periodo.<br>   
   ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/a7779d9d-035c-42e2-aa14-f2837c5302a7)<br>
   Mas, como nos casos anteriores, precisamos aplicar a normalização para trazer a mesma escala.<br><br>
   ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/8aba2f6c-f2f3-49fe-b40e-1c8e125ac2d9)<br>
   E assim percebemos que as ações da VIVT3 conseguiram um melhor rendimento dentro do periodo histórico.<br><br>

TAXAS DE RETORNO:<br> A princípio, definiremos aleatóriamente os pesos para as carteiras de manera igualmente distribuido, e na sequencia aplicaremos esses pesos para saber quais seriam as nossas taxas de retorno ao longo do periodo.<br>
Para fins ilustrativos aplicaremos os seguintes pesos:<br>
**pesos_carteira = np.array([0.36, 0.32,0.32])** <br><br>
Aplicaremos a seguinte fórmula: retorno_carteira_lb = **(df_normalizado / df_normalizado.shift(1)) -1** <br>
Essa é a taxa de retorno simples, já que estamos fazendo comparativo em um mesmo periodo de tempo para várias empresas.<br>
Na sequencia, fazemos o calculo anual: retorno_anual = retorno_carteira.mean( ) * 246; normalmente aplica-se 246 dias de operação no ano.<br>
E assim, obtemos os seguintes resultados de taxa de retorno:<br><br>
* Portfólio_LB: 11.77
* Portfólio_TN: 28.94
* Portfólio_BP: 5.38 <br>
A taxa de retorno mais promissora, com esses pesos distribuidos aleatóriamente, seria o Portfolio_TN com taxa de retorno anual de 28.94<br><br>
   ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/6eab4132-1c29-4900-909f-b1f7f186c8ce)<br>
No gráfico podemos observar os três (03) portfólios normalizados, confirmando o melhor comportamento de taxa de retorno do Portfólio_TN, conseguindo performar e capitalizar pós Pandemia <br><br>

CÁLCULO DE RISCOS DOS PORTFÓLIOS:<br> 
Para calcular o risco basearemos o estudo na aplicação dos conceitos de variância, desvio padrão e coeficiente de variação, assim como covariância e correlação (normalmente, os portfólios com empresas do mesmo setor apresentam correlações entre elas). <br> Calculamos o risco medio anual para cada um dos portfólios, assim como a volatilidade dos portfólios por meio do cálculo de covariância e desvio padrão. 

   ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/7917c647-3b30-4bb5-9a20-75faf3ae47ad)<BR>
   Atenção no Portfólio_TN, onde acabamos apostando mais do que o 50% numa empresa, chegando a 0.61 de taxa de retorno, e no global podemos mencionar as seguintes observações:<br>
* O Portfolio_TN é o mais Arrojado, com media de 0.47 de risco.
* O Portfolio_BP é o mais Conservador, com media de 0.28 de risco.
* O Portfolio_LB acaba sendo Moderado, com media de 0.35 de risco.<br><br>
Agora, mostramos o gráfico de correlações de cada um dos portfólios:<br>
   ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/a57ab0ae-52f8-4561-8774-4f5af6d2b9fd)<br>
Nas correlações faz total sentido ter esse grau de correlação entre as empresas do Portfólio LB já que temos uma empresa que representa um grande banco brasileiro e outra empresa que não deixa de ser um braço de atuação do mesmo banco. Cabe salientar que é uma correlação intermédia e não extremamente forte<br> <br><br>

VOLATILIDADE DOS PORTFÓLIOS:<br>
A volatilidade de cada um dos portfólios será calculado aplicando o desvio padrão da nossa taxa de retorno: <br> volatilidade_portfolio = **math.sqrt(np.dot(pesos_carteira, np.dot(taxa_retorno.cov()* 246, pesos_carteira)))** <br>
Obtendo os seguintes resultados:<br>
   * Portfolio_TN com 32.5%.
   * Portfolio_LB 29%
   * Portfolio_BP 21% <br><br>
Claramente, a maior taxa de volatidade, maior risco, maiores opçõesde maior taxa de retorno.<br><br>

RISCO NÃO SISTEMÁTICO DOS PORTFÓLIOS:<br>
Este tipo de risco, responde a eventos específicos na empresa e depende do tipo de empresa. Em contrapartida com o risco sistemático, que responde a eventos externos, não pode ser eliminado e afeta a todas as empresas, podem ser resultado de eventos como recessão da economia (neste estudo não abordaremos este tipo de riscos).<br>
Para o cãlculo deste risco não sistemático, precisamos definir os pesos, calcular a variância anual e multiplicar pelos pesos, calcular a substração das variâncias e por último o <br>risco não sistemático é a **((variância do portfólio) - (substração das variâncias))** <br><br>
* Portfolio_TN sai na frente com 0.22
* Portfolio_LB que é o mais próximo com 0.10
* Portfolio_BP recua com 0.075<br><br>
Podemos comparar essa métrica com a Volatilidade e inclusive usar ambas métricas para analisar o risco dos portfólios.<br><br>

ALGORITMOS DE OTIMIZAÇÃO:<br>
Neste apartado DEFINIMOS pesos, calculamos o Sharpe Ratio (que é usado para medir o desempenho de uma carteira de ações) que no seu cálculo também utiliza o risco, também adicionamos o cálculo de Markowitz, para finalmente aplicar: <br>
   1. Alocação randômica de pesos para maximizar o valor do Sharpe Ratio.
   2. Hill Climb (subida da encosta).
   3. Simulated Anneling (têmpera simulada).<br><br>
   
ALOCAÇÃO RANDÔMICA:<br>
Por medio desta técnica obtimos os seguintes resultados

   ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/b3c77412-4df3-44c2-bad6-f05c5d23cfd5)<br>
   O método acabou definindo os seguintes PESOS:<br>
   ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/afda98d6-c221-4742-8356-074d6956d86f)<br><br>

   ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/655d1169-b3f7-41ee-a868-2a8c476874f1)<br>
   O método acabou definindo os seguintes PESOS:<br>
   ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/1d577cc1-153c-4739-b02f-c357f58fb7e4)<br><br>

   ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/0129bf46-e57d-4677-9e06-35a665ece8bd)<BR>
   O método acabou definindo os seguintes PESOS:<br>
   ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/7d2f5e50-15ba-4a94-b0e5-72fef5f5d2a0)<br><br>
   Finalmente, podemos observer em reais a evolução do patrimônio dos portfólios no periodo.<br>
   ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/e8f1baf7-249c-49f1-aae5-78c61d5416a5)<br><br>
   Destaca-se o Portfólio TN, conseguindo uma distância significativa em relação aos outros portfólios, que no periodo não conseguiram crescer de manera exponencial, objetivo que sim atingiu o Portfólio TN creciendo 5x do capital inicial.<br>

ÍNDICE DE SHARPE RATIO:<br>
O índice Sharpe Ratio, é uma medida de desempenho de um investimento que leva em consideração o retorno em excesso (retorno acima de um ativo livre de risco) em relação à sua volatilidade. 
![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/8aa8b25e-6acd-4bf7-a520-a6de9ccba8ca)<br>
Uma vez aplicada a fórmula, conseguimos os seguintes resultados:
* Portfólio LB: 0.35
* Portfólio TN: 0.78
* Portfólio BP: 0.31 <br><br>

ALOCAÇÃO COM MODELO MARKOWITZ: <br>
A alocação de ativos com o modelo de Markowitz envolve a distribuição de recursos em uma carteira de investimentos para otimizar o equilíbrio entre risco e retorno. Usamos a matriz de covariância e o conceito da fronteira eficiente de Markowitz para calcular a alocação de ativos que otimiza o equilíbrio entre risco e retorno.<br>
![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/4e413e8d-cb71-4eca-8341-cac89f7096de)<br><br>
   Depois de aplicar o modelo Markowitz, obtemos os seguintes resultados:<br><br>
   1. PORTFÓLIO LB
      * Pesos para BBAS3: 7.13
      * Pesos para BBSE3: 0.65
      * Pesos para CSAN3: 92.2
      * RETORNO: 7981.37 <br><br>
      ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/d0fc8fb7-86e8-4210-a030-d1b944462999)<br><br>

        
   2. PORTFÓLIO TN
      * Pesos para RAPT3: 31.27
      * Pesos para RANI3: 57.64
      * Pesos para LEVE3: 11.08
      * RETORNO: 22712.67 <br><br>
      ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/7d0c50f7-dc62-4247-83a9-beb7203e0905)<br><br>

       
   3. PORTFÓLIO BP
      * Pesos para ITSA3: 1.12
      * Pesos para VIVT3: 98.85
      * Pesos para SAPR11: 0.01
      * RETORNO: 5807.78 <br><br>
      ![image](https://github.com/DavidPanduro/portfolio_invest_otimization/assets/45201867/cb3ed4ee-3106-4303-9b4d-5d5d5dc19c51)<br><br>

MÉTODO HILL CLIMB (SUBIDA DE ENCOSTA): <br>
O método Hill Climbing é um algoritmo de otimização utilizado em problemas de busca heurística, onde o objetivo é encontrar a melhor solução em um espaço de busca, geralmente caracterizado por um espaço de estados e uma função de avaliação (função de custo) que atribui um valor a cada possível solução. Aplicando o método obtemos os seguintes resultados:<br><br>
   1. PORTFÓLIO LB
      * Pesos para BBAS3: 100.0
      * Pesos para BBSE3: 0.0
      * Pesos para CSAN3: 0.0
      * RETORNO: 7163.68 <br><br>
   2. PORTFÓLIO TN
      * Pesos para RAPT3: 0.0
      * Pesos para RANI3: 100.0
      * Pesos para LEVE3: 0.0
      * RETORNO: 32321.42 <br><br>
       
   3. PORTFÓLIO BP
      * Pesos para ITSA3: 0.0
      * Pesos para VIVT3: 99.98
      * Pesos para SAPR11: 0.01
      * RETORNO: 5812.75 <br><br>
   Os resultados do método Hill Climb ficaram muito parecidos em dois (02) portfólios em relação ao Método de Alocação Randômico. Já no Portfólio TN foi superior ao seu concorrente, crescendo em 30% aproximadamente.

SIMULATED ANNEALING (TÊMPERA SIMULADA):<br>
O Simulated Annealing é um algoritmo de otimização probabilística inspirado no processo físico de recozimento de metais, onde um material é gradualmente resfriado para alcançar um estado de menor energia e maior ordenação. Esse método é usado para encontrar soluções aproximadamente ótimas em problemas de otimização, especialmente quando a paisagem de busca é complexa, possui muitos mínimos locais e a função de custo é ruidosa ou não diferenciável.<br>
Aplicando o método obtemos os seguintes resultados:<br><br>
   1. PORTFÓLIO LB
      * Pesos para BBAS3: 0.0
      * Pesos para BBSE3: 100.0
      * Pesos para CSAN3: 0.0
      * RETORNO: 5437.28 <br><br>
   2. PORTFÓLIO TN
      * Pesos para RAPT3: 0.0
      * Pesos para RANI3: 100.0
      * Pesos para LEVE3: 0.0
      * RETORNO: 32321.42 <br><br>
       
   3. PORTFÓLIO BP
      * Pesos para ITSA3: 0.0
      * Pesos para VIVT3: 90.90
      * Pesos para SAPR11: 9.09
      * RETORNO: 5758.49 <br><br>  
Os resultados do método Simulated Annealing foram levemente menores que os resultados do Hill Climb, isso pode-se dever a falta de profundidade na hora da procura dos pesos, para seguintes oportunidades poderiamos colocar um maior número de iterações para conseguir aproveitar ao máximo o potencial dos algoritmos de otimização.

PREVISÃO DE PREÇOS:<br>
Neste ponto aplicaremos tanto a simulação de Monte Carlo e Arima para comparação de resultados com o algoritmo de aprendizagem profunda Long Short Term Memory. 
