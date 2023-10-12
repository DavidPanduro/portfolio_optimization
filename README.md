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
   * Portfolio_BP 21%
Claramente, a maior taxa de volatidade, maior risco, maiores opçõesde maior taxa de retorno.<br><br>

RISCO NÃO SISTEMÁTICO DOS PORTFÓLIOS:<br>
Este tipo de risco, responde a eventos específicos na empresa e depende do tipo de empresa. Em contrapartida com o risco sistemático, que responde a eventos externos, não pode ser eliminado e afeta a todas as empresas, podem ser resultado de eventos como recessão da economia (neste estudo não abordaremos este tipo de riscos).<br>
Para o cãlculo deste risco não sistemático, precisamos definir os pesos, calcular a variância anual e multiplicar pelos pesos, calcular a substração das variâncias e por último o <br>risco não sistemático é a **((variância do portfólio) - (substração das variâncias))** <br><br>
* Portfolio_TN sai na frente com 0.22
* Portfolio_LB que é o mais próximo com 0.10
* Portfolio_BP recua com 0.075<br><br>
Podemos comparar essa métrica com a Volatilidade e inclusive usar ambas métricas para analisar o risco dos portfólios.<br><br>

Algoritmos de Otimização:<br>
No estudo, aplicaremos os seguintes métodos:
1. Aleátorização.
2. Hill Climb.
3. Simulated Anneling

Previsão de Preços:<br>
Neste ponto aplicaremos tanto a simulação de Monte Carlo e Arima para comparação de resultados com o algoritmo de aprendizagem profunda Long Short Term Memory. 

Finalmente, apresentaremos um quandro de resumo contendo os resultados obtidos assim como os gráficos dos nossos rendimentos simulados.
