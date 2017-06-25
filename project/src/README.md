# Simulação de tracking da dinâmica dos elétrons em um acelerador síncrotron

Gustavo Ciotto Pinton - RA117136
Introdução à Programação Paralela

### Conteúdo

Nesta pasta, encontram-se todos as classes implementadas para a simulação da dinâmica dos elétrons. Cada uma possui as seguintes funcionalidades:

- Classe `DynamicSearch` (definida em `DynamicSearch.h` e implementada em `DynamicSearch.cpp`): classe abstrata base que fornece métodos e a estrutura necessários para diferentes tipos de implementações da simulação. O método `createRing()` cria um vetor de atuadores que realizarão as modificações nos estados dos elétrons de acordo com suas características. A função `testSolution(r)`, por sua vez, testa se o vetor `r` satisfaz algumas condições para se tornar uma solução válida. `performOneTurn(r)` realiza uma volta inteira no anel, isto é, realiza a passagem de `r` por todos os `M` atuadores do anel apenas uma vez, através de chamadas ao método `pass(r)` de cada atuador. Enfim, as implementações do método abstrato `dynamical_aperture_search()` devem, para cada condição inicial Xo, realizar `N` chamadas a `performOneTurn(r)`. `plot()` é apenas uma solução para salvar em um arquivo texto a norma de cada um dos vetores calculados após as `N` voltas.

- Classe `DefaultDynamicSearch` (definida em `DynamicSearch.h` e implementada em `DynamicSearch.cpp`): implementação sequencial da classe `DynamicSearch`. Não possui nenhum tipo de otimização, isto é, não foi utilizada nenhuma técnica de paralelização, tal como OpenMP ou CUDA. A implementação do método abstrato `dynamical_aperture_search()` possui dois loops externos, que iteram sobre as condições iniciais, e um loop interno, responsável por realizar as `N` voltas pelos atuadores.

- Classe `RingElement` (definida em `RingElement.h` e implementada em `RingElement.cpp`): classe abstrata que define a estrutura necessária a um atuador. O método `pass()`, abstrato, é chamado para simular a passagem de um elétron pelo respectivo atuador e, portanto, deve refletir sua física.

- Classes `Drift`, `Quadrupole` e `Sextupole` (definidas em `RingElement.h` e implementadas em `RingElement.cpp`): alguns modelos de atuadores, definindo a classe `RingElement`. Todos fornecem um implementação ao método `pass(r)`.

- Classe `CudaDynamicSearch` (definida em `CudaDynamicSearch.h` e implementada em `CudaDynamicSearch.cu`): implementação em CUDA da classe `DynamicSearch`. A implementação do método abstrato `dynamical_aperture_search()` aloca memória no dispositivo, copia todos os dados necessários para esta memória e lança a execução do kernel. `dynamicSearchKernel` representa a função __global__ executada pela GPU.

- Arquivos `SerialTracking.cpp` e `ParallelTracking.cpp`: funções `main` para os programas serial e paralelo, respectivamente. Internamente às funções `main`, ocorre a instanciação de um objeto `CudaDynamicSearch` ou `DefaultDynamicSearch` e seus métodos `dynamical_aperture_search()` e `plot()` são chamados.

### Entradas

Além dos elementos apresentados acima, uma pasta contendo algumas entradas de teste também é disponibilizada. Cada entrada de teste é composta por dois números inteiros
, M e N. M é a quantidade de vezes que um conjunto de atuadores se repete ao longo do anel e N, o número de voltas que cada elétron realiza.

### Execução

A execução pode ser realizada através de `make run`. Conforme descrito no relatório, duas implementações paralelas foram disponibilizadas, uma utilizando instruções FMA e a outra, funções intrínsicas que garantem que os resultados obtidos sejam idênticos aqueles da execução serial. A escolha de qual delas utilizar é feita através da constante `NVCCFLAGS` em `Makefile`. Para executar a implementação com FMA, adicione a esta constante `-DCUDA_FMA` e, para a tradicional, `-DCUDA_INTRINSICS`.

