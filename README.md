# PDS - Processamento Digital de Sinais

Este projeto consiste em uma coleção de scripts Python voltados para o processamento digital de sinais, com ênfase em filtragem e equalização de sinais e imagens.

## Objetivo do Projeto

Fornecer ferramentas e exemplos práticos para o processamento digital de sinais, utilizando técnicas de filtragem e equalização. Os scripts aqui presentes são destinados a profissionais e estudantes que desejam aplicar essas técnicas em projetos de áudio e imagem.

## Scripts Disponíveis

### 1. eq5_git.py

Este script implementa uma equalização por bibliotecas para sinais de áudio. O script permite ajustar os ganhos de diferentes bandas de frequência, aplicando filtros adequados para modificar a resposta em frequência do sinal de entrada.

### 2. eq7_git.py

Semelhante ao eq5_git.py, este script implementa uma equalização manual sobre as bandas de frequência.

### 3. filtragem_imagens_v3_git.py

Este script aplica técnicas de filtragem em imagens, utilizando a Transformada Discreta de Fourier (DFT) e convolução clássica. As imagens são transformadas para o domínio da frequência, onde diferentes filtros podem ser aplicados, como filtros passa-baixa, passa-alta, entre outros. Após o processamento, a imagem filtrada é revertida para o domínio espacial.

## Como Usar

### Equalizadores (eq5_git.py e eq7_git.py)
- Execute o script correspondente:
    ```
    streamlit run eq5_git.py
    ```
- Siga as instruções para selecionar as bandas de frequência e aplicar os ganhos desejados.

### Filtragem de Imagens (filtragem_imagens_v3_git.py)
- Execute o script:
    ```
    python filtragem_imagens_v3_git.py
    ```
- Forneça a imagem de entrada e selecione o filtro desejado para aplicar no domínio da frequência.

## Contribuição

Contribuições são bem-vindas! Se você deseja melhorar ou adicionar novas funcionalidades, por favor, abra um pull request.

## Licença

Este projeto está licenciado sob a Licença MIT. Consulte o arquivo LICENSE para obter mais informações.
