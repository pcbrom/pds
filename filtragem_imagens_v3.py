import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from PIL import Image

# Função para ler uma imagem PNG e converter para escala de cinza
def ler_imagem_png(caminho):
    with open(caminho, 'rb') as f:
        imagem = Image.open(f)
        imagem = imagem.convert('L')  # Converte para escala de cinza
        largura, altura = imagem.size
        imagem_cinza = np.array(imagem)
    return largura, altura, imagem_cinza

# Função para exibir a imagem original e a imagem filtrada
def exibir_imagens(imagem_original, imagem_filtrada, titulo_filtrada):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Imagem Original')
    plt.imshow(imagem_original, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title(titulo_filtrada)
    plt.imshow(imagem_filtrada, cmap='gray')
    plt.axis('off')
    plt.show()

# Função auxiliar para realizar a convolução em uma única linha da imagem
"""
\[
O(i, j) = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} P(i+m, j+n) \cdot K(m, n)
\]
"""
def convolve_line(args):
    i, padded_image, kernel, image_width, kernel_height, kernel_width = args
    output_line = np.zeros(image_width)
    for j in range(image_width):
        region = padded_image[i:i+kernel_height, j:j+kernel_width]
        output_line[j] = np.sum(region * kernel)
    return i, output_line

# Função para realizar a convolução 2D manualmente com paralelismo
def convolve2d(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    padded_image = np.pad(image, ((pad_height, pad_width), (pad_height, pad_width)), mode='constant')
    output_image = np.zeros_like(image)
    
    args = [(i, padded_image, kernel, image_width, kernel_height, kernel_width) for i in range(image_height)]
    
    with Pool(cpu_count()) as pool:
        results = pool.map(convolve_line, args)
    
    for i, output_line in results:
        output_image[i, :] = output_line
    
    return output_image


# Carregar e converter a imagem PNG para escala de cinza
path = 'sua/pasta/no/computador'
img_path = path + '/sunset.png'
largura, altura, imagem = ler_imagem_png(img_path)



# CONVOLUÇÕES -----------------------------------------------------------------


# 1. Aplique um filtro passa baixas com um filtro de médias

# Criando o filtro de médias 3x3
filtro_media_3x3 = np.array([[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]], dtype=np.float32) / 9.0

# Aplicar o filtro na imagem usando a convolução 2D manual
imagem_filtrada = convolve2d(imagem, filtro_media_3x3)

# Exibir a imagem original e a imagem filtrada
exibir_imagens(imagem, imagem_filtrada, 'Imagem Filtrada - Passa-Baixas 3x3')



# 2. Aplique também filtros de média maior, como o de 5x5

# Criando o filtro de médias 5x5
filtro_media_5x5 = np.array([[1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1]], dtype=np.float32) / 25.0

# Aplicar o filtro na imagem usando a convolução 2D manual
imagem_filtrada = convolve2d(imagem, filtro_media_5x5)

# Exibir a imagem original e a imagem filtrada
exibir_imagens(imagem, imagem_filtrada, 'Imagem Filtrada - Passa-Baixas 5x5')



# 3. Aplique também filtro 7x7

# Criando o filtro de médias 7x7
filtro_media_7x7 = np.array([[1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1]], dtype=np.float32) / 49.0

# Aplicar o filtro na imagem usando a convolução 2D manual
imagem_filtrada = convolve2d(imagem, filtro_media_7x7)

# Exibir a imagem original e a imagem filtrada
exibir_imagens(imagem, imagem_filtrada, 'Imagem Filtrada - Passa-Baixas 7x7')



# 4. Faça filtros passa-alta correspondentes como sendo os inversos do passa-baixas

# Criando o filtro passa-alta 3x3
filtro_passa_alta_3x3 = np.array([[-1, -1, -1],
                                  [-1,  8, -1],
                                  [-1, -1, -1]], dtype=np.float32) / 9.0

# Aplicar o filtro na imagem usando a convolução 2D manual
imagem_filtrada = convolve2d(imagem, filtro_passa_alta_3x3)

# Exibir a imagem original e a imagem filtrada
exibir_imagens(imagem, imagem_filtrada, 'Imagem Filtrada - Passa-Alta 3x3')



# 5. Aplique h′5[n1, n2] à imagem

# Criando o filtro passa-alta 5x5
filtro_passa_alta_5x5 = np.array([[-1, -1, -1, -1, -1],
                                  [-1,  1,  2,  1, -1],
                                  [-1,  2,  4,  2, -1],
                                  [-1,  1,  2,  1, -1],
                                  [-1, -1, -1, -1, -1]], dtype=np.float32)

# Aplicar o filtro na imagem usando a convolução 2D manual
imagem_filtrada = convolve2d(imagem, filtro_passa_alta_5x5)

# Exibir a imagem original e a imagem filtrada
exibir_imagens(imagem, imagem_filtrada, 'Imagem Filtrada - Passa-Alta 5x5')



# 6. Aplique h′7[n1, n2] à imagem

# Criando o filtro passa-alta 7x7
filtro_passa_alta_7x7 = np.array([[-1, -1, -1, -1, -1, -1, -1],
                                  [-1,  1,  1,  1,  1,  1, -1],
                                  [-1,  1,  2,  2,  2,  1, -1],
                                  [-1,  1,  2,  4,  2,  1, -1],
                                  [-1,  1,  2,  2,  2,  1, -1],
                                  [-1,  1,  1,  1,  1,  1, -1],
                                  [-1, -1, -1, -1, -1, -1, -1]], dtype=np.float32)

# Aplicar o filtro na imagem usando a convolução 2D manual
imagem_filtrada = convolve2d(imagem, filtro_passa_alta_7x7)

# Exibir a imagem original e a imagem filtrada
exibir_imagens(imagem, imagem_filtrada, 'Imagem Filtrada - Passa-Alta 7x7')



# DFT -------------------------------------------------------------------------



# Função para calcular a DFT 2D manualmente usando operações matriciais
"""
\[
\text{DFT}(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} I(x, y) \cdot e^{-2\pi j \left(\frac{ux}{M} + \frac{vy}{N}\right)}
\]
"""
def dft2d(image):
    rows, cols = image.shape
    u = np.arange(rows).reshape((rows, 1))
    v = np.arange(cols).reshape((1, cols))
    x = np.arange(rows).reshape((rows, 1))
    y = np.arange(cols).reshape((1, cols))
    
    # Matriz de expoentes
    exp_factor = -2j * np.pi * (u @ x.T / rows + v @ y.T / cols)
    
    # Aplicar a DFT
    dft = np.dot(np.dot(np.exp(exp_factor), image), np.exp(exp_factor).T)
    
    return dft

# Função para calcular a IDFT 2D manualmente usando operações matriciais
"""
\[
I(x, y) = \frac{1}{M \cdot N} \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} F(u, v) \cdot e^{2\pi j \left(\frac{ux}{M} + \frac{vy}{N}\right)}
\]
"""
def idft2d(dft):
    rows, cols = dft.shape
    u = np.arange(rows).reshape((rows, 1))
    v = np.arange(cols).reshape((1, cols))
    x = np.arange(rows).reshape((rows, 1))
    y = np.arange(cols).reshape((1, cols))
    
    # Matriz de expoentes
    exp_factor = 2j * np.pi * (u @ x.T / rows + v @ y.T / cols)
    
    # Aplicar a IDFT
    idft = np.dot(np.dot(np.exp(exp_factor), dft), np.exp(exp_factor).T)
    idft = idft / (rows * cols)
    
    return idft

# Função para deslocar a DFT manualmente usando operações matriciais
"""
\[
\text{shifted\_DFT}(u, v) = 
\begin{cases} 
F(u + \frac{M}{2}, v + \frac{N}{2}) & \text{se } u < \frac{M}{2}, v < \frac{N}{2} \\
F(u - \frac{M}{2}, v + \frac{N}{2}) & \text{se } u \geq \frac{M}{2}, v < \frac{N}{2} \\
F(u + \frac{M}{2}, v - \frac{N}{2}) & \text{se } u < \frac{M}{2}, v \geq \frac{N}{2} \\
F(u - \frac{M}{2}, v - \frac{N}{2}) & \text{se } u \geq \frac{M}{2}, v \geq \frac{N}{2} 
\end{cases}
\]
"""
def shift_dft(dft):
    rows, cols = dft.shape
    half_rows, half_cols = rows // 2, cols // 2
    
    # Rearranjar os quadrantes
    shifted_dft = np.zeros_like(dft)
    shifted_dft[:half_rows, :half_cols] = dft[half_rows:, half_cols:]
    shifted_dft[:half_rows, half_cols:] = dft[half_rows:, :half_cols]
    shifted_dft[half_rows:, :half_cols] = dft[:half_rows, half_cols:]
    shifted_dft[half_rows:, half_cols:] = dft[:half_rows, :half_cols]
    
    return shifted_dft

# Função para deslocar de volta a DFT manualmente usando operações matriciais
"""
\[
\text{unshifted\_DFT}(u, v) = 
\begin{cases} 
F(u - \frac{M}{2}, v - \frac{N}{2}) & \text{se } u \geq \frac{M}{2}, v \geq \frac{N}{2} \\
F(u + \frac{M}{2}, v - \frac{N}{2}) & \text{se } u < \frac{M}{2}, v \geq \frac{N}{2} \\
F(u - \frac{M}{2}, v + \frac{N}{2}) & \text{se } u \geq \frac{M}{2}, v < \frac{N}{2} \\
F(u + \frac{M}{2}, v + \frac{N}{2}) & \text{se } u < \frac{M}{2}, v < \frac{N}{2}
\end{cases}
\]
"""
def unshift_dft(dft):
    rows, cols = dft.shape
    half_rows, half_cols = rows // 2, cols // 2
    
    # Rearranjar os quadrantes de volta
    unshifted_dft = np.zeros_like(dft)
    unshifted_dft[half_rows:, half_cols:] = dft[:half_rows, :half_cols]
    unshifted_dft[:half_rows, :half_cols] = dft[half_rows:, half_cols:]
    unshifted_dft[half_rows:, :half_cols] = dft[:half_rows, half_cols:]
    unshifted_dft[:half_rows, half_cols:] = dft[half_rows:, :half_cols]
    
    return unshifted_dft



# Realizando a Transformada de Fourier na imagem
dft = dft2d(imagem)
dft_shifted = shift_dft(dft)



# 7. Aplique um filtro passa-baixas no domínio da DFT com corte em ωc = π/2
rows, cols = imagem.shape
crow, ccol = rows // 2, cols // 2
filtro_passa_baixa = np.zeros((rows, cols), np.float32)
filtro_passa_baixa[crow-rows//4:crow+rows//4, ccol-cols//4:ccol+cols//4] = 1

dft_filtrado = dft_shifted * filtro_passa_baixa
dft_unshifted = unshift_dft(dft_filtrado)
imagem_filtrada = idft2d(dft_unshifted)
imagem_filtrada = np.abs(imagem_filtrada)

exibir_imagens(imagem, imagem_filtrada, 'Imagem Filtrada - Passa-Baixa DFT (ωc = π/2)')



# 8. Repita para ωc = π/4
filtro_passa_baixa = np.zeros((rows, cols), np.float32)
filtro_passa_baixa[crow-rows//8:crow+rows//8, ccol-cols//8:ccol+cols//8] = 1

dft_filtrado = dft_shifted * filtro_passa_baixa
dft_unshifted = unshift_dft(dft_filtrado)
imagem_filtrada = idft2d(dft_unshifted)
imagem_filtrada = np.abs(imagem_filtrada)

exibir_imagens(imagem, imagem_filtrada, 'Imagem Filtrada - Passa-Baixa DFT (ωc = π/4)')



# 9. Repita para ωc = π/8
filtro_passa_baixa = np.zeros((rows, cols), np.float32)
filtro_passa_baixa[crow-rows//16:crow+rows//16, ccol-cols//16:ccol+cols//16] = 1

dft_filtrado = dft_shifted * filtro_passa_baixa
dft_unshifted = unshift_dft(dft_filtrado)
imagem_filtrada = idft2d(dft_unshifted)
imagem_filtrada = np.abs(imagem_filtrada)

exibir_imagens(imagem, imagem_filtrada, 'Imagem Filtrada - Passa-Baixa DFT (ωc = π/8)')



# 10. Aplique um filtro passa-alta no domínio da DFT com corte em ωc = π/2
filtro_passa_alta = np.ones((rows, cols), np.float32)
filtro_passa_alta[crow-rows//4:crow+rows//4, ccol-cols//4:ccol+cols//4] = 0

dft_filtrado = dft_shifted * filtro_passa_alta
dft_unshifted = unshift_dft(dft_filtrado)
imagem_filtrada = idft2d(dft_unshifted)
imagem_filtrada = np.abs(imagem_filtrada)

exibir_imagens(imagem, imagem_filtrada, 'Imagem Filtrada - Passa-Alta DFT (ωc = π/2)')



# 11. Repita para ωc = π/4
filtro_passa_alta = np.ones((rows, cols), np.float32)
filtro_passa_alta[crow-rows//8:crow+rows//8, ccol-cols//8:ccol+cols//8] = 0

dft_filtrado = dft_shifted * filtro_passa_alta
dft_unshifted = unshift_dft(dft_filtrado)
imagem_filtrada = idft2d(dft_unshifted)
imagem_filtrada = np.abs(imagem_filtrada)

exibir_imagens(imagem, imagem_filtrada, 'Imagem Filtrada - Passa-Alta DFT (ωc = π/4)')



# 12. Repita para ωc = π/8
filtro_passa_alta = np.ones((rows, cols), np.float32)
filtro_passa_alta[crow-rows//16:crow+rows//16, ccol-cols//16:ccol+cols//16] = 0

dft_filtrado = dft_shifted * filtro_passa_alta
dft_unshifted = unshift_dft(dft_filtrado)
imagem_filtrada = idft2d(dft_unshifted)
imagem_filtrada = np.abs(imagem_filtrada)

exibir_imagens(imagem, imagem_filtrada, 'Imagem Filtrada - Passa-Alta DFT (ωc = π/8)')



# 13. Aplique um filtro passa-baixa no domínio da DFT com corte em ωc = π/8 apenas na direção horizontal
filtro_passa_baixa = np.zeros((rows, cols), np.float32)
filtro_passa_baixa[crow-rows//16:crow+rows//16, :] = 1  # Filtrando na direção horizontal

dft_filtrado = dft_shifted * filtro_passa_baixa
dft_unshifted = unshift_dft(dft_filtrado)
imagem_filtrada = idft2d(dft_unshifted)
imagem_filtrada = np.abs(imagem_filtrada)

exibir_imagens(imagem, imagem_filtrada, 'Imagem Filtrada - Passa-Baixa DFT (ωc = π/8) - Horizontal')



# 14. Aplique um filtro passa-alta no domínio da DFT com corte em ωc = π/8 apenas na direção horizontal
filtro_passa_alta = np.ones((rows, cols), np.float32)
filtro_passa_alta[crow-rows//16:crow+rows//16, :] = 0  # Filtrando na direção horizontal

dft_filtrado = dft_shifted * filtro_passa_alta
dft_unshifted = unshift_dft(dft_filtrado)
imagem_filtrada = idft2d(dft_unshifted)
imagem_filtrada = np.abs(imagem_filtrada)

exibir_imagens(imagem, imagem_filtrada, 'Imagem Filtrada - Passa-Alta DFT (ωc = π/8) - Horizontal')
