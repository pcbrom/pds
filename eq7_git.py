# -*- coding: iso-8859-1 -*-
import numpy as np
import soundfile as sf
import pyaudio
import streamlit as st

# Função para criar filtros FIR manualmente usando a janela de Hamming
def create_fir_filter(numtaps, cutoff, rate):
    nyquist = 0.5 * rate
    norm_cutoff = cutoff / nyquist
    M = numtaps - 1
    h = []
    for n in range(numtaps):
        if n == M / 2:
            h.append(norm_cutoff)
        else:
            h.append(norm_cutoff * (np.sin(2 * np.pi * norm_cutoff * (n - M / 2)) / (np.pi * (n - M / 2))))
        # Aplicar janela de Hamming
        h[n] *= 0.54 - 0.46 * np.cos(2 * np.pi * n / M)
    return h

# Implementação manual da FFT
def fft_manual(signal):
    N = len(signal)
    if N <= 1:
        return signal
    even = fft_manual(signal[0::2])
    odd = fft_manual(signal[1::2])
    T = [np.e**(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

def ifft_manual(signal):
    N = len(signal)
    signal_conj = [x.conjugate() for x in signal]
    fft_conj = fft_manual(signal_conj)
    return [x.conjugate() / N for x in fft_conj]

def overlap_add_convolution_matrix_manual(audio_chunk, filters, gains):
    # Define o comprimento do filtro e do chunk de áudio
    filter_length = len(filters[0])
    chunk_length = len(audio_chunk)
    fft_length = 2**int(np.ceil(np.log2(chunk_length + filter_length - 1)))

    # Converte os filtros para o domínio da frequência utilizando FFT manual
    filters_fft = []
    for taps in filters:
        taps_extended = taps + [0] * (fft_length - len(taps))
        filters_fft.append(fft_manual(taps_extended))
    
    # Processamento Mono ou Estéreo
    if len(audio_chunk.shape) == 1:  # Mono
        audio_chunk_extended = np.concatenate((audio_chunk, [0] * (fft_length - len(audio_chunk))))
        audio_fft = fft_manual(audio_chunk_extended)
        output_fft = [0] * len(audio_fft)
        for i in range(len(filters)):
            for j in range(len(output_fft)):
                output_fft[j] += gains[i] * (filters_fft[i][j] * audio_fft[j])
        output_audio = ifft_manual(output_fft)
    else:  # Multicanal (estéreo)
        output_audio = np.zeros((fft_length, audio_chunk.shape[1]), dtype=complex)
        for channel in range(audio_chunk.shape[1]):
            audio_chunk_extended = np.concatenate((audio_chunk[:, channel], [0] * (fft_length - len(audio_chunk[:, channel]))))
            audio_fft = fft_manual(audio_chunk_extended)
            output_fft = [0] * len(audio_fft)
            for i in range(len(filters)):
                for j in range(len(output_fft)):
                    output_fft[j] += gains[i] * (filters_fft[i][j] * audio_fft[j])
            output_audio[:, channel] = ifft_manual(output_fft)

    # Convertendo a lista para numpy array e retornando a parte real
    return np.array(output_audio[:chunk_length + filter_length - 1]).real

# Função para formatar as barras de visualização
def format_frequency(cutoff):
    """ Formata a frequência para usar 'k' em valores acima de 1000 Hz """
    if cutoff >= 1000:
        return f"{cutoff // 1000}k Hz"
    else:
        return f"{cutoff} Hz"

def update_bars(energies):
    """ Atualiza as barras de visualização de energia """
    bars_html = """
    <div style="display: flex; justify-content: space-around; align-items: flex-end;">
        {}
    </div>
    """

    bar_template = """
    <div style="width: 40px; height: 150px; position: relative; text-align: center;">
        <div style="position: absolute; top: -25px; width: 100%; font-size: 18px; font-weight: bold; color: white; text-shadow: 1px 1px 2px black;">
            {:.2f}
        </div>
        <div style="position: absolute; bottom: 0; width: 100%; height: {}%; background-color: {}; transition: height 0.1s;"></div>
        <div style="position: absolute; bottom: -30px; width: 100%; font-size: 16px; color: white; text-shadow: 1px 1px 2px black; white-space: nowrap;">
            {}
        </div>
    </div>
    """

    max_energy = max(energies) if max(energies) > 0 else 1
    scaled_energies = [energy / max_energy * 100 for energy in energies]  # Escala para o máximo ser 100%

    colors = ['#FF5733', '#FF8D1A', '#FFC300', '#DAF7A6', '#33FF57', 
              '#1AFF8D', '#1AC3FF', '#3357FF', '#9A33FF', '#FF33A8']

    bars_content = ""
    for i, energy in enumerate(scaled_energies):
        formatted_cutoff = format_frequency(cutoffs[i])
        bars_content += bar_template.format(gains[i], energy, colors[i % len(colors)], formatted_cutoff)

    return bars_html.format(bars_content)


# Carregar o arquivo de áudio
file_path = './input.wav'
audio_data, sample_rate = sf.read(file_path)

# Definir a duração do áudio a ser processado (5 segundos)
duration = 5  # em segundos
num_samples = int(sample_rate * duration)
audio_data = audio_data[:num_samples]

# Configurações do PyAudio
CHUNK = 1024 * 30
FORMAT = pyaudio.paInt16
CHANNELS = 1 if len(audio_data.shape) == 1 else audio_data.shape[1]
RATE = sample_rate

# Frequências de corte para os filtros (em Hz)
cutoffs = [32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000]

# Criar os filtros FIR manualmente
filters = []
for cutoff in cutoffs:
    taps = create_fir_filter(numtaps=101, cutoff=cutoff, rate=RATE)
    filters.append(taps)

# Inicializa a interface de áudio
p = pyaudio.PyAudio()
stream_output = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       output=True,
                       frames_per_buffer=CHUNK)

# Streamlit interface
st.markdown("<h1 style='font-size: 36px;'>Equalizador de Audio<br/></h1>", unsafe_allow_html=True)
st.sidebar.title('Controle de Ganhos')

# Sliders para controle de ganho
gains = []
for i, cutoff in enumerate(cutoffs):
    gain = st.sidebar.slider(f'Ganho {cutoff} Hz', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    gains.append(gain)

bars_placeholder = st.empty()

# Manter estado da posição do áudio usando st.session_state
if 'audio_position' not in st.session_state:
    st.session_state.audio_position = 0

# Loop de processamento em tempo real
try:
    for i in range(st.session_state.audio_position, len(audio_data), CHUNK):
        st.session_state.audio_position = i  # Atualiza a posição corrente
        audio_chunk = audio_data[i:i + CHUNK]

        if len(audio_chunk) < CHUNK:
            if len(audio_chunk.shape) == 1:  # Mono
                audio_chunk = np.pad(audio_chunk, (0, CHUNK - len(audio_chunk)), 'constant')
            else:  # Estéreo
                audio_chunk = np.pad(audio_chunk, ((0, CHUNK - len(audio_chunk)), (0, 0)), 'constant')

        output_audio = overlap_add_convolution_matrix_manual(audio_chunk, filters, gains)

        # Normalizar o sinal de saída
        max_val = np.max(np.abs(output_audio))
        if max_val > 0:
            output_audio = output_audio / max_val  # Apenas normaliza pelo valor máximo para evitar clipping

        # Converter para o tipo de dado apropriado
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            output_audio = (output_audio * 32767).astype(np.int16)
        elif audio_data.dtype == np.int24:
            output_audio = (output_audio >> 8).astype(np.int16)
        else:
            output_audio = output_audio.astype(np.int16)

        # Limitar o valor máximo para evitar clipping
        output_audio = np.clip(output_audio, -32767, 32767)

        # Escreve o áudio processado na saída
        stream_output.write(output_audio.tobytes())

        # Calcular energias para as barras de visualização
        energies = []
        for i, taps in enumerate(filters):
            filtered_audio = overlap_add_convolution_matrix_manual(audio_chunk[:, 0] if len(audio_chunk.shape) > 1 else audio_chunk, [taps], [1.0])
            energy = np.sum((filtered_audio * gains[i]) ** 2) / len(filtered_audio)
            energies.append(energy)

        # Normaliza e atualiza as barras CSS
        bars_html_content = update_bars(energies)
        bars_placeholder.markdown(bars_html_content, unsafe_allow_html=True)

except KeyboardInterrupt:
    pass

# Fechar os streams e o PyAudio
stream_output.stop_stream()
stream_output.close()
p.terminate()
