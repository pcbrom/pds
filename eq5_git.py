# -*- coding: iso-8859-1 -*-

# Importar as bibliotecas necess‡rias
import numpy as np
import scipy.signal as signal
import soundfile as sf
import pyaudio
import streamlit as st

# Caminho do arquivo de ‡udio de entrada
file_path = './input.wav'

# Carregar o arquivo de ‡udio
audio_data, sample_rate = sf.read(file_path)

# Configura›es do PyAudio
CHUNK = 1024 * 30
FORMAT = pyaudio.paInt16
CHANNELS = 1 if len(audio_data.shape) == 1 else audio_data.shape[1]
RATE = sample_rate

# Frequncias de corte para os filtros (em Hz)
cutoffs = [32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000]

# Criar os filtros FIR utilizando a janela Hamming
filters = []
nyquist = 0.5 * RATE
for cutoff in cutoffs:
    norm_cutoff = cutoff / nyquist
    taps = signal.firwin(numtaps=101, cutoff=norm_cutoff, window="hamming")
    filters.append(taps)

# Inicializa a interface de ‡udio
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

# Fun‹o para formatar as barras de visualiza‹o
def format_frequency(cutoff):
    """ Formata a frequncia para usar 'k' em valores acima de 1000 Hz """
    if cutoff >= 1000:
        return f"{cutoff // 1000}k Hz"
    else:
        return f"{cutoff} Hz"

def update_bars(energies):
    """ Atualiza as barras de visualiza‹o de energia """
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
    scaled_energies = [energy / max_energy * 100 for energy in energies]  # Escala para o m‡ximo ser 100%

    colors = ['#FF5733', '#FF8D1A', '#FFC300', '#DAF7A6', '#33FF57', 
              '#1AFF8D', '#1AC3FF', '#3357FF', '#9A33FF', '#FF33A8']

    bars_content = ""
    for i, energy in enumerate(scaled_energies):
        formatted_cutoff = format_frequency(cutoffs[i])
        bars_content += bar_template.format(gains[i], energy, colors[i % len(colors)], formatted_cutoff)

    return bars_html.format(bars_content)

bars_placeholder = st.empty()

# Manter estado da posi‹o do ‡udio usando st.session_state
if 'audio_position' not in st.session_state:
    st.session_state.audio_position = 0

# Fun‹o Overlap-Add utilizando opera›es matriciais e FFT
def overlap_add_convolution_matrix(audio_chunk, filters, gains):
    # Converte os filtros para o dom’nio da frequncia
    filter_length = len(filters[0])
    chunk_length = len(audio_chunk)
    fft_length = 2**np.ceil(np.log2(chunk_length + filter_length - 1)).astype(int)
    
    filters_fft = np.array([np.fft.rfft(taps, n=fft_length) for taps in filters])
    
    if len(audio_chunk.shape) == 1:  # Mono
        audio_fft = np.fft.rfft(audio_chunk, n=fft_length)
        output_fft = np.sum([gains[i] * filters_fft[i] * audio_fft for i in range(len(filters))], axis=0)
        output_audio = np.fft.irfft(output_fft, n=fft_length)
    else:  # Multicanal (estŽreo)
        output_audio = np.zeros((fft_length, audio_chunk.shape[1]))
        for channel in range(audio_chunk.shape[1]):
            audio_fft = np.fft.rfft(audio_chunk[:, channel], n=fft_length)
            output_fft = np.sum([gains[i] * filters_fft[i] * audio_fft for i in range(len(filters))], axis=0)
            output_audio[:, channel] = np.fft.irfft(output_fft, n=fft_length)
    
    return output_audio[:chunk_length + filter_length - 1]

# Loop de processamento em tempo real
try:
    for i in range(st.session_state.audio_position, len(audio_data), CHUNK):
        st.session_state.audio_position = i  # Atualiza a posi‹o corrente
        audio_chunk = audio_data[i:i + CHUNK]

        if len(audio_chunk) < CHUNK:
            if len(audio_chunk.shape) == 1:  # Mono
                audio_chunk = np.pad(audio_chunk, (0, CHUNK - len(audio_chunk)), 'constant')
            else:  # EstŽreo
                audio_chunk = np.pad(audio_chunk, ((0, CHUNK - len(audio_chunk)), (0, 0)), 'constant')

        output_audio = overlap_add_convolution_matrix(audio_chunk, filters, gains)

        # Normalizar o sinal de sa’da
        max_val = np.max(np.abs(output_audio))
        if max_val > 0:
            output_audio = output_audio / max_val  # Apenas normaliza pelo valor m‡ximo para evitar clipping

        # Converter para o tipo de dado apropriado
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            output_audio = (output_audio * 32767).astype(np.int16)
        elif audio_data.dtype == np.int24:
            output_audio = (output_audio >> 8).astype(np.int16)
        else:
            output_audio = output_audio.astype(np.int16)

        # Limitar o valor m‡ximo para evitar clipping
        output_audio = np.clip(output_audio, -32767, 32767)

        # Escreve o ‡udio processado na sa’da
        stream_output.write(output_audio.tobytes())

        # Calcular energias para as barras de visualiza‹o
        energies = []
        for i, taps in enumerate(filters):
            filtered_audio = signal.fftconvolve(audio_chunk[:, 0] if len(audio_chunk.shape) > 1 else audio_chunk, taps, mode='valid')
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
