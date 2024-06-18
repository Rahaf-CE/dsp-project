from tkinter import Frame, Button, Label, Entry, StringVar, RAISED, font
import tkinter as tk
from tkinter import filedialog as fd, messagebox
import sounddevice as sd
import numpy as np
from scipy import fftpack
import wave
import scipy.io as sio
import sounddevice
import time
import scipy.signal as signal
import soundfile as sf
from tkinter import *
import matplotlib.pyplot as plt

# The `frq` dictionary is a key component in an audio encoding scheme, representing a character with three distinct frequencies.
# These frequencies are used to generate unique audio patterns, ensuring robustness and reducing errors during decoding.
# The encoding mechanism creates cosine waves at each frequency, preserving spaces between words.
# The chosen frequencies fall within specific frequency ranges.
frq = {
    'a': [100, 1100, 2500],
    'b': [100, 1100, 3000],
    'c': [100, 1100, 3500],
    'd': [100, 1300, 2500],
    'e': [100, 1300, 3000],
    'f': [100, 1300, 3500],
    'g': [100, 1500, 2500],
    'h': [100, 1500, 3000],
    'i': [100, 1500, 3500],
    'j': [300, 1100, 2500],
    'k': [300, 1100, 3000],
    'l': [300, 1100, 3500],
    'm': [300, 1300, 2500],
    'n': [300, 1300, 3000],
    'o': [300, 1300, 3500],
    'p': [300, 1500, 2500],
    'q': [300, 1500, 3000],
    'r': [300, 1500, 3500],
    's': [500, 1100, 2500],
    't': [500, 1100, 3000],
    'u': [500, 1100, 3500],
    'v': [500, 1300, 2500],
    'w': [500, 1300, 3000],
    'x': [500, 1300, 3500],
    'y': [500, 1500, 2500],
    'z': [500, 1500, 3000],
    ' ': [500, 1500, 3500],
}

#########################
num_zeros = 200
signal_duration = 0.04  # seconds
fs = 8000  # samplin freq
ffts = 1024  # fft sampling frequency
num_samples = int(signal_duration * fs)


def bp(d, l, h, fs, order=5):
    nyq = 0.5 * fs  # Nyquest Frequency
    low = l / nyq  # Normalizes the lower cutoff frequency by dividing it by the Nyquist frequency.
    high = h / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, d)
    return y


class Window(Frame):
    def __init__(self, master=None):
        super().__init__(master, bg='pink')  # Set the background color of the main window to violet
        self.master = master
        # Label at the top of the window
        header_label = tk.Label(root, text="بسم الله الرحمن الرحيم", font=('Arial', 20, 'bold'), fg='black', bg='pink')
        header_label.pack(side='top', pady=10)
        header_label = tk.Label(root, text="DSP project", font=('Arial', 18,), fg='black', bg='pink')
        header_label.pack(side='top', pady=10)
        # Frame for the Encoder section
        encoder_frame = tk.Frame(root, bg='pink', padx=10, pady=10)
        encoder_frame.pack(side='left', anchor='nw', padx=10, pady=10)

        # Title for Encoder section
        title_label = tk.Label(encoder_frame, text="Encoder", font=('Arial', 18, 'bold'), bg='purple', width=20)
        title_label.pack(pady=10)

        # Entry and buttons for Encoder section
        text_prompt_label = tk.Label(encoder_frame, text="Enter string below:", font=('Arial', 12), bg='lightgray',
                                     width=20)
        text_prompt_label.pack(pady=5)

        self.entry = tk.Entry(encoder_frame, width=40)
        self.entry.pack(pady=10)

        convert_button = tk.Button(encoder_frame, text="Encode", command=self.encode, bg='lightblue', fg='black',
                                   width=20)
        convert_button.pack(pady=5)

        play_button = tk.Button(encoder_frame, text="Play Audio", command=self.play_signal, bg='lightblue', fg='black',
                                width=20)
        play_button.pack(pady=5)

        save_button = tk.Button(encoder_frame, text="Save as .WAV file", command=self.save_audio_files, bg='lightblue',
                                fg='black', width=20)
        save_button.pack(pady=5)
        self.decode_prompt_text_label = StringVar()
        self.decode_resu = StringVar()
        self.decode_resu2 = StringVar()
        plot_button = tk.Button(encoder_frame, text="Plot Fourier Transform", command=self.plot_fourier_transform,
                                bg='lightblue', fg='black', width=20)
        plot_button.pack(pady=5)

        # Frame for the Decoder section
        decoder_frame = tk.Frame(root, bg='pink', padx=10, pady=10)
        decoder_frame.pack(side='right', anchor='ne', padx=10, pady=10)

        # Title for Decoder section
        decode_prompt_label = tk.Label(decoder_frame, text="Decoder", font=('Arial', 18, 'bold'), bg='purple', width=20)
        decode_prompt_label.pack(pady=5)

        decode_prompt_label_FilePath = tk.Label(decoder_frame, textvariable=self.decode_prompt_text_label,
                                                font=('Arial', 8), bg='lightgray', width=20)
        decode_prompt_label_FilePath.pack(pady=5)

        run_decoding_button = tk.Button(decoder_frame, text="Run Decoding Using \nFourier Transform",
                                        command=self.decode, bg='lightblue', fg='black', width=20)
        run_decoding_button.pack(pady=5)

        result_label = tk.Label(decoder_frame, textvariable=self.decode_resu, font=('Arial', 12), wraplength=380)
        result_label.pack(pady=5)

        run_decoding_bandpass_button = tk.Button(decoder_frame, text="Run Decoding Using Bandpass Filters",
                                                 command=self.pandBass_decode, bg='lightblue', fg='black', width=20)
        run_decoding_bandpass_button.pack(pady=5)

        result_label_bandpass = tk.Label(decoder_frame, textvariable=self.decode_resu2, font=('Arial', 12),
                                         wraplength=380)
        result_label_bandpass.pack(pady=5)

        footer_label = tk.Label(decoder_frame, text="Tala Abahra\n Rahaf Naser\n Ibaa taleeb", font=('Arial', 14),
                                fg='black', bg='pink')
        footer_label.pack(pady=5)

        self.is_encoded = False

    ##____________________________________________________________________________________
    # 1) first methoud for encode massage using sampiler to convert it to discrate massage>
    def encode(self):
        # Initialize an empty list to store the encoded audio data.
        self.l = []
        # Retrieves the input string from a user interface element (e.g., text entry field).
        stri = self.entry.get()

        # The block iterates through each character in the input string, checking for uppercase letters, digits, or non-alphabetic, non-space characters, and then prints the result.
        if any(char.isupper() or char.isdigit() or (not char.isalpha() and not char.isspace()) for char in stri):
            # If such characters are found, it shows an error message to the user and exits the function.
            messagebox.showerror("Invalid Input", "Please enter only lowercase letters and spaces.")
            return

        # This loop processes each character in the input string to encode it into an audio signal.
        for i in stri:
            # The code adds a zero sequence to'self.l', defining its length as 'num_zeros', which can indicate a pause or separation between encoded characters in the audio signal.
            self.l = np.concatenate((self.l, np.zeros(num_zeros)))

            # The code generates an audio wave for each character by combining cosine waves from a dictionary 'frq', where each key represents a character and the value is a tuple of three frequencies.
            # The range function generates a series of samples, with 'n/fs' determining the time point for each sample,
            # resulting in a unique audio pattern for each character in the input string.
            self.l = np.concatenate((self.l, [np.cos(frq[i][0] * 2 * np.pi * n / fs) +
                                              np.cos(frq[i][1] * 2 * np.pi * n / fs) +
                                              np.cos(frq[i][2] * 2 * np.pi * n / fs)
                                              for n in range(num_samples)]), axis=None)

    # ---------------------------------------------------------------------------------------------------------------------
    # 2)# Function to play audio without saving
    def play_signal(self):
        sounddevice.play(self.l, fs)
        time.sleep(1)

    # ____________________________________________________________________________________________________________________
    # 3) this just to save the audio in Wav ....
    def save_audio_files(self):
        # The function saves processed audio data into a.wav file, prompting the user to choose the file name and location.
        # The 'file_types' list defines available file formats, including 'Wave files' with '*.wav' extension and a general 'All files' option with '*.*' extension.
        file_types = [("Wave files", "*.wav"), ("All files", "*.*")]

        # Opens a file dialog window for the user to specify the file name and location for the saved audio file.
        # The dialog defaults to saving files with a '.wav' extension, and the user can choose from the defined file types.
        file_path = fd.asksaveasfilename(defaultextension=".wav", filetypes=file_types)

        # Checks if the user has canceled the file dialog. If the user cancels, the function returns early without saving the file.
        if not file_path:
            return

        # The code saves audio data from'self.l' to a.wav file at the specified 'file_path', using the'sf.write'
        # function from the'soundfile' library, containing the audio signal data.
        sf.write(file_path, self.l, fs)

    # ________________________________________________________________________________________________________________
    # 4) plot the massage and signal
    def plot_fourier_transform(self):
        _, amplitudes = self.perform_fourier_transform(self.l)
        plt.figure(figsize=(8, 4))
        plt.plot(_, amplitudes)
        plt.title("Fourier Transform of Encoded Signal")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()
        # The `perform_fourier_transform` function computes the Fast Fourier Transform (FFT) of an audio segment,
        # converting it from the time domain to the frequency domain.
        # It normalizes the result and returns only the first half of the symmetric frequency spectrum,
        # providing the frequencies and their corresponding magnitudes in the segment. This is used to analyze the frequency content of the audio data.

    def perform_fourier_transform(self, segment):
        n = len(segment)
        k = np.arange(n)
        T = n / fs
        frq = k / T
        frq = frq[range(n // 2)]  # one side frequency range
        Y = np.fft.fft(segment) / n  # FFT computing and normalization
        Y = Y[range(n // 2)]
        return frq, np.abs(Y)

    # ---------------------------------------------------------------------------------------------------
    def decode(self):
        name = fd.askopenfilename()
        if name != '':
            self.decode_prompt_text_label.set(name)
            samplerate, self.l = sio.wavfile.read(name)
        decoded_text = ""
        zeros = np.argwhere(self.l == 0).ravel()
        conseczeros = self.ArrayCountZero(zeros)
        ######################################################
        for i in range(0, len(self.l), num_samples + max(conseczeros)):  # for each letter
            freqMag = abs(fftpack.fft(self.l[i:i + num_samples],
                                      ffts))  # find the magnitude of the signal in the frequency domain
            maxpointsFreq = [(int(np.argmax(freqMag[int(100 * (ffts / fs)):int(500 * (ffts / fs))]) + 100 * (
                        ffts / fs)) * (fs / ffts)), int((fs / ffts) * (
                        np.argmax(freqMag[int(1100 * (ffts / fs)):int(1500 * (ffts / fs))]) + 1100 * (ffts / fs))),
                             int((fs / ffts) * (
                                         np.argmax(freqMag[int(2500 * (ffts / fs)):int(3500 * (ffts / fs))]) + 2500 * (
                                             ffts / fs)))]
            # argmax find the index of the max value
            for j in 0, 1, 2:
                maxpointsFreq[j] = int(round(maxpointsFreq[
                                                 j] / 100) * 100)  # round them up to the values of letters above (even if the value is 520, it should be rounded to 500 and so on)
            if maxpointsFreq in frq.values():  # if the three frequencies found exist in the  freq dictionary defined above
                for x, y in frq.items():
                    if y == maxpointsFreq:
                        decoded_text = decoded_text + x  # when you find that the array is equal to the one we found, add the array's key to the string
                        self.decode_resu.set(decoded_text)

    # The `ArrayCountZero` function calculates the lengths of consecutive zero sequences in an array.
    # It iterates through the array, counts consecutive zeros, and stores the counts in a list, which it then returns.
    # This is useful for identifying continuous silent periods in an array, typically representing silence in audio data.
    def ArrayCountZero(self, zeros):
        zeroA = 0
        conseczeros = []
        for i in range(len(zeros) - 1):  # algorithm to see the length of consecutive zeros
            if zeros[i] + 1 == (zeros[i + 1]):
                zeroA = zeroA + 1
            elif zeroA != 0:
                conseczeros.append(zeroA + 1)
                zeroA = 0
        return conseczeros

    # _________________________________________________________________________________________________________________

    def pandBass_decode(self):
        # Holds the decoded result as a string
        decode_resu = ""
        self.l = np.trim_zeros(self.l)
        # Finds indices of zeros in the audio data, indicating silence
        zeros = np.argwhere(self.l == 0).ravel()
        # Counts the number of consecutive zeros for identifying silent periods
        arrzero = self.ArrayCountZero(zeros)
        # Defines frequency bands for filtering the audio signal
        frequency_bands = [(100, 500), (1100, 1500), (2500, 3500)]
        # The frequency ranges used in audio decoding are (100, 500), (1100, 1500), and (2500, 3500).
        # The lower range targets lower frequencies, the mid-range captures higher frequencies within the lower mid-range,
        # and the higher range identifies elements in the upper spectrum. Analyzing these bands helps decode different components of the audio signal.
        # Processes audio data in chunks, determined by sample size and maximum count of consecutive zeros
        for segment_start in range(0, len(self.l), num_samples + max(arrzero)):
            # Resets the frequency count for each audio segment
            letterfrq = [0, 0, 0]

            for band_index, (lowcut, highcut) in enumerate(frequency_bands):
                # Calculates the end index of the current segment
                segment_end = segment_start + num_samples
                # Extracts the current segment from the audio data
                segment = self.l[segment_start:segment_end]

                # Filters the segment with a bandpass filter and applies FFT
                x = bp(segment, lowcut, highcut, fs, order=1)
                # Finds the dominant frequency in the FFT result
                letterfrq[band_index] = np.argmax(abs(fftpack.fft(x, ffts))) * (fs / ffts)

            # Rounds the frequencies to the nearest 100 and stores them
            rounded_letterfrq = [int(round(freq / 100) * 100) for freq in letterfrq]

            # The code decodes an audio signal into meaningful information by checking if the set of rounded frequencies (rounded_letterfrq) matches a predefined frequency pattern.
            # The variable 'frq' is a dictionary with each key representing a character and corresponding frequency list.
            # 'rounded_letterfrq' is a list of dominant frequencies in the current audio segment, rounded to the nearest 100 Hz.
            # The 'if' statement checks if the rounded frequency pattern of the current segment exists in the 'frq' dictionary's values. This is to see if the observed pattern matches any known character encoding.
            if rounded_letterfrq in frq.values():
                # If a match is found, the code iterates over the 'frq' dictionary items to find which character corresponds to the matched frequency pattern.
                for character, pattern in frq.items():
                    # Compares the current pattern from 'frq' with the rounded frequency pattern from the audio segment.
                    if pattern == rounded_letterfrq:
                        # If a match is found, the corresponding character is appended to the 'decode_resu' string. This string accumulates the decoded characters, gradually forming the final decoded message.
                        decode_resu += character
                        # Once a match is found and the character is appended, the loop breaks to process the next segment of audio data.
                        break

        # Stores the final decoded result
        self.decode_resu2.set(decode_resu)


# _________________________________________________________________________________________________________________

# GUI window

root = Tk()
app = Window(root)
root.title("DSP project")
root.configure(bg='pink')
root.geometry("600x500")

root.mainloop()

