🎙️ DSP Project – Audio Encoding & Decoding with Tkinter GUI
This project demonstrates a digital communication system that encodes and decodes text messages using frequency-based modulation techniques. Built with Python, it uses Fourier Transform and Bandpass Filtering to analyze and decode audio signals into readable text. A Tkinter-based GUI allows easy interaction for encoding, decoding, and visualizing signals.

🧠 How It Works
🔤 Encoding:
Each lowercase letter and space is represented by three distinct frequencies (from a predefined dictionary).

Cosine waves are generated using these frequencies to form an audio signal for each character.

The full encoded signal is concatenated with zeros between characters to help in separation during decoding.

🎧 Decoding:
Fourier Transform (FFT):

Converts the audio signal to the frequency domain.

Detects the three dominant frequencies in each segment to recover characters.

Bandpass Filtering:

Applies filters to isolate the frequency bands.

Uses argmax on the filtered spectrum to identify each character.

🖥 GUI Features
🎛 Encoder Section:
Enter text (lowercase letters and spaces only)

Encode to an audio signal

Play the encoded signal

Save the signal as a .wav file

Visualize the Fourier Transform of the signal

🧠 Decoder Section:
Upload a .wav file

Decode using:

Fourier Transform

Bandpass Filtering

View decoded text for both methods

📁 File Structure
plaintext
Copy
Edit
|-- dsp_audio_encoder_decoder.py   # Main application
|-- README.md                      # Description and instructions
|-- requirements.txt               # Dependencies (optional)
📸 Screenshots (optional)
(Insert screenshots of the GUI: encoding, playing audio, decoding, and Fourier plot)

📦 Requirements
Python 3.x

numpy

scipy

sounddevice

soundfile

matplotlib

tkinter (usually included with Python)

Install using:

bash
Copy
Edit
pip install numpy scipy sounddevice soundfile matplotlib 
🚀 How to Run
bash
Copy
Edit
python dsp_audio_encoder_decoder.py
Then follow the GUI instructions to:

Encode → Play/Save audio

Decode → Recover original text

📚 Educational Context
This project was built as part of a Digital Signal Processing (DSP) course to explore audio signal modulation, Fourier analysis, and filtering techniques.
