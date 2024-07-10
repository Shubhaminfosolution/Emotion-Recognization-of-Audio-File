import librosa
import librosa.display
import os
import numpy 
import matplotlib.pyplot as plt

"""audiofile = []

for filename in os.listdir(r"Actor_02"):    
    filepath = os.path.join(r"Actor_02",filename)
    
    print(filepath)"""
#we taken the file path of the audio
audio = r"C:\Users\MORE SIR\Desktop\NULLCLASS\Project\tasks\Actor_02\03-01-01-01-01-01-02.wav"
#y is the time series the sampling rate and the loading is done here...By default, all audio is mixed to mono and resampled to 22050 Hz at load time.
y, sr = librosa.load(audio)
plt.figure(figsize=(20,20))
librosa.display.waveplot(y,sr=sr)
plt.show(audio)
#here beat_frame is means the frame of audio of signal y which is seperated by hop_lenght = 512 and it is centered by the 5*ho_length
tempo, beat_frames = librosa.beat.beat_track(y=y , sr=sr)
#     print(beat_frames)
# in this step we have get the actual time of the windows(audio)..suppose we cuts the audio into seperate frame theb=nit matches with time
beat_times = librosa.frames_to_time(beat_frames)

print(beat_times)