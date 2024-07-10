import librosa
import os 
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.utils import to_categorical
import gradio as gr
emotion_mapping = {
    '01' : 'Neutral',
    '02' : 'Calm',
    '03' : 'Happy',
    '04' : 'Sad',
    '05' : 'Angry',
    '06' : 'Disgusted',
    '07' : 'Surprised',
}

data_dir = "Dataset"

X_train = []
y_train = []


for root, _, files in os.walk(data_dir):
    for filename in files:
        if filename.endswith(".wav"):
            filepath = os.path.join(root, filename)
            emotion = filename.split("-")[2]
            
            y, sr = librosa.load(filepath, sr = None)
            
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            mfccs_len = 1000
            mfccs_padded = librosa.util.pad_center(mfccs, size = mfccs_len)
            
            X_train.append(mfccs_padded)
            y_train.append(int(emotion) -1)

X_train = np.array(X_train)
y_train = np.array(y_train)

num_classes = len(emotion_mapping)
y_train = np.clip(y_train, 0, num_classes -1)

scaler =  StandardScaler()
X_train = np.array([scaler.fit_transform(x.T).T for x in X_train])

X_train = X_train[..., np.newaxis]


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= 0.2, random_state = 42)

y_train = to_categorical(y_train, num_classes=len(emotion_mapping))
y_val = to_categorical(y_val, num_classes=len(emotion_mapping))


model = Sequential()            
model.add(Conv2D(32, (3,3), activation = "relu", input_shape = (X_train.shape[1], X_train.shape[2], 1)))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dense(len(emotion_mapping), activation ="softmax"))

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ['accuracy'])    

model.fit(X_train, y_train, epochs = 10, validation_data = (X_val, y_val))

def Predict_emotion(audio_file):
    y, sr = librosa.load(audio_file, sr = None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = 20)
    mfccs_padded = librosa.util.pad_center(mfccs, size = mfccs_len)
    
    mfccs_padded = scaler.transform(mfccs_padded.T).T
    mfccs_input = mfccs_padded[np.newaxis, ..., np.newaxis]
    
    predictions = model.predict(mfccs_input)
    predicted_index = np.argmax(predictions[0])
    predicted_emotion = emotion_mapping[str(predicted_index + 1).zfill(2)]
    
    return predicted_emotion



#new_audio_file = r"C:\Users\MORE SIR\Desktop\NULLCLASS\Project\Dataset\02F\03-01-01-01-01-01-04.wav"
#predicted_emotion = Predict_emotion(new_audio_file)
#print(f"predicted emotion :  {predicted_emotion}")



def predict_emotion_from_file(file_path):
    return Predict_emotion(file_path)


interface = gr.Interface(
    fn = predict_emotion_from_file,
    inputs = gr.Audio(type="filepath", label="Upload Audio"),
    outputs = "text",
    title = "---AUDIO---",
    description = "Upload An Audio File To Predict The Emotion"
)

if __name__ =="__main__":
    interface.launch()