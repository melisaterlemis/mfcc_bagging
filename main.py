from glob import glob
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

datasets_dir = "TurEV-DB-master\\Sound Source\\"
input_folder_list = glob(datasets_dir+"*")

max_len = 250

audio_inputs = []
audio_targets = []
target_id = -1
for input_folder in input_folder_list:
    waw_files = glob(input_folder + "\\*")
    target_id += 1
    for waw_file in waw_files:
        sampling_freq, audio = wavfile.read(waw_file)
        mfcc_feat = mfcc(audio, sampling_freq, nfft=2048)
        number_of_pad = max_len - mfcc_feat.shape[0]
        paddings = np.zeros((number_of_pad, mfcc_feat.shape[1]))
        mfcc_feat = np.concatenate((mfcc_feat, paddings))
        audio_inputs.append(mfcc_feat)
        audio_targets.append(target_id)

audio_inputs = np.array(audio_inputs)
audio_targets = np.array(audio_targets)

audio_inputs = np.reshape(audio_inputs, (audio_inputs.shape[0], -1))

x_train, x_test, y_train, y_test = train_test_split(audio_inputs, audio_targets, test_size=0.2)



max_sampels_array = [0.1, 0.3, 0.5, 0.8, 0.9]
max_features_array = [0.2, 0.4, 0.6, 0.7, 0.9]


best_max_samples = 0.0
best_max_features = 0.0
best_score = 0.0


for max_samples in max_sampels_array:
    for max_features in max_features_array:
        base_model = DecisionTreeClassifier()
        bagging_model = BaggingClassifier(base_model,
                                          max_samples=max_samples,
                                          max_features=max_features)

        bagging_model.fit(x_train, y_train)
        score = bagging_model.score(x_test, y_test)
        print("Score: ", score)
        if score > best_score:
            best_score = score
            best_max_samples = max_samples
            best_max_features = max_features
print("Best Test Score: {:.2f}".format(best_score))
print("Best max-samples", best_max_samples)
print("best max-features", best_max_features)
