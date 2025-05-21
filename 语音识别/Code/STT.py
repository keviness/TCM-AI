# ================== 导入模块 ==================
import speech_recognition as sr
import numpy as np
import librosa
import io
from sklearn.ensemble import RandomForestClassifier
import joblib

# ================== 特征提取函数 ==================
def extract_features(audio_data, sample_rate):
    # MFCC
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    # 能量
    energy = np.mean(librosa.feature.rms(y=audio_data))
    # 过零率
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio_data))
    # 谱质心
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))
    # 谱带宽
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate))
    # 谱滚降
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate))
    # 音高（基频）
    pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
    pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    # 合并所有特征
    features = np.concatenate([
        mfccs_mean,
        [energy, zero_crossing_rate, spectral_centroid, spectral_bandwidth, spectral_rolloff, pitch]
    ])
    return features

# ================== 模型加载函数 ==================
def load_model(model_path="tcm_constitution_model.pkl"):
    try:
        model = joblib.load(model_path)
    except Exception:
        # 未训练模型，仅作接口演示
        model = RandomForestClassifier()
        model.n_classes_ = 5
        model.n_features_in_ = 19  # 13 MFCC + 6 其他特征
        model.classes_ = np.array([0, 1, 2, 3, 4])
    return model

# ================== 主逻辑 ==================
def main():
    recognizer = sr.Recognizer()
    model = load_model()
    with sr.Microphone() as source:
        print("请描述您的身体状况（采集音色和声学特征，进行体质判断）...")
        audio = recognizer.listen(source)
        print("特征提取中...")
        try:
            wav_data = audio.get_wav_data()
            audio_np, sr_librosa = librosa.load(io.BytesIO(wav_data), sr=None)
            features = extract_features(audio_np, sr_librosa)
            print("声学特征向量:", features)
            # 送入模型进行预测
            features_reshaped = features.reshape(1, -1)
            pred = model.predict(features_reshaped)[0]
            label_map = {0: "阳虚体质", 1: "阴虚体质", 2: "气虚体质", 3: "痰湿体质", 4: "平和体质"}
            constitution = label_map.get(pred, "无法判断")
            print("基于声学特征的中医体质初步判断：", constitution)
        except Exception as e:
            print("特征提取或体质判断失败:", e)

if __name__ == "__main__":
    main()