"""
中医体质语音识别系统
依赖库安装：pip install sounddevice librosa scikit-learn numpy matplotlib
"""
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier

# ================== 配置参数 ==================
RECORD_DURATION = 5  # 录音时长(秒)
SAMPLE_RATE = 44100  # 采样率
MODEL_PATH = 'tcm_model.pkl'  # 模型保存路径

# 中医体质类型映射
TCM_TYPES = {
    0: "气虚质（建议补气健脾）",
    1: "阳虚质（建议温补阳气）",
    2: "阴虚质（建议滋阴润燥）",
    3: "痰湿质（建议化痰祛湿）",
    4: "湿热质（建议清热利湿）"
}

# ================== 核心功能 ==================
def record_audio(filename='patient_voice.wav'):
    """录制患者语音"""
    print(f"开始录音，请保持安静...（时长{RECORD_DURATION}秒）")
    try:
        recording = sd.rec(int(RECORD_DURATION * SAMPLE_RATE),
                          samplerate=SAMPLE_RATE, 
                          channels=1,
                          dtype='float32')
        sd.wait()
        write(filename, SAMPLE_RATE, recording)
        print(f"录音已保存至 {filename}")
        return True
    except Exception as e:
        print(f"录音失败: {str(e)}")
        return False

def extract_features(file_path):
    """提取声学特征"""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # 提取多维特征
        features = {
            'mfcc': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)),
            'pitch': np.mean(librosa.yin(y, fmin=50, fmax=2000)),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            'jitter': np.mean(np.abs(np.diff(np.diff(y)))),
            'shimmer': np.mean(np.abs(np.diff(y))),
            'hnr': np.mean(librosa.effects.harmonic(y))
        }
        return np.array(list(features.values())).reshape(1, -1)
    except Exception as e:
        print(f"特征提取失败: {str(e)}")
        return None

def train_model():
    """训练示例模型（实际需替换真实数据）"""
    # 示例数据（需替换为标注数据集）
    X_train = np.random.rand(100, 6)  # 100个样本，6个特征
    y_train = np.random.randint(0, 5, 100)  # 5种体质
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    print("示例模型已训练并保存")

def predict_tcm(features):
    """中医体质预测"""
    try:
        model = joblib.load(MODEL_PATH)
        prediction = model.predict(features)
        return TCM_TYPES.get(prediction[0], "未知体质")
    except Exception as e:
        print(f"预测失败: {str(e)}")
        return None

# ================== 主程序 ==================
if __name__ == "__main__":
    # 首次运行需训练示例模型
    train_model()  # 实际应用时应注释此行，使用专业训练的模型
    
    # 1.录音采集
    if record_audio():
        # 2.特征提取
        features = extract_features('patient_voice.wav')
        if features is not None:
            # 3.体质预测
            result = predict_tcm(features)
            print("\n=== 诊断结果 ===")
            print(result)
            print("=================")
            
            # 示例特征可视化（可选）
            import matplotlib.pyplot as plt
            plt.bar(range(6), features.flatten())
            plt.title('声学特征分布')
            plt.xlabel('特征索引')
            plt.ylabel('归一化值')
            plt.show()