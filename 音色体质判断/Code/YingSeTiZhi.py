"""
中医体质语音识别系统
依赖库安装：pip install sounddevice librosa scikit-learn numpy matplotlib
"""
import sounddevice as sd  # 语音录制库
import numpy as np        # 数值计算库
from scipy.io.wavfile import write  # 保存wav音频文件
import librosa           # 音频特征提取库
import joblib            # 模型保存与加载
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器
import matplotlib.pyplot as plt  # 可视化库

# ================== 配置参数 ==================
RECORD_DURATION = 5  # 录音时长(秒)
SAMPLE_RATE = 44100  # 采样率
MODEL_PATH = 'D:/ProjectOpenSource/中医诊断AI/音色体质判断/Model/tcm_model.pkl'  # 模型保存路径
filename = 'D:/ProjectOpenSource/中医诊断AI/音色体质判断/Data/patient_voice.wav'  # 录音文件名

# 中医体质类型映射（扩充至9种常见体质）
TCM_TYPES = {
    0: "平和质（体质均衡，建议保持良好生活习惯）",
    1: "气虚质（建议补气健脾，增强体力）",
    2: "阳虚质（建议温补阳气，注意保暖）",
    3: "阴虚质（建议滋阴润燥，避免辛辣）",
    4: "痰湿质（建议化痰祛湿，清淡饮食）",
    5: "湿热质（建议清热利湿，避免油腻）",
    6: "血瘀质（建议活血化瘀，适度运动）",
    7: "气郁质（建议疏肝解郁，保持心情舒畅）",
    8: "特禀质（建议防护过敏，注意体质特殊性）"
}

# ================== 核心功能 ==================
def record_audio(filename):
    """录制患者语音"""
    print(f"开始录音，请保持安静...（时长{RECORD_DURATION}秒）")
    try:
        # 录制音频，返回float32类型的numpy数组
        recording = sd.rec(
            int(RECORD_DURATION * SAMPLE_RATE),  # 总采样点数
            samplerate=SAMPLE_RATE,              # 采样率
            channels=1,                          # 单声道
            dtype='float32'                      # 数据类型
        )
        sd.wait()  # 等待录音结束
        write(filename, SAMPLE_RATE, recording)  # 保存为wav文件
        print(f"录音已保存至 {filename}")
        return True
    except Exception as e:
        print(f"录音失败: {str(e)}")
        return False

def extract_features(file_path):
    """提取声学特征"""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)  # 加载音频，y为波形，sr为采样率
        # MFCC：梅尔频率倒谱系数，反映音色和语音特征
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
        # Pitch：基频，反映声音的高低
        pitch = np.mean(librosa.yin(y, fmin=50, fmax=2000))
        # Spectral Centroid：谱质心，反映声音的明亮度
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        # Jitter：基频微扰，反映声音的稳定性
        jitter = np.mean(np.abs(np.diff(np.diff(y))))
        # Shimmer：振幅微扰，反映声音的颤抖程度
        shimmer = np.mean(np.abs(np.diff(y)))
        # HNR：谐噪比，反映声音的清晰度
        hnr = np.mean(librosa.effects.harmonic(y))
        # Zero Crossing Rate：过零率，反映信号变化频率
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        # Spectral Bandwidth：谱带宽度，反映频谱分布宽度
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        # Spectral Contrast：谱对比度，反映不同频段能量差异
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        # Root Mean Square Energy：均方根能量，反映声音强度
        rmse = np.mean(librosa.feature.rms(y=y))
        # Chroma STFT：色度特征，反映音高分布
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        # Tonnetz：音调网络特征，反映音调关系
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr))
        # 组合所有特征为一个数组
        features = [
            mfcc, pitch, spectral_centroid, jitter, shimmer, hnr,
            zcr, spectral_bandwidth, spectral_contrast, rmse, chroma_stft, tonnetz
        ]
        return np.array(features).reshape(1, -1)  # 返回形状为(1, 特征数)的数组
    except Exception as e:
        print(f"特征提取失败: {str(e)}")
        return None

def train_model():
    """训练示例模型（实际需替换真实数据）"""
    # 生成随机特征数据，实际应用需替换为真实标注数据
    X_train = np.random.rand(200, 12)  # 200个样本，12个特征
    y_train = np.random.randint(0, 9, 200)  # 9种体质标签
    model = RandomForestClassifier(n_estimators=100)  # 创建随机森林分类器
    model.fit(X_train, y_train)  # 拟合模型
    joblib.dump(model, MODEL_PATH)  # 保存模型到本地
    print("示例模型已训练并保存")

def predict_tcm(features):
    """中医体质预测"""
    try:
        model = joblib.load(MODEL_PATH)  # 加载训练好的模型
        prediction = model.predict(features)  # 预测体质类型
        return TCM_TYPES.get(prediction[0], "未知体质")  # 返回体质类型描述
    except Exception as e:
        print(f"预测失败: {str(e)}")
        return None

# ================== 主程序 ==================
if __name__ == "__main__":
    train_model()  # 首次运行需训练示例模型，实际应用时应注释此行
    # 1.录音采集
    if record_audio(filename):
        # 2.特征提取
        features = extract_features(filename)
        if features is not None:
            # 3.体质预测
            result = predict_tcm(features)
            print("\n=== 诊断结果 ===")
            print(result)
            print("=================")
            '''
            # 示例特征可视化（可选）
            plt.bar(range(12), features.flatten())
            plt.title('声学特征分布')
            plt.xlabel('特征索引')
            plt.ylabel('归一化值')
            plt.show()
            '''