# tts demo
from melo_onnx import MeloTTX_ONNX
import soundfile

model_path = "path/to/folder/of/model_tts"
tts = MeloTTX_ONNX(model_path)
audio = tts.speak("今天天气真nice。", tts.speakers[0])

soundfile.write("path/of/result.wav", audio, samplerate=tts.sample_rate)


#         optimizer (torch.optim.Optimizer): 优化器。
# Tone clone demo
from melo_onnx import OpenVoiceToneClone_ONNX
tc = OpenVoiceToneClone_ONNX("path/to/folder/of/model_tone_clone")
import soundfile
tgt = soundfile.read("path/of/audio_for_tone_color", dtype='float32')
tgt = tc.resample(tgt[0], tgt[1])
tgt_tone_color = tc.extract_tone_color(tgt)
src = soundfile.read("path/of/audio_to_change_tone", dtype='float32')
src = tc.resample(src[0], src[1])
result = tc.tone_clone(src, tgt_tone_color)
soundfile.write("path/of/result.wav", result, tc.sample_rate)