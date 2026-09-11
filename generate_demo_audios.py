import sys, os, json
from pathlib import Path
import torch
import torchaudio

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
NEUMARK_ROOT = Path(os.environ.get('NEUMARK_ROOT', SCRIPT_DIR)).resolve()

for p in [str(NEUMARK_ROOT), str(NEUMARK_ROOT / 'train'), str(SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models import WMEmbedder, WMDetector
from STmodels.model import SpeechTokenizer
from lhotse import load_manifest_lazy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

st_cfg_path = NEUMARK_ROOT / 'STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json'
st_ckpt_path = NEUMARK_ROOT / 'STmodels/pretrained_model/SpeechTokenizer.pt'
neumark_ckpt_path = '/home/wu25/mrnas04home/projects/vall-e/egs/libritts/exp/tts_native_neumark/20260816-192445/NeuMark_epoch_000.pt'

# 1. Load SpeechTokenizer
print('[1/4] Loading SpeechTokenizer...')
with open(st_cfg_path) as f:
    st_cfg = json.load(f)
st_model = SpeechTokenizer(st_cfg)
st_state = torch.load(st_ckpt_path, map_location='cpu')
st_model.load_state_dict(st_state)
st_model.eval().to(device)

# 2. Load NeuMark
print('[2/4] Loading NeuMark Checkpoint...')
msg_processor = WMEmbedder(nbits=16, input_dim=1024, nchunk_size=4).to(device)
detector = WMDetector(input_channels=1024, nbits=16, nchunk_size=4).to(device)

ckpt = torch.load(neumark_ckpt_path, map_location='cpu')
msg_processor.load_state_dict(ckpt['msg_processor'])
detector.load_state_dict(ckpt['detector'])
msg_processor.eval()
detector.eval()

# 3. Load cuts
print('[3/4] Loading cuts_dev.jsonl.gz...')
manifest = SCRIPT_DIR / 'data/tokenized_voicemark/cuts_dev.jsonl.gz'
cuts = load_manifest_lazy(manifest)

out_dir = SCRIPT_DIR / 'exp/tts_native_neumark/demo_samples'
out_dir.mkdir(parents=True, exist_ok=True)

test_indices = [0, 1, 2]
message_str = '1011001110001101'
message_bits = torch.tensor([[int(c) for c in message_str]], dtype=torch.int64, device=device)

print('[4/4] Generating Audio Files...')
for s_idx in test_indices:
    cut = None
    for idx, c in enumerate(cuts):
        if idx == s_idx:
            cut = c
            break
    if cut is None:
        continue

    print(f'=== Processing Sample {s_idx} (ID: {cut.id}, Duration: {cut.duration:.2f}s) ===')
    codes_np = cut.load_features()
    codes = torch.from_numpy(codes_np).long().transpose(0, 1).unsqueeze(0).to(device)

    # Prompt Audio
    try:
        audio_np = cut.load_audio()
        prompt_audio = torch.from_numpy(audio_np).float()
        if cut.sampling_rate != 16000:
            prompt_audio = torchaudio.functional.resample(prompt_audio, cut.sampling_rate, 16000)
    except Exception as e:
        print(f'Warning loading raw audio: {e}')
        prompt_audio = torch.zeros((1, codes.shape[-1] * 320))

    with torch.no_grad():
        codes_qbt = codes.permute(1, 0, 2).contiguous() if codes.shape[1] == 8 else codes
        quantized_layers = [st_model.quantizer.decode(codes_qbt[k : k + 1], st=k) for k in range(8)]

        # Clean synthesis
        z_clean = sum(quantized_layers)
        clean_audio = st_model.decoder(z_clean).squeeze(0).cpu()

        # Watermarked synthesis
        wm_layers = [msg_processor(q, message_bits) for q in quantized_layers]
        z_wm = sum(wm_layers)
        wm_audio = st_model.decoder(z_wm).squeeze(0).cpu()

        # Detection verification
        embedding = st_model.forward_feature(wm_audio.unsqueeze(0).to(device))
        detect_prob, pred_bits, detected = detector.detect_watermark(embedding)

        bit_matches = (pred_bits.long().cpu() == message_bits.long().cpu()).sum().item()
        bit_acc = bit_matches / 16.0
        det_prob = detect_prob.item() if isinstance(detect_prob, torch.Tensor) else float(detect_prob)
        pred_str = ''.join(str(b.item()) for b in pred_bits[0])

    prompt_path = out_dir / f'sample_{s_idx:02d}_prompt.wav'
    clean_path = out_dir / f'sample_{s_idx:02d}_clean.wav'
    wm_path = out_dir / f'sample_{s_idx:02d}_watermarked.wav'
    diff_path = out_dir / f'sample_{s_idx:02d}_diff_x10.wav'

    torchaudio.save(str(prompt_path), prompt_audio, 16000)
    torchaudio.save(str(clean_path), clean_audio, 16000)
    torchaudio.save(str(wm_path), wm_audio, 16000)

    min_len = min(clean_audio.shape[-1], wm_audio.shape[-1])
    diff = torch.clamp((wm_audio[..., :min_len] - clean_audio[..., :min_len]) * 10.0, -1.0, 1.0)
    torchaudio.save(str(diff_path), diff, 16000)

    print(f'  Target Watermark:    {message_str}')
    print(f'  Extracted Watermark: {pred_str}')
    print(f'  Bit Accuracy:        {bit_acc * 100:.2f}%')
    print(f'  Detect Probability:  {det_prob:.4f}')
    print(f'  Saved Prompt:        {prompt_path}')
    print(f'  Saved Clean:         {clean_path}')
    print(f'  Saved Watermark:     {wm_path}')
    print(f'  Saved Diff (x10):    {diff_path}')

print('ALL DEMO AUDIOS GENERATED!')
