"""話し方の分析: 話速・フィラー・沈黙・声量・抑揚。"""

from __future__ import annotations

import re

import numpy as np

from .media import SAMPLE_RATE

FRAME_SEC = 0.05  # 音量解析のフレーム長
LONG_PAUSE_SEC = 2.0

# (表示名, 正規表現)。上から順にマッチさせ、重複カウントを避ける。
FILLER_PATTERNS: list[tuple[str, str]] = [
    ("えっと", r"え[ーっ]?と(?![う-ん])"),
    ("えー", r"えー+"),
    ("あのー", r"あの[ーぉ]+|あの(?=[、,。\s])"),
    ("そのー", r"その[ーぉ]+"),
    ("まあ", r"まぁ+|まあ+"),
    ("うーん", r"うー+ん|んー+"),
    ("なんか", r"なんか(?![ら-ろ])"),
]

_PUNCT_RE = re.compile(r"[\s、。，．,.!?！？「」『』（）()・…ー〜~\-]")


def _count_chars(text: str) -> int:
    return len(_PUNCT_RE.sub("", text))


# --------------------------------------------------------------------------- 音響

def _frame_rms_db(samples: np.ndarray) -> np.ndarray:
    hop = int(SAMPLE_RATE * FRAME_SEC)
    n = len(samples) // hop
    if n == 0:
        return np.zeros(0)
    frames = samples[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def _speech_mask(db: np.ndarray) -> tuple[np.ndarray, float]:
    """エネルギーベースの簡易 VAD。背景ノイズレベルに合わせて閾値を決める。"""
    if len(db) == 0:
        return np.zeros(0, dtype=bool), -50.0
    noise_floor = float(np.percentile(db, 10))
    peak = float(np.percentile(db, 95))
    threshold = max(noise_floor + 10.0, noise_floor + (peak - noise_floor) * 0.3)
    mask = db > threshold
    # 0.3 秒未満の短い無音は発話中の息継ぎとみなして埋める
    gap = int(0.3 / FRAME_SEC)
    idx = np.flatnonzero(mask)
    if len(idx) > 1:
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= gap:
                mask[a:b] = True
    return mask, threshold


def _runs(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    """mask 中で value が連続する区間 [start, end) のリスト。"""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v == value and start is None:
            start = i
        elif v != value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _pitch_track(samples: np.ndarray, speech: np.ndarray) -> np.ndarray:
    """自己相関による簡易ピッチ推定 (Hz)。無声・無音フレームは NaN。"""
    hop = int(SAMPLE_RATE * FRAME_SEC)
    win = int(SAMPLE_RATE * 0.04)
    min_lag, max_lag = SAMPLE_RATE // 400, SAMPLE_RATE // 70  # 70〜400Hz
    n = min(len(speech), (len(samples) - win) // hop)
    f0 = np.full(len(speech), np.nan)
    idx = np.flatnonzero(speech[:n])
    nfft = 1 << (2 * win - 1).bit_length()
    window = np.hanning(win)
    for chunk in np.array_split(idx, max(1, len(idx) // 2000)):
        if len(chunk) == 0:
            continue
        frames = np.stack([samples[i * hop: i * hop + win] for i in chunk]) * window
        frames -= frames.mean(axis=1, keepdims=True)
        spec = np.fft.rfft(frames, nfft)
        ac = np.fft.irfft(spec * np.conj(spec), nfft)[:, : max_lag + 1]
        energy = ac[:, :1] + 1e-12
        ac = ac / energy
        lag = np.argmax(ac[:, min_lag:], axis=1) + min_lag
        strength = ac[np.arange(len(chunk)), lag]
        voiced = strength > 0.45
        f0[chunk[voiced]] = SAMPLE_RATE / lag[voiced]
    return f0


def analyze_audio(samples: np.ndarray, duration: float) -> dict:
    db = _frame_rms_db(samples)
    speech, threshold = _speech_mask(db)
    speaking_time = float(speech.sum() * FRAME_SEC)

    # 長い沈黙 (冒頭・末尾は除く)
    pauses = []
    for s, e in _runs(speech, False):
        if s == 0 or e == len(speech):
            continue
        length = (e - s) * FRAME_SEC
        if length >= LONG_PAUSE_SEC:
            pauses.append({"start": round(s * FRAME_SEC, 2), "end": round(e * FRAME_SEC, 2),
                           "duration": round(length, 2)})

    speech_db = db[speech]
    f0 = _pitch_track(samples, speech)
    voiced_f0 = f0[~np.isnan(f0)]
    if len(voiced_f0) > 20:
        # 外れ値 (オクターブ誤り) を除いてから半音単位の標準偏差を取る
        lo, hi = np.percentile(voiced_f0, [5, 95])
        core = voiced_f0[(voiced_f0 >= lo) & (voiced_f0 <= hi)]
        semitones = 12 * np.log2(core / np.median(core))
        pitch_median = float(np.median(core))
        pitch_std_st = float(np.std(semitones))
    else:
        pitch_median, pitch_std_st = None, None

    # 1 秒ごとのタイムライン (グラフ表示用)
    per_sec = int(1 / FRAME_SEC)
    timeline = []
    for sec in range(int(np.ceil(len(db) / per_sec))):
        sl = slice(sec * per_sec, (sec + 1) * per_sec)
        sp = speech[sl]
        seg_f0 = f0[sl]
        seg_f0 = seg_f0[~np.isnan(seg_f0)]
        timeline.append({
            "t": sec,
            "volume_db": round(float(db[sl][sp].max()), 1) if sp.any() else None,
            "pitch_hz": round(float(np.median(seg_f0)), 1) if len(seg_f0) >= 3 else None,
        })

    return {
        "speaking_time": round(speaking_time, 1),
        "speech_ratio": round(speaking_time / duration, 3) if duration else 0,
        "long_pauses": pauses,
        "volume_mean_db": round(float(speech_db.mean()), 1) if len(speech_db) else None,
        "volume_std_db": round(float(speech_db.std()), 1) if len(speech_db) else None,
        "vad_threshold_db": round(threshold, 1),
        "pitch_median_hz": round(pitch_median, 1) if pitch_median else None,
        "pitch_std_semitones": round(pitch_std_st, 2) if pitch_std_st is not None else None,
        "timeline": timeline,
    }


# --------------------------------------------------------------------------- テキスト

def find_fillers(segments: list[dict]) -> dict:
    counts: dict[str, int] = {label: 0 for label, _ in FILLER_PATTERNS}
    occurrences = []
    for i, seg in enumerate(segments):
        text = seg["text"]
        taken = [False] * len(text)
        for label, pattern in FILLER_PATTERNS:
            for m in re.finditer(pattern, text):
                if any(taken[m.start():m.end()]):
                    continue
                for k in range(m.start(), m.end()):
                    taken[k] = True
                counts[label] += 1
                occurrences.append({"segment": i, "t": seg["start"], "label": label,
                                    "start": m.start(), "end": m.end()})
    total = sum(counts.values())
    return {"total": total, "counts": {k: v for k, v in counts.items() if v},
            "occurrences": occurrences}


def speaking_rate(segments: list[dict]) -> dict:
    total_chars, total_time = 0, 0.0
    per_segment = []
    for seg in segments:
        dur = max(seg["end"] - seg["start"], 0.01)
        chars = _count_chars(seg["text"])
        total_chars += chars
        total_time += dur
        if dur >= 1.5:
            per_segment.append({"t": seg["start"], "cpm": round(chars / dur * 60)})
    cpm = total_chars / total_time * 60 if total_time else 0
    return {"chars": total_chars, "chars_per_minute": round(cpm), "per_segment": per_segment}


# --------------------------------------------------------------------------- 総合

def _rate(level: str, title: str, value: str, advice: str) -> dict:
    return {"level": level, "title": title, "value": value, "advice": advice}


def evaluate(audio: dict, rate: dict, fillers: dict, duration: float) -> list[dict]:
    """ルールベースの評価 (AI 講評なしでも使えるアドバイス)。"""
    items = []
    cpm = rate["chars_per_minute"]
    if cpm == 0:
        pass
    elif cpm > 380:
        items.append(_rate("bad", "話すスピード", f"{cpm} 字/分",
                           "かなり早口です。一文を短く区切り、句点ごとに一呼吸おきましょう (目安 300 字/分)。"))
    elif cpm > 340:
        items.append(_rate("warn", "話すスピード", f"{cpm} 字/分",
                           "やや早口です。結論や数字など大事な部分はゆっくり話すと伝わりやすくなります。"))
    elif cpm < 220:
        items.append(_rate("warn", "話すスピード", f"{cpm} 字/分",
                           "ややゆっくりです。間延びして聞こえないよう、テンポを少し上げてみましょう。"))
    else:
        items.append(_rate("good", "話すスピード", f"{cpm} 字/分", "聞き取りやすいスピードです。"))

    minutes = max(audio["speaking_time"] / 60, 0.1)
    fpm = fillers["total"] / minutes
    top = "、".join(f"「{k}」{v}回" for k, v in sorted(fillers["counts"].items(), key=lambda x: -x[1])[:3])
    if fpm >= 6:
        items.append(_rate("bad", "フィラー (えー・あのー等)", f"{fillers['total']} 回 ({fpm:.1f} 回/分)",
                           f"多めです ({top})。言葉に詰まったら『えー』の代わりに無言で一拍おく練習をしましょう。"))
    elif fpm >= 3:
        items.append(_rate("warn", "フィラー (えー・あのー等)", f"{fillers['total']} 回 ({fpm:.1f} 回/分)",
                           f"やや目立ちます ({top})。話し始めのフィラーを意識して減らしましょう。"))
    else:
        items.append(_rate("good", "フィラー (えー・あのー等)", f"{fillers['total']} 回 ({fpm:.1f} 回/分)",
                           "フィラーは少なく、すっきり話せています。"))

    pauses = audio["long_pauses"]
    longest = max((p["duration"] for p in pauses), default=0)
    if longest >= 5:
        items.append(_rate("warn", "長い沈黙", f"{len(pauses)} 回 (最長 {longest:.1f} 秒)",
                           "5 秒以上の沈黙があります。考える時は『少し考えさせてください』と一言添えると印象が良くなります。"))
    else:
        items.append(_rate("good", "長い沈黙", f"{len(pauses)} 回 (最長 {longest:.1f} 秒)",
                           "目立つ沈黙はありません。"))

    st = audio["pitch_std_semitones"]
    if st is not None:
        if st < 1.8:
            items.append(_rate("warn", "声の抑揚", f"{st:.1f} 半音",
                               "声の高さの変化が小さく、単調に聞こえる可能性があります。強調したい語で少し声を張りましょう。"))
        else:
            items.append(_rate("good", "声の抑揚", f"{st:.1f} 半音", "適度に抑揚がついています。"))

    vol = audio["volume_mean_db"]
    if vol is not None:
        if vol < -35:
            items.append(_rate("warn", "声の大きさ", f"{vol:.0f} dBFS",
                               "録音上の声が小さめです。マイク位置を確認し、もう一段はっきり発声しましょう。"))
        else:
            items.append(_rate("good", "声の大きさ", f"{vol:.0f} dBFS", "十分な音量で録音されています。"))
    return items


def analyze(segments: list[dict], samples: np.ndarray, duration: float) -> dict:
    audio = analyze_audio(samples, duration)
    rate = speaking_rate(segments)
    fillers = find_fillers(segments)
    return {
        "audio": audio,
        "rate": rate,
        "fillers": fillers,
        "evaluation": evaluate(audio, rate, fillers, duration),
    }
