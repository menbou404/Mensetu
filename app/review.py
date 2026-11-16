"""Claude による回答内容の講評。"""

from __future__ import annotations

import json

from . import config

SYSTEM_PROMPT = """\
あなたは新卒・中途採用の面接官経験が豊富なキャリアアドバイザーです。
面接練習の録画を文字起こししたものと、話し方・表情の計測データを受け取り、
受験者本人が次の面接に向けて改善できるよう、具体的で率直なフィードバックを日本語で返します。

前提:
- 文字起こしには話者ラベルがありません。面接官の質問と受験者の回答が混在しているので、文脈から判別してください。
  受験者が一人で話している練習動画 (質問なし) の場合は、話の区切りごとに「想定される質問」を補って評価してください。
- 文字起こしは音声認識の結果なので、誤変換が含まれることがあります。誤変換と思われる箇所を受験者のミスとして指摘しないでください。
- 計測データは機械的な推定値で誤差があります。数値を断定的に扱わず、内容の評価を主にしてください。

評価の観点:
- 質問の意図に答えているか / 結論から話しているか (PREP)
- 具体的なエピソード・数字・自分の役割や行動が語られているか
- 論理の一貫性、長さの適切さ (1 回答 1〜2 分が目安)
- 志望度・人柄・再現性 (入社後も活躍できそうか) が伝わるか

出力のルール:
- score は 0〜100 の整数、各回答の score は 1〜5 の整数。
- question_time は質問 (または回答の開始) の時刻を秒で。
- better_answer は受験者の実際の経験の範囲で書き直した回答例 (事実を捏造しない。不足情報は【 】で補う箇所を示す)。
- action_items は次回までに取り組むべきことを重要な順に 3 つ。
"""

_str = {"type": "string"}
_str_list = {"type": "array", "items": _str}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "integer"},
        "overall_summary": _str,
        "strengths": _str_list,
        "qa": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": _str,
                    "question_time": {"type": "number"},
                    "answer_summary": _str,
                    "score": {"type": "integer"},
                    "good_points": _str_list,
                    "improvements": _str_list,
                    "better_answer": _str,
                },
                "required": ["question", "question_time", "answer_summary", "score",
                             "good_points", "improvements", "better_answer"],
                "additionalProperties": False,
            },
        },
        "delivery_feedback": _str,
        "nonverbal_feedback": _str,
        "action_items": _str_list,
    },
    "required": ["overall_score", "overall_summary", "strengths", "qa",
                 "delivery_feedback", "nonverbal_feedback", "action_items"],
    "additionalProperties": False,
}


def _fmt_time(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def _metrics_digest(speech: dict, vision: dict | None) -> dict:
    """タイムラインなど大きいデータを除いた要約だけを渡す。"""
    audio = speech["audio"]
    digest = {
        "話速_字毎分": speech["rate"]["chars_per_minute"],
        "フィラー": speech["fillers"]["counts"],
        "フィラー合計": speech["fillers"]["total"],
        "発話時間_秒": audio["speaking_time"],
        "2秒以上の沈黙": [f"{_fmt_time(p['start'])} ({p['duration']}秒)" for p in audio["long_pauses"]],
        "声の抑揚_半音SD": audio["pitch_std_semitones"],
    }
    if vision and vision.get("available"):
        digest.update({
            "カメラ方向を見ていた割合": vision["eye_contact_ratio"],
            "笑顔の割合": vision["smile_ratio"],
            "頭の動きのばらつき_度": vision["head_movement_std"],
            "瞬き_回毎分": vision["blinks_per_minute"],
            "目線が2秒以上外れた区間": [f"{_fmt_time(p['start'])}-{_fmt_time(p['end'])}"
                                         for p in vision["look_away_periods"][:20]],
        })
    return digest


def review(transcript: dict, speech: dict, vision: dict | None, context: str = "") -> dict:
    if not config.has_anthropic_key():
        raise RuntimeError("ANTHROPIC_API_KEY が設定されていません (.env を確認してください)")
    import anthropic

    lines = "\n".join(f"[{_fmt_time(s['start'])}] {s['text']}" for s in transcript["segments"])
    user = (
        f"## 面接の情報 (受験者による入力)\n{context.strip() or '(なし)'}\n\n"
        f"## 計測データ\n```json\n{json.dumps(_metrics_digest(speech, vision), ensure_ascii=False, indent=1)}\n```\n\n"
        f"## 文字起こし\n{lines}"
    )

    client = anthropic.Anthropic()
    # Opus 5.5 が安全分類で回答を断った場合に、サーバー側で別モデルに自動フォールバックさせる
    with client.beta.messages.stream(
        model=config.CLAUDE_MODEL,
        max_tokens=32000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        thinking={"type": "adaptive"},
        output_config={
            "effort": config.CLAUDE_EFFORT,
            "format": {"type": "json_schema", "schema": REVIEW_SCHEMA},
        },
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        raise RuntimeError("AI が講評の生成を辞退しました。")
    if message.stop_reason == "max_tokens":
        raise RuntimeError("講評が長すぎて途中で切れました。動画を短くして再試行してください。")
    text = next(b.text for b in message.content if b.type == "text")
    data = json.loads(text)
    data["model"] = message.model
    return data
