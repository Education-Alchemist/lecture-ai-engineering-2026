"""第1回: 会話履歴を保ちながら素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 講座に固有の質問から会話を始める

ツールを持たない「素の」呼び出し。終了するまで、会話履歴を保って質問できる。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time

from google import genai
from google.genai import errors

from app.config import api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
MODELS = ("gemini-3.5-flash-lite", "gemini-3.6-flash")


def ask(chat: genai.chats.Chat, question: str) -> str:
    started_at = time.perf_counter()
    response = chat.send_message(question)
    elapsed = time.perf_counter() - started_at
    print(f"[応答時間] {elapsed:.2f}秒")

    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[トークン] 入力: {usage.prompt_token_count}, "
            f"出力: {usage.candidates_token_count}"
        )

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[比較モデル] {', '.join(MODELS)}")
    print("終了するには exit または quit を入力してください。\n")
    try:
        client = genai.Client(api_key=get_api_key())
        chats = {model: client.chats.create(model=model) for model in MODELS}
        question = args.question

        while True:
            print(f"[質問] {question}\n")
            for model in MODELS:
                for run in range(1, 4):
                    print(f"--- model={model} / {run}回目 ---")
                    print(ask(chats[model], question))
            try:
                while True:
                    question = input("\n次の質問（exit/quit で終了）: ").strip()
                    if question.lower() in ("exit", "quit"):
                        return
                    if question:
                        break
            except EOFError:
                break
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
