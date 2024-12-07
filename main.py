from PIL import Image
import pyautogui
import pytesseract
import time
import asyncio
from langchain_openai import ChatOpenAI
from openai import OpenAI
import os

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# 指定範囲でスクリーンショットを撮影
def take_screenshot_region(save_path, region):
    """
    region: (x, y, width, height)
    """
    screenshot = pyautogui.screenshot(region=region)
    screenshot.save(save_path)
    return save_path

# 非同期でOpenAI APIを呼び出す関数
async def generate_answer_async(question):
    prompt = f"""
あなたはクイズマスターです。以下の質問に答えてください。次の質問に対してひらがなもしくはアルファベットで回答してください。

# Output Format

- 回答は短く簡潔にしてください。
- ひらがなかアルファベットのみを使用して回答してください。
- 漢字で回答する場合は括弧書きで読み仮名を記載してください。

# Notes

- 回答において文字種を混ぜず、ひらがなかアルファベットのいずれかを選んでください。
- 質問に対して簡潔に、直接関連した回答をしてください。
- 単語で答えてください。文章での回答は不正解となります。

# 質問
{question}
"""
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "あなたはクイズマスターです。以下の質問に答えてください。次の質問に対してひらがなもしくはアルファベットで回答してください。"
                ),
            },
            {
                "role": "user",
                "content": (
                    prompt
                ),
            },
        ]
        client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")

        # chat completion without streaming
        response = client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online",
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI APIエラー: {e}")
        return None

# OCRで文字を抽出
def perform_ocr(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang="jpn")  # 日本語の場合は "jpn"
    return text

# メイン処理
async def main():
    # スクリーンショットを保存するパス
    screenshot_path = "region_screenshot.png"

    # スクリーンショット範囲を指定 (例: x=100, y=200, width=400, height=300)
    region = (33, 265, 247, 150)
    pre_text = ""
    pre_text_first = ""
    is_first = False

    while True:
        time.sleep(0.2)
        # 指定範囲でスクリーンショットを撮影
        take_screenshot_region(screenshot_path, region)

        extracted_text = perform_ocr(screenshot_path)
        extracted_text = extracted_text.replace('\n', '')

        # テキストが短すぎる場合はスキップ
        if len(extracted_text) < 10:
            continue

        # 同じ質問を連続して処理しないようにする
        if extracted_text == pre_text:
            if pre_text_first != extracted_text[:3]:
                pre_text_first = extracted_text[:3]
                is_first = True

        if is_first:
            is_first = False
            print(f"質問: {extracted_text}")
            answer = await generate_answer_async(extracted_text)  # 非同期で回答を取得
            print(f"回答: {answer}")
            print("\n\n\n")

        pre_text = extracted_text

# エントリーポイント
if __name__ == "__main__":
    asyncio.run(main())
