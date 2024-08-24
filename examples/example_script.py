import os

import translation_agent as ta


if __name__ == "__main__":
    source_lang, target_lang, country = "英语", "中文", "中国"

    # relative_path = "sample-texts/sample-short1.txt"
    # relative_path = "sample-texts/sample-long1.txt"
    relative_path = "sample-texts/data_points_samples.json"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    full_path = os.path.join(script_dir, relative_path)

    with open(full_path, encoding="utf-8") as file:
        source_text = file.read()

    print(f"源文本:\n\n{source_text}\n------------\n")

    translation = ta.translate(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        country=country,
    )

    print(f"翻译结果:\n\n{translation}")
