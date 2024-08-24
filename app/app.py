import os
import re
from glob import glob

import gradio as gr
from process import (
    diff_texts,
    extract_docx,
    extract_pdf,
    extract_text,
    model_load,
    translator,
    translator_sec,
)


def huanik(
    endpoint: str,
    base: str,
    model: str,
    api_key: str,
    choice: str,
    endpoint2: str,
    base2: str,
    model2: str,
    api_key2: str,
    source_lang: str,
    target_lang: str,
    source_text: str,
    country: str,
    max_tokens: int,
    temperature: int,
    rpm: int,
):
    if not source_text or source_lang == target_lang:
        raise gr.Error(
            "请检查内容或选项是否输入正确。"
        )

    try:
        model_load(endpoint, base, model, api_key, temperature, rpm)
    except Exception as e:
        raise gr.Error(f"发生了意外的错误：{e}") from e

    source_text = re.sub(r"(?m)^\s*$\n?", "", source_text)

    if choice:
        init_translation, reflect_translation, final_translation = (
            translator_sec(
                endpoint2=endpoint2,
                base2=base2,
                model2=model2,
                api_key2=api_key2,
                source_lang=source_lang,
                target_lang=target_lang,
                source_text=source_text,
                country=country,
                max_tokens=max_tokens,
            )
        )

    else:
        init_translation, reflect_translation, final_translation = translator(
            source_lang=source_lang,
            target_lang=target_lang,
            source_text=source_text,
            country=country,
            max_tokens=max_tokens,
        )

    final_diff = gr.HighlightedText(
        diff_texts(init_translation, final_translation),
        label="翻译对比",
        combine_adjacent=True,
        show_legend=True,
        visible=True,
        color_map={"removed": "red", "added": "green"},
    )

    return init_translation, reflect_translation, final_translation, final_diff


def update_model(endpoint):
    endpoint_model_map = {
        "Groq": "llama3-70b-8192",
        "OpenAI": "gpt-4o",
        "TogetherAI": "Qwen/Qwen2-72B-Instruct",
        "Ollama": "llama3.1",
        "CUSTOM": "",
    }
    if endpoint == "CUSTOM":
        # 如果选择自定义，使基础模型输入框可见
        base = gr.update(visible=True)
    else:
        # 否则，隐藏基础模型输入框
        base = gr.update(visible=False)
    # 返回更新后的模型映射和基础模型输入框的可见性状态
    return gr.update(value=endpoint_model_map[endpoint]), base


def read_doc(path):
    file_type = path.split(".")[-1]
    print("文件类型：", file_type)
    if file_type in ["pdf", "txt", "py", "docx", "json", "cpp", "md"]:
        if file_type.endswith("pdf"):
            content = extract_pdf(path)
        elif file_type.endswith("docx"):
            content = extract_docx(path)
        else:
            content = extract_text(path)
        # 清除字符串开头和结尾的空白字符和换行符
        return re.sub(r"(?m)^\s*$\n?", "", content)
    else:
        # 如果文件类型不支持，抛出错误
        raise gr.Error("哎呀，不支持的文件类型。")


def enable_sec(choice):
    if choice:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def update_menu(visible):
    return not visible, gr.update(visible=not visible)


def export_txt(strings):
    if strings:
        os.makedirs("outputs", exist_ok=True)
        base_count = len(glob(os.path.join("outputs", "*.txt")))
        file_path = os.path.join("outputs", f"{base_count:06d}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(strings)
        return gr.update(value=file_path, visible=True)
    else:
        return gr.update(visible=False)


def switch(source_lang, source_text, target_lang, output_final):
    if output_final:
        return (
            gr.update(value=target_lang),
            gr.update(value=output_final),
            gr.update(value=source_lang),
            gr.update(value=source_text),
        )
    else:
        return (
            gr.update(value=target_lang),
            gr.update(value=source_text),
            gr.update(value=source_lang),
            gr.update(value=""),
        )


def close_btn_show():
    return gr.update(visible=False), gr.update(visible=True)


def close_btn_hide(output_diff):
    if output_diff:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)


TITLE = """
    <div style="display: inline-flex;">
        <div style="margin-left: 6px; font-size:32px; color: #6366f1"><b>翻译代理</b> WebUI</div>
    </div>
"""

CSS = """
    h1 {
        text-align: center;
        display: block;
        height: 10vh;
        align-content: center;
    }
    footer {
        visibility: hidden;
    }
    .menu_btn {
        width: 48px;
        height: 48px;
        max-width: 48px;
        min-width: 48px;
        padding: 0px;
        background-color: transparent;
        border: none;
        cursor: pointer;
        position: relative;
        box-shadow: none;
    }
    .menu_btn::before,
    .menu_btn::after {
        content: '';
        position: absolute;
        width: 30px;
        height: 3px;
        background-color: #4f46e5;
        transition: transform 0.3s ease;
    }
    .menu_btn::before {
        top: 12px;
        box-shadow: 0 8px 0 #6366f1;
    }
    .menu_btn::after {
        bottom: 16px;
    }
    .menu_btn.active::before {
        transform: translateY(8px) rotate(45deg);
        box-shadow: none;
    }
    .menu_btn.active::after {
        transform: translateY(-8px) rotate(-45deg);
    }
    .lang {
        max-width: 100px;
        min-width: 100px;
    }
"""

JS = """
    function () {
        const menu_btn = document.getElementById('menu');
        menu_btn.classList.toggle('active');
    }

"""

with gr.Blocks(theme="default", css=CSS, fill_height=True) as demo:
    with gr.Row():
        visible = gr.State(value=True)
        menu_btn = gr.Button(
            value="", elem_classes="menu_btn", elem_id="menu", size="sm"
        )
        gr.HTML(TITLE)
    with gr.Row():
        with gr.Column(scale=1) as menubar:
            endpoint = gr.Dropdown(
                label="端点",
                choices=["OpenAI", "Groq", "TogetherAI", "Ollama", "CUSTOM"],
                value="OpenAI",
            )
            choice = gr.Checkbox(
                label="附加端点",
                info="用于反思的附加端点",
            )
            model = gr.Textbox(
                label="模型",
                value="gpt-4o",
            )
            api_key = gr.Textbox(
                label="API密钥",
                type="password",
            )
            base = gr.Textbox(label="基础URL", visible=False)
            with gr.Column(visible=False) as AddEndpoint:
                endpoint2 = gr.Dropdown(
                    label="附加端点",
                    choices=[
                        "OpenAI",
                        "Groq",
                        "TogetherAI",
                        "Ollama",
                        "CUSTOM",
                    ],
                    value="OpenAI",
                )
                model2 = gr.Textbox(
                    label="模型",
                    value="gpt-4o",
                )
                api_key2 = gr.Textbox(
                    label="API密钥",
                    type="password",
                )
                base2 = gr.Textbox(label="基础URL", visible=False)
            with gr.Row():
                source_lang = gr.Textbox(
                    label="源语言",
                    value="中文",
                    elem_classes="lang",
                )
                target_lang = gr.Textbox(
                    label="目标语言",
                    value="英语",
                    elem_classes="lang",
                )
            switch_btn = gr.Button(value="🔄️ 切换")
            country = gr.Textbox(
                label="国家", value="China", max_lines=1
            )
            with gr.Accordion("高级选项", open=False):
                max_tokens = gr.Slider(
                    label="每块最大令牌数",
                    minimum=512,
                    maximum=2046,
                    value=1000,
                    step=8,
                )
                temperature = gr.Slider(
                    label="温度",
                    minimum=0,
                    maximum=1.0,
                    value=0.3,
                    step=0.1,
                )
                rpm = gr.Slider(
                    label="每分钟请求次数",
                    minimum=1,
                    maximum=1000,
                    value=60,
                    step=1,
                )
                
# 源文本：If one advances confidently in the direction of his dreams, and endeavors to live the life which he has imagined, he will meet with a success unexpected in common hours.
# 初始翻译：如果一个人自信地朝着梦想的方向前进，并努力过着他所设想的人生，那么他将在平常时刻遇到出乎意料的成功。
# 最终结果：如果他朝着梦想迈进，努力实现他所设想的人生，那么他就会在平常时刻获得意外的成功。
        with gr.Column(scale=4):
            source_text = gr.Textbox(
                label="源文本",
                value="如果她朝着梦想迈进，努力实现她所设想的人生，那么她就会在平常时刻获得意外的成功。",
                lines=12,
            )
            with gr.Tab("最终结果"):
                output_final = gr.Textbox(
                    label="最终翻译", lines=12, show_copy_button=True
                )
            with gr.Tab("初始翻译"):
                output_init = gr.Textbox(
                    label="初始翻译", lines=12, show_copy_button=True
                )
            with gr.Tab("反思"):
                output_reflect = gr.Textbox(
                    label="反思", lines=12, show_copy_button=True
                )
            with gr.Tab("差异对比"):
                output_diff = gr.HighlightedText(visible=False)
    with gr.Row():
        submit = gr.Button(value="翻译")
        upload = gr.UploadButton(label="上传", file_types=["text"])
        export = gr.DownloadButton(label="下载", visible=False)
        clear = gr.ClearButton(
            [source_text, output_init, output_reflect, output_final],
            value="清除"
        )
        close = gr.Button(value="停止", visible=False)

    switch_btn.click(
        fn=switch,
        inputs=[source_lang, source_text, target_lang, output_final],
        outputs=[source_lang, source_text, target_lang, output_final],
    )

    menu_btn.click(
        fn=update_menu, inputs=visible, outputs=[visible, menubar], js=JS
    )
    endpoint.change(fn=update_model, inputs=[endpoint], outputs=[model, base])

    choice.select(fn=enable_sec, inputs=[choice], outputs=[AddEndpoint])
    endpoint2.change(
        fn=update_model, inputs=[endpoint2], outputs=[model2, base2]
    )

    start_ta = submit.click(
        fn=huanik,
        inputs=[
            endpoint,
            base,
            model,
            api_key,
            choice,
            endpoint2,
            base2,
            model2,
            api_key2,
            source_lang,
            target_lang,
            source_text,
            country,
            max_tokens,
            temperature,
            rpm,
        ],
        outputs=[output_init, output_reflect, output_final, output_diff],
    )
    upload.upload(fn=read_doc, inputs=upload, outputs=source_text)
    output_diff.change(fn=export_txt, inputs=output_final, outputs=[export])

    submit.click(fn=close_btn_show, outputs=[clear, close])
    output_diff.change(
        fn=close_btn_hide, inputs=output_diff, outputs=[clear, close]
    )
    close.click(fn=None, cancels=start_ta)

if __name__ == "__main__":
    demo.queue(api_open=False).launch(show_api=False, share=False)
