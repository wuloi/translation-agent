## 翻译代理 WebUI

该仓库包含一个 Gradio WebUI，用于翻译代理，该代理利用各种语言模型进行翻译。

### 预览

![webui](image.png)

**特性：**

- **分词文本：** 显示带有分词的翻译文本，突出显示原始词和翻译词之间的差异。
- **文档上传：** 支持上传各种文档格式（PDF、TXT、DOC 等）进行翻译。
- **多个 API 支持：** 与流行的语言模型集成，如：
    - Groq
    - OpenAI
    - Ollama
    - Together AI
    ...
- **不同的 LLM 用于反思：** 现在你可以启用第二个端点，使用另一个 LLM 进行反思。

**开始使用：**

1. **安装依赖：**

    **Linux**
    ```bash
    git clone https://github.com/andrewyng/translation-agent.git 
    cd translation-agent
    poetry install --with app
    poetry shell
    ```
    **Windows**
    ```bash
    git clone https://github.com/andrewyng/translation-agent.git 
    cd translation-agent
    poetry install --with app
    poetry shell
    ```

2. **设置 API 密钥：**
   - 将 `.env.sample` 重命名为 `.env`，你可以为每项服务添加你的 API 密钥：

     ```
     OPENAI_API_KEY="sk-xxxxx" # 保留此字段
     GROQ_API_KEY="xxxxx"
     TOGETHER_API_KEY="xxxxx"
     ```
    - 然后你也可以在 WebUI 中设置 API_KEY。

3. **运行 Web UI：**

    **Linux**
    ```bash
    python app/app.py
    ```
    **Windows**
    ```bash
    python .\app\app.py
    ```

4. **访问 Web UI：**
   打开你的网络浏览器，导航到 `http://127.0.0.1:7860/`。

**使用方法：**

1. 从端点下拉菜单中选择你想要使用的翻译 API。
2. 输入源语言、目标语言和国家（可选）。
3. 输入源文本或上传你的文档文件。
4. 提交并获取翻译，UI 将显示带有分词和突出显示差异的翻译文本。
5. 启用第二个端点，你可以添加另一个端点，通过不同的 LLMs 进行反思。
6. 使用自定义端点，你可以输入一个与 OpenAI 兼容的 API 基础 URL。

**定制：**

- **添加新的 LLMs：** 修改 `patch.py` 文件以集成额外的 LLMs。

**贡献：**

欢迎贡献！随时提出问题或提交拉取请求。

**许可证：**

本项目根据 MIT 许可证授权。

**演示：**

[Huggingface 演示](https://huggingface.co/spaces/vilarin/Translation-Agent-WebUI)
