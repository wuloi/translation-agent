# 翻译代理：使用反思工作流程的代理翻译

这是一个用于机器翻译的反思代理工作流程的 Python 演示。主要步骤包括：
1. 提示一个大型语言模型（LLM）将文本从 `源语言` 翻译到 `目标语言`；
2. 让 LLM 反思翻译内容，提出改进建议；
3. 使用这些建议来改进翻译。

## 可定制性

通过将 LLM 作为翻译引擎的核心，该系统具有高度的可操作性。例如，通过改变提示，使用这个工作流程比传统的机器翻译（MT）系统更容易：
- 修改输出的风格，如正式/非正式。
- 指定如何处理成语和特殊术语，如姓名、技术术语和缩写。例如，在提示中包含术语表可以确保特定术语（如开源、H100 或 GPU）的翻译一致性。
- 指定特定地区的语言使用或特定方言，以服务目标受众。例如，拉丁美洲的西班牙语与西班牙的西班牙语不同；加拿大的法语与法国的法语不同。

**这不是成熟的软件**，是 Andrew 在过去几个月的周末玩转翻译的结果，加上合作者（Joaquin Dominguez、Nedelina Teneva、John Santerre）帮助重构代码。

根据我们使用 BLEU 分数在传统翻译数据集上的评估，这种工作流程有时与领先的商业产品相当，但有时也不如。然而，我们也偶尔用这种方法得到了非常好的结果（优于商业产品）。我们认为这只是代理翻译的起点，并且这是一个有希望的翻译方向，有很大的改进空间，这就是我们发布这个演示的原因，以鼓励更多的讨论、实验、研究和开源贡献。

如果代理翻译能够产生比传统架构（例如一次性输入文本并直接输出翻译的端到端变换器）更好的结果——这些通常运行速度更快/成本更低——这也提供了一种自动生成训练数据（平行文本语料库）的机制，这些数据可以用来进一步训练和改进传统算法。（另见 The Batch 中的[这篇文章](https://www.deeplearning.ai/the-batch/building-models-that-learn-from-themselves/)，关于使用 LLM 生成训练数据。）

非常欢迎对如何改进这个的评论和建议！

## 开始使用

要开始使用 `translation-agent`，请按照以下步骤操作：

### 安装：
- 安装需要 Poetry 包管理器。[Poetry 安装](https://python-poetry.org/docs/#installation) 根据您的环境，以下命令可能会工作：

```bash
pip install poetry
```

- 运行工作流程需要一个包含 OPENAI_API_KEY 的 .env 文件。请参阅 .env.sample 文件作为示例。
```bash
git clone https://github.com/andrewyng/translation-agent.git 
cd translation-agent
poetry install
poetry shell # 激活虚拟环境
```

### 使用方法：

```python
import translation_agent as ta
source_lang, target_lang, country = "English", "Spanish", "Mexico"
translation = ta.translate(source_lang, target_lang, source_text, country)
```
请参阅 examples/example_script.py 获取示例脚本。

## 许可证

翻译代理根据 **MIT 许可证** 发布。您可以自由使用、修改和分发代码，无论是商业还是非商业目的。

## 扩展思路

这里有一些我们没有时间尝试，但我们希望开源社区将会尝试的想法：
- **尝试其他 LLMs。** 我们主要使用 gpt-4-turbo 原型设计。我们希望其他人尝试其他 LLMs 以及其他超参数选择，看看是否有一些在特定语言对上的表现更好。
- **术语表创建。** 什么是有效构建术语表的最好方法——可能使用 LLM——我们希望一致翻译的最重要术语？例如，许多企业使用互联网上不常见的专业术语，因此 LLMs 可能不知道，而且还有许多术语可以以多种方式翻译。例如，“开源”在西班牙语中可以是“Código abierto”或“Fuente abierta”；两种都可以，但最好选择一个并在单个文档中坚持使用。
- **术语表的使用和实现。** 给定一个术语表，最好的包含方式是什么？
- **不同语言的评估。** 它的性能在不同语言中如何变化？是否有改变使其对特定源语言或目标语言更有效？（注意，对于 MT 系统正在接近的非常高的性能水平，我们不确定 BLEU 是否是一个好指标。）此外，它在资源较少的语言上的性能需要进一步研究。
- **错误分析。** 我们发现指定语言和国家/地区（例如，“墨西哥口语西班牙语”）对我们的应用来说做得相当不错。当前方法在哪里不足？我们还特别感兴趣了解其在专业主题（如法律、医学）或特殊文本类型（如电影字幕）上的性能，以了解其局限性。
- **更好的评估。** 最后，我们认为更好的评估（评估）是一个巨大且重要的研究课题。与其他生成自由文本的 LLM 应用一样，当前的评估指标似乎不够充分。例如，即使在我们代理工作流程在文档级别上更好地捕获上下文和术语，导致人类评估者更倾向于选择我们的翻译而不是当前的商业产品，使用 [FLORES](https://github.com/facebookresearch/flores) 数据集在句子级别上的评估结果显示，代理系统在 BLEU 上得分较低。我们能否设计更好的指标（也许使用 LLM 来评估翻译？），这些指标能够在文档级别上捕捉翻译质量，更好地与人类偏好相关？

## 相关工作

一些学术研究小组也开始关注基于 LLM 和代理翻译。我们认为这个领域还处于早期阶段！
- *ChatGPT MT: Competitive for High- (but not Low-) Resource Languages*, Robinson 等人 (2023), https://arxiv.org/pdf/2309.07423
- *How to Design Translation Prompts for ChatGPT: An Empirical Study*, Gao 等人 (2023), https://arxiv.org/pdf/2304.02182v2
- *Beyond Human Translation: Harnessing Multi-Agent Collaboration for Translating Ultra-Long Literary Texts*, Wu 等人 (2024), https://arxiv.org/pdf/2405.11804
