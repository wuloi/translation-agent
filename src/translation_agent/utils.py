import os
from typing import List, Union

import openai
import tiktoken
from dotenv import load_dotenv
from icecream import ic
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 读取本地 .env 文件
load_dotenv()
# 初始化 OpenAI API 客户端
client = openai.OpenAI(api_key=os.getenv("GROQ_API_KEY"),
                       base_url=os.getenv("OpenAI_Compatibility_BASE_URL"),
            )

# 定义每个文本块的最大令牌数
MAX_TOKENS_PER_CHUNK = (
    1000  # 如果文本的令牌数超过这个数值，我们将把它分成多个小块，一次翻译一个小块
)



def get_completion(
    prompt: str,
    system_message: str = "你是一个提供帮助的助手。",
    model: str = os.getenv("OpenAI_Compatibility_MODEL"),
    temperature: float = 0.3,
    json_mode: bool = False,
) -> Union[str, dict]:
    """
    使用 OpenAI API 生成一个补全。

    参数:
        prompt (str): 用户的提示或查询。
        system_message (str, 可选): 为助手设置上下文的系统消息。
            默认为 "你是一个提供帮助的助手。"。
        model (str, 可选): 用于生成补全的 OpenAI 模型的名称。
            默认为 "llama-3.1-70b-versatile"。
        temperature (float, 可选): 控制生成文本的随机性的采样温度。
            默认为 0.3。
        json_mode (bool, 可选): 是否以 JSON 格式返回响应。
            默认为 False。

    返回:
        Union[str, dict]: 生成的补全。
            如果 json_mode 为 True，则返回完整的 API 响应作为一个字典。
            如果 json_mode 为 False，则返回生成的文本作为一个字符串。
    """

    if json_mode:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            top_p=1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    else:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            top_p=1,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content


def one_chunk_initial_translation(
    source_lang: str, target_lang: str, source_text: str
) -> str:
    """
    使用大型语言模型将整个文本作为一个块进行翻译。

    参数:
        source_lang (str): 文本的源语言。
        target_lang (str): 翻译的目标语言。
        source_text (str): 要翻译的文本。

    返回:
        str: 翻译后的文本。
    """

    # 设置系统信息，指明翻译方向
    system_message = f"您是一位专业的语言学家，专注于从 {source_lang} 到 {target_lang} 的翻译。"

    # 构建翻译提示，指定翻译任务和源文本
    translation_prompt = f"""这是一段从 {source_lang} 到 {target_lang} 的翻译，请为这段文本提供 {target_lang} 的翻译。
不要提供除翻译外的任何解释或文本。
{source_lang}: {source_text}

{target_lang}:"""

    # 调用 get_completion 函数获取翻译结果
    translation = get_completion(translation_prompt, system_message=system_message)

    return translation


def one_chunk_reflect_on_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation_1: str,
    country: str = "",
) -> str:
    """
    利用大型语言模型反思翻译过程，将整个文本视为一个单一的块。

    参数:
        source_lang (str): 文本的源语言。
        target_lang (str): 翻译的目标语言。
        source_text (str): 原文。
        translation_1 (str): 源文本的初始翻译。
        country (str): 目标语言对应的国家。

    返回:
        str: 语言模型对翻译的反思，提供建设性的批评和改进建议。
    """

    system_message = f"您是专注于从 {source_lang} 翻译到 {target_lang} 的专家语言学家。您将获得一个源文本及其翻译，目标是改进翻译。"

    if country != "":
        reflection_prompt = f"""您的任务是仔细阅读从 {source_lang} 到 {target_lang} 的源文本和翻译，并给出建设性的批评和有用的建议来改进翻译。
翻译的最终风格和语调应符合在 {country} 通常说的 {target_lang}。

源文本和初始翻译由 XML 标签 <SOURCE_TEXT></SOURCE_TEXT> 和 <TRANSLATION></TRANSLATION> 界定，如下所示：

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

在写建议时，注意是否有方法改进翻译的：
(i) 准确性（通过纠正增加的错误、误译、遗漏或未翻译的文本），
(ii) 流畅性（应用 {target_lang} 的语法、拼写和标点规则，并确保没有不必要的重复），
(iii) 风格（确保翻译反映源文本的风格并考虑任何文化背景），
(iv) 术语（确保术语的使用一致并反映源文本的领域；并且只确保使用等效的 {target_lang} 成语）。

为改进翻译写一份具体、有用和建设性的建议清单。
每条建议应针对翻译的一个具体部分。
只输出建议，不要输出其他任何内容。"""

    else:
        reflection_prompt = f"""您的任务是仔细阅读从 {source_lang} 到 {target_lang} 的源文本和翻译，并给出建设性的批评和有用的建议来改进翻译。

源文本和初始翻译由 XML 标签 <SOURCE_TEXT></SOURCE_TEXT> 和 <TRANSLATION></TRANSLATION> 界定，如下所示：

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

在写建议时，注意是否有方法改进翻译的：
(i) 准确性（通过纠正增加的错误、误译、遗漏或未翻译的文本），
(ii) 流畅性（应用 {target_lang} 的语法、拼写和标点规则，并确保没有不必要的重复），
(iii) 风格（确保翻译反映源文本的风格并考虑任何文化背景），
(iv) 术语（确保术语的使用一致并反映源文本的领域；并且只确保使用等效的 {target_lang} 成语）。

为改进翻译写一份具体、有用和建设性的建议清单。
每条建议应针对翻译的一个具体部分。
只输出建议，不要输出其他任何内容。"""

    reflection = get_completion(reflection_prompt, system_message=system_message)
    return reflection


def one_chunk_improve_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation_1: str,
    reflection: str,
) -> str:
    """
    利用反思来改进翻译，将整个文本作为一个单一的块进行处理。

    参数:
        source_lang (str): 文本的源语言。
        target_lang (str): 翻译的目标语言。
        source_text (str): 源语言的原始文本。
        translation_1 (str): 源文本的初始翻译。
        reflection (str): 专家对改进翻译的建议和建设性批评。

    返回:
        str: 根据专家的建议改进后的翻译。
    """

    system_message = f"您是专注于从 {source_lang} 到 {target_lang} 的翻译编辑专家。"

    prompt = f"""您的任务是仔细阅读并编辑从 {source_lang} 到 {target_lang} 的翻译，同时考虑专家的建议和建设性批评。

源文本、初始翻译和专家的语言学家建议由 XML 标签 <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> 和 <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> 界定，如下所示：

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

在编辑翻译时，请考虑专家的建议。通过以下方式编辑翻译：

(i) 准确性（通过纠正增加的错误、误译、遗漏或未翻译的文本），
(ii) 流畅性（应用 {target_lang} 的语法、拼写和标点规则，确保没有不必要的重复），
(iii) 风格（确保翻译反映源文本的风格），
(iv) 术语（使用不当或使用不一致），
(v) 其他错误。

只输出新的翻译，不要输出其他任何内容。"""

    translation_2 = get_completion(prompt, system_message)

    return translation_2


def one_chunk_translate_text(
    source_lang: str, 
    target_lang: str, 
    source_text: str, 
    country: str = ""
) -> str:
    """
    将单一文本块从源语言翻译到目标语言。

    该函数执行一个两步翻译过程：
    1. 获取源文本的初始翻译。
    2. 反思初始翻译并生成改进后的翻译。

    参数:
        source_lang (str): 文本的源语言。
        target_lang (str): 翻译的目标语言。
        source_text (str): 要翻译的文本。
        country (str): 为目标语言指定的国家。

    返回:
        str: 源文本的改进翻译。
    """
    # 获取源文本的初始翻译
    translation_1 = one_chunk_initial_translation(
        source_lang, target_lang, source_text
    )

    # 反思初始翻译，并生成改进建议
    reflection = one_chunk_reflect_on_translation(
        source_lang, target_lang, source_text, translation_1, country
    )

    # 根据反思建议改进翻译
    translation_2 = one_chunk_improve_translation(
        source_lang, target_lang, source_text, translation_1, reflection
    )

    return translation_2


def num_tokens_in_string(
    input_str: str, 
    encoding_name: str = "cl100k_base"
) -> int:
    """
    使用指定的编码计算给定字符串中的令牌数量。

    参数:
        input_str (str): 要被分词的输入字符串。
        encoding_name (str, 可选): 要使用的编码名称。默认为 "cl100k_base"，
            这是最常用的编码器（由 GPT-4 使用）。

    返回:
        int: 输入字符串中的令牌数量。

    示例:
        >>> text = "你好，你好吗？"
        >>> num_tokens = num_tokens_in_string(text)
        >>> print(num_tokens)
        5
    """
    # 获取指定名称的编码器
    encoding = tiktoken.get_encoding(encoding_name)
    # 计算并返回输入字符串的令牌数量
    num_tokens = len(encoding.encode(input_str))
    return num_tokens


def multichunk_initial_translation(
    source_lang: str, 
    target_lang: str, 
    source_text_chunks: List[str]
) -> List[str]:
    """
    将文本分成多个块从源语言翻译到目标语言。

    参数:
        source_lang (str): 文本的源语言。
        target_lang (str): 翻译的目标语言。
        source_text_chunks (List[str]: 要翻译的文本块列表。

    返回:
        List[str]: 翻译后的文本块列表。
    """

    system_message = f"您是一位专注于从 {source_lang} 到 {target_lang} 的专业翻译专家。"

    translation_prompt = """您的任务是提供文本部分的专业翻译，从 {source_lang} 到 {target_lang}。

源文本如下，由 XML 标签 <SOURCE_TEXT> 和 </SOURCE_TEXT> 界定。只翻译源文本中由 <TRANSLATE_THIS> 和 </TRANSLATE_THIS> 界定的部分。
您可以使用其余的源文本作为上下文，但不要翻译其他文本。除了指定部分的翻译外，不要输出任何其他内容。

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

再次强调，只翻译文本的这一部分，如 <TRANSLATE_THIS> 和 </TRANSLATE_THIS> 之间所示：
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

只输出您被要求翻译的部分的翻译，不要输出其他任何内容。
"""

    translation_chunks = []
    for i in range(len(source_text_chunks)):
        # 将要翻译第 i 块
        tagged_text = (
            "".join(source_text_chunks[0:i])
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )

        prompt = translation_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            tagged_text=tagged_text,
            chunk_to_translate=source_text_chunks[i],
        )

        translation = get_completion(prompt, system_message=system_message)
        translation_chunks.append(translation)

    return translation_chunks


def multichunk_reflect_on_translation(
    source_lang: str,
    target_lang: str,
    source_text_chunks: List[str],
    translation_1_chunks: List[str],
    country: str = "",
) -> List[str]:
    """
    提供对部分翻译的建设性批评和改进建议。

    参数:
        source_lang (str): 文本的源语言。
        target_lang (str): 翻译的目标语言。
        source_text_chunks (List[str]): 分块的源文本。
        translation_1_chunks (List[str]): 与源文本块相对应的翻译块。
        country (str): 为目标语言指定的国家。

    返回:
        List[str]: 包含对每个翻译块改进建议的反思列表。
    """

    system_message = f"您是专注于从 {source_lang} 到 {target_lang} 翻译的专家语言学家。您将获得源文本及其翻译，目标是改进翻译。"

    if country != "":
        reflection_prompt = """您的任务是仔细阅读源文本和该文本的部分翻译，从 {source_lang} 翻译到 {target_lang}，并给出建设性的批评和有用的建议来改进翻译。
翻译的最终风格和语调应符合在 {country} 通常说的 {target_lang}。

源文本如下，由 XML 标签 <SOURCE_TEXT> 和 </SOURCE_TEXT> 界定，已翻译的部分在源文本中由 <TRANSLATE_THIS> 和 </TRANSLATE_THIS> 界定。您可以使用其余的源文本作为翻译部分的上下文。

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

再次强调，只有文本的部分正在被翻译，再次显示在 <TRANSLATE_THIS> 和 </TRANSLATE_THIS> 之间：
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

指定部分的翻译如下，由 <TRANSLATION> 和 </TRANSLATION> 界定：
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

在写建议时，注意是否有方法改进翻译的：
(i) 准确性（通过纠正增加的错误、误译、遗漏或未翻译的文本），
(ii) 流畅性（应用 {target_lang} 的语法、拼写和标点规则，并确保没有不必要的重复），
(iii) 风格（确保翻译反映源文本的风格并考虑任何文化背景），
(iv) 术语（确保术语的使用一致并反映源文本的领域；并只确保使用等效的 {target_lang} 成语）。

为改进翻译写一份具体、有用和建设性的建议清单。
每条建议应针对翻译的一个具体部分。
只输出建议，不要输出其他任何内容。"""

    else:
        reflection_prompt = """您的任务是仔细阅读源文本和该文本的部分翻译，从 {source_lang} 翻译到 {target_lang}，并给出建设性的批评和有用的建议来改进翻译。

源文本如下，由 XML 标签 <SOURCE_TEXT> 和 </SOURCE_TEXT> 界定，已翻译的部分在源文本中由 <TRANSLATE_THIS> 和 </TRANSLATE_THIS> 界定。您可以使用其余的源文本作为翻译部分的上下文。

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

再次强调，只有文本的部分正在被翻译，再次显示在 <TRANSLATE_THIS> 和 </TRANSLATE_THIS> 之间：
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

指定部分的翻译如下，由 <TRANSLATION> 和 </TRANSLATION> 界定：
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

在写建议时，注意是否有方法改进翻译的：
(i) 准确性（通过纠正增加的错误、误译、遗漏或未翻译的文本），
(ii) 流畅性（应用 {target_lang} 的语法、拼写和标点规则，并确保没有不必要的重复），
(iii) 风格（确保翻译反映源文本的风格并考虑任何文化背景），
(iv) 术语（确保术语的使用一致并反映源文本的领域；并只确保使用等效的 {target_lang} 成语）。

为改进翻译写一份具体、有用和建设性的建议清单。
每条建议应针对翻译的一个具体部分。
只输出建议，不要输出其他任何内容。"""

    reflection_chunks = []
    for i in range(len(source_text_chunks)):
        # 将翻译第 i 块
        tagged_text = (
            "".join(source_text_chunks[0:i])
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )
        if country != "":
            prompt = reflection_prompt.format(
                source_lang=source_lang,
                target_lang=target_lang,
                tagged_text=tagged_text,
                chunk_to_translate=source_text_chunks[i],
                translation_1_chunk=translation_1_chunks[i],
                country=country,
            )
        else:
            prompt = reflection_prompt.format(
                source_lang=source_lang,
                target_lang=target_lang,
                tagged_text=tagged_text,
                chunk_to_translate=source_text_chunks[i],
                translation_1_chunk=translation_1_chunks[i],
            )

        reflection = get_completion(prompt, system_message)
        reflection_chunks.append(reflection)

    return reflection_chunks


def multichunk_improve_translation(
    source_lang: str,
    target_lang: str,
    source_text_chunks: List[str],
    translation_1_chunks: List[str],
    reflection_chunks: List[str],
) -> List[str]:
    """
    通过考虑专家的建议来改进源语言到目标语言的文本翻译。

    参数:
        source_lang (str): 文本的源语言。
        target_lang (str): 翻译的目标语言。
        source_text_chunks (List[str]): 分成块的源文本。
        translation_1_chunks (List[str]): 每个块的初始翻译。
        reflection_chunks (List[str]): 专家对每个翻译块的改进建议。

    返回:
        List[str]: 每个块的改进翻译。
    """

    system_message = f"您是专注于从 {source_lang} 到 {target_lang} 翻译编辑的专家语言学家。"

    improvement_prompt = """您的任务是仔细阅读并改进从 {source_lang} 到 {target_lang} 的翻译，同时考虑专家的建议和建设性批评。下面提供了源文本、初始翻译和专家的建议。

源文本如下，由 XML 标签 <SOURCE_TEXT> 和 </SOURCE_TEXT> 界定，已翻译的部分在源文本中由 <TRANSLATE_THIS> 和 </TRANSLATE_THIS> 界定。您可以使用其余的源文本作为上下文，但只需提供由 <TRANSLATE_THIS> 和 </TRANSLATE_THIS> 指示部分的翻译。

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

再次强调，只有文本的部分正在被翻译，再次显示在 <TRANSLATE_THIS> 和 </TRANSLATE_THIS> 之间：
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

指定部分的翻译如下，由 <TRANSLATION> 和 </TRANSLATION> 界定：
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

专家对指定部分的翻译建议如下，由 <EXPERT_SUGGESTIONS> 和 </EXPERT_SUGGESTIONS> 界定：
<EXPERT_SUGGESTIONS>
{reflection_chunk}
</EXPERT_SUGGESTIONS>

考虑到专家的建议，请重写翻译以改进它，注意是否有方法改进翻译的：

(i) 准确性（通过纠正增加的错误、误译、遗漏或未翻译的文本），
(ii) 流畅性（应用 {target_lang} 的语法、拼写和标点规则，并确保没有不必要的重复），
(iii) 风格（确保翻译反映源文本的风格），
(iv) 术语（在上下文中不合适或使用不一致），或
(v) 其他错误。

只输出指定部分的新翻译，不要输出其他任何内容。"""

    translation_2_chunks = []
    for i in range(len(source_text_chunks)):
        # 将翻译第 i 块
        tagged_text = (
            "".join(source_text_chunks[0:i])
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )

        prompt = improvement_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            tagged_text=tagged_text,
            chunk_to_translate=source_text_chunks[i],
            translation_1_chunk=translation_1_chunks[i],
            reflection_chunk=reflection_chunks[i],
        )

        translation_2 = get_completion(prompt, system_message)
        translation_2_chunks.append(translation_2)

    return translation_2_chunks


def multichunk_translation(
    source_lang, target_lang, source_text_chunks, country: str = ""
):
    """
    基于初始翻译和反思，改进多个文本块的翻译。

    参数:
        source_lang (str): 文本块的源语言。
        target_lang (str): 翻译的目标语言。
        source_text_chunks (List[str]): 需要翻译的源文本块列表。
        translation_1_chunks (List[str]): 每个源文本块的初始翻译列表。
        reflection_chunks (List[str]): 对初始翻译的反思列表。
        country (str): 目标语言指定的国家
    返回:
        List[str]: 每个源文本块的改进翻译列表。
    """

    translation_1_chunks = multichunk_initial_translation(
        source_lang, target_lang, source_text_chunks
    )

    reflection_chunks = multichunk_reflect_on_translation(
        source_lang,
        target_lang,
        source_text_chunks,
        translation_1_chunks,
        country,
    )

    translation_2_chunks = multichunk_improve_translation(
        source_lang,
        target_lang,
        source_text_chunks,
        translation_1_chunks,
        reflection_chunks,
    )

    return translation_2_chunks


def calculate_chunk_size(token_count: int, token_limit: int) -> int:
    """
    根据令牌总数和令牌限制来计算块的大小。

    参数:
        token_count (int): 令牌的总数。
        token_limit (int): 每个块允许的最大令牌数。

    返回:
        int: 计算出的块大小。

    描述:
        该函数基于给定的令牌总数和令牌限制来计算块的大小。
        如果令牌总数小于或等于令牌限制，则函数将返回令牌总数作为块大小。
        否则，它将计算在令牌限制内容纳所有令牌所需的块数。
        块大小由令牌限制除以块数来确定。
        如果在令牌总数除以令牌限制后还有剩余的令牌，
        则通过将剩余的令牌除以块数来调整块大小。

    示例:
        >>> calculate_chunk_size(1000, 500)
        500
        >>> calculate_chunk_size(1530, 500)
        389
        >>> calculate_chunk_size(2242, 500)
        496
    """

    if token_count <= token_limit:
        return token_count

    num_chunks = (token_count + token_limit - 1) // token_limit
    chunk_size = token_count // num_chunks

    remaining_tokens = token_count % token_limit
    if remaining_tokens > 0:
        chunk_size += remaining_tokens // num_chunks

    return chunk_size


def translate(
    source_lang,
    target_lang,
    source_text,
    country,
    max_tokens=MAX_TOKENS_PER_CHUNK,
):
    """将 source_text 从 source_lang 翻译到 target_lang."""

    # 计算输入文本的令牌数
    num_tokens_in_text = num_tokens_in_string(source_text)

    ic(num_tokens_in_text)

    # 如果文本的令牌数小于最大令牌限制，作为一个整体块进行翻译
    if num_tokens_in_text < max_tokens:
        ic("将文本作为一个单独的块进行翻译")

        final_translation = one_chunk_translate_text(
            source_lang, target_lang, source_text, country
        )

        return final_translation

    else:
        ic("将文本分成多个块进行翻译")

        # 计算每个块应有的令牌数
        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=max_tokens
        )

        ic(token_size)

        # 使用 RecursiveCharacterTextSplitter 对文本进行分块
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=token_size,
            chunk_overlap=0,
        )

        source_text_chunks = text_splitter.split_text(source_text)

        # 对每个文本块进行多步翻译过程
        translation_2_chunks = multichunk_translation(
            source_lang, target_lang, source_text_chunks, country
        )

        # 将所有翻译后的块拼接成最终翻译结果
        return "".join(translation_2_chunks)
