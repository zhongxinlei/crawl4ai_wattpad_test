import csv
import spacy
from tqdm.asyncio import tqdm_asyncio
from transformers import pipeline
from googletrans import Translator
import torch
import os
from litellm import completion
import asyncio
from pathlib import Path
from  utuil import *

raw_novel_path = 'raw_novel_test/'
os.environ['LITELLM_LOG'] = 'DEBUG'
comic_descriptions_chunk_size = 700
polish_chunk_size = 1500
Kay_role_name = 'Jay'

class WattpadProcessor:
    def __init__(self):
        # 初始化NLP模型
        self.nlp = spacy.load("en_core_web_lg")

        # 初始化翻译器（使用备用方案）
        self.translator = Translator()

        self.raw_translator = pipeline("translation_en_to_zh",
                     model="/home/lane/ai/models/nlp/translation/Helsinki-NLP-en-zh")

        # 初始化漫画描述生成器
        self.summarizer = pipeline(
            "summarization",
            # model="facebook/bart-large-cnn",
            model="/home/lane/ai/models/facebook/bart",
            device=1 if torch.cuda.is_available() else -1
        )

        # 初始化翻译记忆库
        self.translations = self._load_translations()

    def _load_translations(self):
        """加载已有的翻译记忆"""
        translations = {}
        try:
            with open('names/translations.csv', 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        translations[row[0]] = row[1]
        except FileNotFoundError as e:
            print(f'_load_translations error! {str(e)}')
            pass
        return translations


    async def process_entities(self, text):
        """处理实体识别和翻译"""
        doc = self.nlp(text)
        new_translations = {}

        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'GPE']:
                # 更新翻译记忆库
                ent_text = unicode_clean(ent.text)
                if ent_text not in self.translations:
                    zh_name = await self._translate_name(ent_text)
                    self.translations[ent_text] = zh_name
                    new_translations[ent_text] = zh_name

                # 保存人物描述
                if ent.label_ == 'PERSON':
                    await self._save_character_description(ent_text, text)

        # 保存新增翻译
        if new_translations:
            self._save_translations(new_translations)

        return self.translations

    async def _translate_name(self, name):
        """翻译专有名词（带缓存）"""
        try:
            translated_name = await self.translator.translate(name,src='en',dest='zh-cn')
            return translated_name.text
        except Exception as e:
            print(f'_translate_name exception: {e}')
            return name

    async def _save_character_description(self, name, context):
        """保存人物描述"""
        desc = await self._extract_description(context, name)
        text = ''
        with open('names/character_descriptions.txt', 'r') as f:
            text = f.read()
        if txt_contains_no_name_sentence(text, f'[{name}]'):
            with open('names/character_descriptions.txt', 'a') as f2:
                print(f'_save_character_description: {desc}')
                desc = desc.replace('\n', ';')
                f2.write(f"[{name}] {desc}\n")

    async def _extract_description(self, text, name):
        """提取人物描述上下文"""
        # 定义提示词

        prompt = f"""
            you are a expert in charactor extraction, please extract the detail description of {name} in the given text,
            including it's appearance, dressing and charactor.
            output shall be a full sentence and shall be less then 100 letters.
            text：
            {text}
            """
        # api_key=None, api_base="http://localhost:11434", model="ollama/llama3.3:70b", max_tokens=20000
        # api_key = "sk-b12ef3f919ab4f54ae1905e645762a40", api_base = "https://api.deepseek.com/v1", model = "deepseek/deepseek-chat", max_tokens=8192
        description = await self._call_llm_api(
            prompt,
            api_key=None, api_base="http://localhost:11434", model="ollama/llama3.3:70b", max_tokens=20000
        )
        return description if description else "no desc"

    def _save_translations(self, new_translations):
        """保存新增翻译"""
        with open('names/translations.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            for en, zh in new_translations.items():
                writer.writerow([en, zh])
        # after renew translation, need to load translation again
        self.translations = self._load_translations()

    async def generate_comic_descriptions(self, text, chapter_num):
        """生成漫画描述"""
        chunks = self._chunk_text(text, comic_descriptions_chunk_size)
        descriptions = []

        for chunk in chunks:
            try:
                output = await asyncio.to_thread(
                    self.summarizer,
                    chunk,
                    max_length=100,
                    min_length=50,
                    do_sample=True,
                    num_beams=4
                )
                descs = output[0]['summary_text']
                charactor_desc = contain_in_name_translation(descs, self.translations)
                descriptions.append(charactor_desc)
            except Exception as e:
                print(f"生成描述失败: {str(e)}")

        self._save_descriptions(descriptions, chapter_num)

    def _chunk_text(self, text, chunk_size):
        """智能分块文本"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        char_count = 0

        for sent in sentences:
            sent.replace('\n', ';').replace('I ', f'{Kay_role_name} ').replace('Jayyy', f'{Kay_role_name}').replace('I\'ll', f'{Kay_role_name} will')
            sent_len = len(sent)
            if char_count + sent_len > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                char_count = 0
            current_chunk.append(sent)
            char_count += sent_len

        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def _chunk_polished(self, text, chunk_size):
        """智能分块文本"""
        paragraphs = re.split(r'\n+', text)  # 按一个或多个换行符分割
        paragraphs = [p.strip() for p in paragraphs]  # 去除每段首尾空白
        paragraphs = [p for p in paragraphs if p]  # 过滤空段落
        chunks = []
        current_para = ''
        char_count = 0

        for paragraph in paragraphs:
            sent_len = len(paragraph)
            if char_count + sent_len > chunk_size:
                chunks.append(current_para)
                current_para = ''
                char_count = 0
            if current_para == '':
                current_para = paragraph + '\n\n'
            else:
                current_para = current_para +'\n\n' + paragraph
            char_count += sent_len

        if current_para:
            chunks.append(current_para)
        return chunks

    def _save_descriptions(self, descriptions, chapter_num):
        """保存漫画描述"""
        filename = f"comic_descriptions/{chapter_num}_ComicDescription.txt"
        with open(filename, 'w') as f:
            for desc in descriptions:
                f.write(f"{desc}\n")

    async def translate_content(self, text, chapter_num):
        """翻译章节内容"""
        translated = ''
        paragraphs = text.split('\n\n')

        for para in paragraphs:
            if not para.strip():
                continue

            # 替换专有名词
            translated_para = para
            for en, zh in self.translations.items():
                translated_para = translated_para.replace(en, zh)
            if not has_chinese_or_english(translated_para):
                print(f'translated_para has no chinese, english or number: {translated_para}')
                continue

            # 翻译段落
            try:
                # result = await self.translator.translate(translated_para, dest='zh-cn')
                prompt = f"""
                            you are a expert in translator from english to Chinese, please translate the given text,
                            if there is any non English in the given text, please do not translate them and keep them there without change.
                            if there is any line that can not be translated in the given text, please ignore it and return a space.
                            output must be the translated text directly, do not give anything else, do not include `<think>` part.
                            output must be Simplified Chinese Mandarin.
                            text：
                            {translated_para}
                            """
                # api_key=None, api_base="http://localhost:11434", model="ollama/llama3.3:70b", max_tokens=20000
                # api_key = "sk-b12ef3f919ab4f54ae1905e645762a40", api_base = "https://api.deepseek.com/v1", model = "deepseek/deepseek-chat", max_tokens=8192
                result = await self._call_llm_api(
                    prompt,
                    api_key=None, api_base="http://localhost:11434", model="ollama/llama3.3:70b", max_tokens=20000
                )
                if '翻译' in result:
                    print('warn!!!!!!!!!!!!!!!!!!!!!!!!!!!!, 句子中含有翻译字样')

                if result and len(result) > 0:
                    if translated == '':
                        translated = result
                    else:
                        translated = translated + '\n\n' + result
            except Exception as e:
                print(f"翻译失败: {str(e)}")
                # translated.append(para)

        self._save_translation(translated, chapter_num)

    def _save_translation(self, content, chapter_num):
        """保存翻译结果"""
        filename = f"translated/{chapter_num}_translated.txt"
        clean_content = process_polished_text(content)
        with open(filename, 'w') as f:
            f.write(clean_content)

    async def polish_translation(self, chapter_num):
        """润色翻译结果（需替换为实际API调用）"""
        input_file = f"translated/{chapter_num}_translated.txt"

        try:
            with open(input_file, 'r') as f:
                text = f.read()

            polished = ''
            chunks = self._chunk_polished(text, polish_chunk_size)
            for chunk in chunks:
                # 调用大模型API（示例使用伪代码）
                # api_key=None, api_base="http://localhost:11434", model="ollama/llama3.3:70b", max_tokens=20000
                # api_key = "sk-b12ef3f919ab4f54ae1905e645762a40", api_base = "https://api.deepseek.com/v1", model = "deepseek/deepseek-chat", max_tokens=8192
                output = await self._call_llm_api(
                    f"""
                        Please meticulously refine the following machine-translated content into polished Chinese prose that adheres to professional literary standards. The final output must:
                            only contain the refined part, do not include any other explanation
                            Maintain the original narrative intent and key details
                            Restructure awkward syntax to achieve natural flow
                            Optimize word choice using context-appropriate vocabulary
                            Enhance readability through proper pacing and paragraph transitions
                            Preserve any intentional stylistic elements from the source material
                            Eliminate translationese artifacts while ensuring semantic fidelity
                            must be Simplified Chinese Mandarin
                        Special Requirements:
                            Apply Chinese novel-writing conventions for dialogue formatting and descriptive passages
                            Verify cultural references for local relevance
                            Balance formal/informal registers according to scene context.
                        text：\n{chunk}",
                    """,
                    api_key=None, api_base="http://localhost:11434", model="ollama/llama3.3:70b", max_tokens=20000
                )

                if polished == '':
                    polished = output + '\n\n'
                else:
                    polished = polished + '\n\n' + output

            clean_polished = process_polished_text(polished)

            # 生成标题
            title = await self._call_llm_api(
                f"Generate a contextually appropriate Chinese title for the provided content. title shall be no more than 15 characters：\n{clean_polished}",
                api_key=None, api_base="http://localhost:11434", model="ollama/llama3.3:70b", max_tokens=20000
            )
            clean_title = process_polished_text(title)
            clean_title = re.sub(r'[\\/:*?"<>|]', '', clean_title).strip()
            clean_title = clean_title.replace('《','').replace('》', '')
            # 保存结果
            output_file = f"polished/{chapter_num}_{clean_title}.txt"
            with open(output_file, 'w') as f:
                f.write(f"\t\t{clean_title}\n\n{clean_polished}")

        except Exception as e:
            print(f"润色失败: {str(e)}")

    async def _call_llm_api(self, prompt, api_key, api_base, model, max_tokens):
        try:
            # 配置 litellm 参数
            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max_tokens,
                api_key=api_key,
                api_base=api_base
            )

            # 返回生成的文本内容
            return response.choices[0].message.content

        except Exception as e:
            return f"API 调用失败: {str(e)}"


async def main(chapter_name, chapter_content, processor):
    tasks = []
    print(f"\n📖 正在处理第 {chapter_name} 章（{len(chapter_content)} 字符）")
    # await processor.process_entities(chapter_content)
    # tasks.append(asyncio.create_task(processor.generate_comic_descriptions(chapter_content, chapter_name)))
    # tasks.append(asyncio.create_task(processor.translate_content(chapter_content, chapter_name)))
    tasks.append(asyncio.create_task(processor.polish_translation(chapter_name)))
    with tqdm_asyncio(total=len(tasks), desc="处理进度") as pbar:
        for task in asyncio.as_completed(tasks):
            await task
            pbar.update(1)

    print("\n✅ 处理完成！结果保存在 crawl_wattpad/ 目录中")


if __name__ == "__main__":
    processor = WattpadProcessor()
    print(f'processor created: {processor}')
    folder_path = Path(raw_novel_path)
    for file_path in sorted(folder_path.glob("*")):
        try:
            # 使用utf-8编码打开文件
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # 如果utf-8解码失败，尝试gbk编码
            try:
                with open(file_path, "r", encoding="gbk") as f:
                    content = f.read()
            except Exception as e:
                print(f"无法读取文件 {file_path.name}: {str(e)}")
        except Exception as e:
            print(f"读取文件 {file_path.name} 时发生错误: {str(e)}")
        asyncio.run(main(str(file_path.name), content, processor))
