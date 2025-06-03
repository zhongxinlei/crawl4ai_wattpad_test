from transformers import pipeline

# 加载本地翻译模型
translator = pipeline("translation_en_to_zh",
                     model="/home/lane/ai/models/nlp/translation/Helsinki-NLP-en-zh")

def local_translate(text):
    result = translator(text, max_length=400)
    return result[0]['translation_text']

print(local_translate("Apple iPhon is an excellent product"))
# 输出：Apple iPhone是很好的产品