import unicodedata
import re
from pathlib import Path

def unicode_clean(text):
    """
    处理所有 Unicode 控制字符和非常规空格

    参数:
    text (str): 原始字符串

    返回:
    str: 清理后的字符串
    """
    # 移除控制字符
    cleaned = ''.join(
        ch for ch in text
        if not unicodedata.category(ch).startswith('C')
    )
    # 替换非常规空格
    return re.sub(r'\s+', ' ', cleaned).strip()


def txt_contains_no_name_sentence(text, name, ignore_case=True):
    # 智能分割句子（处理中英文标点）
    sentences = text.split('\n')
    print(f'txt_contains_no_name_sentence: {name} - {sentences}')

    # 检查每个句子
    for sentence in sentences:
        if name in sentence:
            return False
    return True

def contain_in_name_translation(line, translations):
    ls = re.findall(r'\w+[\']*\w*|\.', line)
    for l in ls:
        for item in translations:
            if l == item:
                return f'[{l}] {line}'
    return f'[Jay] {line}'


def has_chinese_or_english(s):
    # 检查是否包含中文
    if re.search(r'[\u4e00-\u9fff]', s):
        return True
    # 检查是否包含英文字母或数字
    if re.search(r'[a-zA-Z0-9]', s):
        return True
    # 都不包含则返回False
    return False


def process_polished_text(text: str) -> str:
    """
    处理文本：去标签 + 按句子换行

    参数：
    text: 原始文本（可能含HTML/XML标签）

    返回：
    处理后的文本，每句话间隔两个换行符
    """

    # 阶段 1: 精准移除标签及内容
    sanitized = re.sub(
        r'''
            <                      # 标签起始
            [^>]+                  # 标签名及属性 (非>字符)
            >                      # 标签结束
            (?:.*?</[^>]+>)?       # 匹配闭合标签内容 (非贪婪)
            |                      # 或匹配自闭合标签
            <[^>]+/?>              # 自闭合标签 (如 <br/>)
            ''',
        '',
        text,
        flags=re.DOTALL | re.VERBOSE
    )

    # 阶段2：去除所有尖括号标签
    clean_text = re.sub(r'<[^>]+>', '', sanitized)

    # 阶段3：分割句子（支持中英文标点）
    # 正则解释：(?<=[.!?。！？]) 匹配标点后的位置
    # [\s　]* 匹配零或多个半角/全角空格
    sentences = re.split(r'[\r\n]+', clean_text)

    # 阶段4：清洗数据
    processed = []
    for sent in sentences:
        # 去除每句话首尾空格并过滤空内容
        stripped = sent.strip()
        if stripped:
            processed.append(stripped)

    # 阶段4：组装最终结果
    return '\n\n'.join(processed).strip()


# 能从文件夹中遍历文件，找到英文字符并列出来，同时列出来出现在哪个文件
def find_english_characters(folder_path, file_extensions=None):
    """
    在文件夹中查找包含英文字符的文件，并列出这些字符及其出现的文件
    参数:
        folder_path (str): 要搜索的文件夹路径
        file_extensions (list, optional): 要搜索的文件扩展名列表，如['.txt', '.py']。如果为None，则搜索所有文件
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"错误: {folder_path} 不是一个有效的文件夹路径")
        return
    # 匹配英文字符（包括大小写字母）
    english_char_pattern = re.compile(r'[a-zA-Z]')
    results = {}
    # 遍历文件夹中的所有文件
    for file_path in folder.rglob('*'):
        if file_path.is_file():
            # 检查文件扩展名
            if file_extensions and file_path.suffix.lower() not in file_extensions:
                continue
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    english_chars = []
                    # 检查每个字符
                    for i, char in enumerate(content):
                        if english_char_pattern.match(char):
                            english_chars.append((i, char))
                    if english_chars:
                        results[str(file_path)] = english_chars
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
    # 输出结果
    if results:
        print("找到包含英文字符的文件:")
        for file_path, chars in results.items():
            print(f"\n文件: {file_path}")
            print("英文字符位置和字符:")
            for pos, char in chars:
                print(f"  位置 {pos}: '{char}'")
    else:
        print("在文件夹中未找到包含英文字符的文件")

# 能从文件夹中遍历文件，找到所有的指定字符串和出现的文件名，如果需要将所有指定字符串替换成目标字符串
def find_and_replace_in_files(folder_path, search_string, replace_string=None, file_extensions=None):
    """
    在文件夹中查找包含特定字符串的文件，并可选择替换该字符串
    参数:
        folder_path (str): 要搜索的文件夹路径
        search_string (str): 要查找的字符串
        replace_string (str, optional): 要替换成的字符串。如果为None，则只查找不替换
        file_extensions (list, optional): 要搜索的文件扩展名列表，如['.txt', '.py']。如果为None，则搜索所有文件
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"错误: {folder_path} 不是一个有效的文件夹路径")
        return
    matched_files = []
    # 遍历文件夹中的所有文件
    for file_path in folder.rglob('*'):
        if file_path.is_file():
            # 检查文件扩展名
            if file_extensions and file_path.suffix.lower() not in file_extensions:
                continue
            # 查找包含搜索字符串的文件
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    if search_string in content:
                        matched_files.append(str(file_path))
                        # 如果需要替换
                        if replace_string is not None:
                            # 备份原始文件
                            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                            file_path.rename(backup_path)
                            # 读取备份文件内容并进行替换
                            with open(backup_path, 'r', encoding='utf-8', errors='ignore') as backup_file:
                                backup_content = backup_file.read()
                            new_content = backup_content.replace(search_string, replace_string)
                            # 写入新文件
                            with open(file_path, 'w', encoding='utf-8') as new_file:
                                new_file.write(new_content)
                            print(f"已替换: {file_path} (原始文件备份为: {backup_path})")
                            #删除备份文件
                            backup_path.unlink()
                            print(f'备份文件已删除{backup_path}')
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
    # 输出结果
    if matched_files:
        print("\n找到包含 '{}' 的文件:".format(search_string))
        for file in matched_files:
            print(f"  - {file}")
        if replace_string is not None:
            print(f"\n已将所有 '{search_string}' 替换为 '{replace_string}'")
    else:
        print(f"在文件夹中未找到包含 '{search_string}' 的文件")


if __name__ == "__main__":
    find_and_replace_in_files('./raw_novel_test', 'Ace')
    # find_english_characters('./polished')