import re
import unicodedata
from pathlib import Path

import pandas as pd
from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

def preprocess_ingredients(ingredient_list):
    # 把配料表拼接为字符串
    text = " ".join(ingredient_list)
    # 替换 'half & half' 为 'halfandhalf'
    text = text.replace('half & half', 'halfandhalf')
    # 替换 '-' 为空格
    text = text.replace('-', '')
    # 字符标准化
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # 转小写
    text = text.lower()
    # 单词级处理
    words = []
    for word in text.split():
        # 剔除包含数字的单词
        if re.findall('[0-9]', word):
            continue
        # 剔除长度 <= 2 的单词
        if len(word) <= 2:
            continue
        # 剔除包含单引号的单词
        # if "'" in word or "’" in word:
        #     continue
        if "’" in word:
            continue
        # 词形还原
        word = lemma.lemmatize(word)
        # 把处理后的单词加入列表 words
        if len(word) > 0:
            words.append(word)
    # 把 words 中的单词拼接为字符串返回
    return ' '.join(words)

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_json(DATA_DIR / 'train.json')
    test_df = pd.read_json(DATA_DIR / 'test.json')

    print(f"原始训练集大小: {len(train_df)}")

    # 新增 num_ingredients 属性表示当前菜谱的配料数量
    train_df['num_ingredients'] = train_df['ingredients'].apply(len)
    # 剔除配料表食材数 <= 2 的食谱
    train_df = train_df[train_df['num_ingredients'] > 2].copy()

    print(f"过滤后训练集大小 (配料数 > 2): {len(train_df)}")

    train_df['ingredients_clean'] = train_df['ingredients'].apply(preprocess_ingredients)
    test_df['ingredients_clean'] = test_df['ingredients'].apply(preprocess_ingredients)

    train_output_path = DATA_DIR / 'train_cleaned.json'
    test_output_path = DATA_DIR / 'test_cleaned.json'

    train_df.to_json(train_output_path, orient='records', force_ascii=False, indent=2)
    print(f"训练集数据处理完毕，已保存至 {train_output_path}。")
    test_df.to_json(test_output_path, orient='records', force_ascii=False, indent=2)
    print(f"测试集数据处理完毕，已保存至 {test_output_path}。")
