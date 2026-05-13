import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

DATA_DIR = "data"

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_colwidth', None)

if __name__ == "__main__":
    df = pd.read_json(os.path.join(DATA_DIR, 'train.json'))

    # 菜系食谱数量分布
    cuisine_counts = df['cuisine'].value_counts()
    plt.figure(figsize=(12, 8))
    sns.barplot(y=cuisine_counts.index, x=cuisine_counts.values, palette="Paired", hue=cuisine_counts.values, dodge=False, legend=False)
    plt.title('菜系食谱数量分布')
    plt.xlabel('食谱数量')
    plt.ylabel('菜系')
    plt.tight_layout()
    plt.show()

    # 配料频次分布
    ingredients = [ingredient for sublist in df['ingredients'] for ingredient in sublist]
    ingredient_counts = Counter(ingredients)
    ingredient_names, ingredient_frequencies = zip(*ingredient_counts.most_common(20))
    plt.figure(figsize=(12, 8))
    sns.barplot(x=ingredient_frequencies, y=ingredient_names, palette="Paired", hue=ingredient_frequencies, dodge=False, legend=False)
    plt.title('配料频次分布')
    plt.xlabel('出现频次')
    plt.ylabel('配料')
    plt.tight_layout()
    plt.show()

    # 非标准化字符统计
    print("\n" + "="*30 + " 非标准化字符统计 " + "="*30)
    all_non_ascii_chars = []
    for ing in ingredients:
        non_ascii_chars = [char for char in ing if not char.isascii()]
        all_non_ascii_chars.extend(non_ascii_chars)

    non_ascii_counts = Counter(all_non_ascii_chars)
    print(f"非标准化字符及其出现次数（共{len(non_ascii_counts)}种字符）:")
    for char, count in non_ascii_counts.items():
        print(f"'{char}': {count}次")

    # 单词长度<=2的配料统计
    print("\n" + "="*30 + " 单词长度<=2的配料统计 " + "="*30)
    short_ingredients = [ing for ing in set(ingredients) if len(ing) <= 2]
    print(f"单词长度<=2的配料（共{len(short_ingredients)}个）: {short_ingredients}")

    # 配料数量较少的食谱统计
    print("\n" + "="*30 + " 配料数量较少的食谱统计 " + "="*30)

    recipes_with_1_ingredient = df[df['ingredients'].apply(len) == 1]
    print(f"配料数量=1的食谱数: {len(recipes_with_1_ingredient)}")
    print("配料数量=1的前10个食谱:")
    print(recipes_with_1_ingredient.head(10)[['cuisine', 'ingredients']])

    recipes_with_2_ingredients = df[df['ingredients'].apply(len) == 2]
    print(f"配料数量=2的食谱数: {len(recipes_with_2_ingredients)}")
    print("配料数量=2的前10个食谱:")
    print(recipes_with_2_ingredients.head(10)[['cuisine', 'ingredients']])

    recipes_with_3_ingredients = df[df['ingredients'].apply(len) == 3]
    print(f"配料数量=3的食谱数: {len(recipes_with_3_ingredients)}")
    print("配料数量=3的前10个食谱:")
    print(recipes_with_3_ingredients.head(10)[['cuisine', 'ingredients']])

    # 缺失值统计
    print("\n" + "="*30 + " 缺失值统计 " + "="*30)
    missing_values = df.isnull().sum()
    print(missing_values)
