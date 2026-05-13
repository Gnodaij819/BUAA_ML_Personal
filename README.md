# ML_Personal

本项目用于完成基于配料列表的菜系分类任务。仓库包含数据清洗、探索性分析、传统机器学习建模、Wide & Deep 实验，以及用于记录过程的 Notebook 与提交结果文件。

## 项目目标

给定一道菜的配料列表 `ingredients`，预测其所属菜系 `cuisine`。

## 目录结构

```text
ML_Personal/
├── data/
│   ├── train.json              # 原始训练数据
│   ├── test.json               # 原始测试数据
│   ├── train_cleaned.json      # 清洗后的训练数据（由 clean_data.py 生成）
│   └── test_cleaned.json       # 清洗后的测试数据（由 clean_data.py 生成）
├── submissions/
│   ├── submission1.csv
│   ├── submission2.csv
│   ├── ...
│   └── submission_TEST.csv     # 历史提交结果
├── clean_data.py               # 数据清洗与预处理
├── eda_data.py                 # 探索性数据分析
├── model.py                    # TF-IDF + SVM/One-vs-Rest 训练与预测
├── wild_deep.py                # Wide & Deep + Ensemble 实验脚本
├── cuisine-prediction.ipynb    # 主要实验笔记本
├── try.ipynb                   # 额外实验笔记本
├── best_cuisine_model.h5       # 已训练模型权重
├── requirements.txt            # Python 依赖列表
└── README.md
```

## 主要脚本说明

- `clean_data.py`
  - 读取 `data/train.json` 和 `data/test.json`
  - 对配料文本做标准化、去噪、词形还原
  - 过滤配料数过少的训练样本
  - 生成 `train_cleaned.json` 与 `test_cleaned.json`

- `eda_data.py`
  - 统计菜系分布与高频配料
  - 检查非 ASCII 字符、短词、低配料样本和缺失值
  - 用于数据理解与清洗前分析

- `model.py`
  - 基于清洗后的文本构建 `TF-IDF + SVM + OneVsRestClassifier`
  - 执行交叉验证
  - 输出预测结果到 `submissions/submission.csv`

- `wild_deep.py`
  - 构建 Wide & Deep 模型
  - 使用逻辑回归结果进行集成
  - 作为更进一步的实验性方案

## 环境依赖

推荐使用 Python 3.10。安装依赖：

```bash
pip install -r requirements.txt
```

如果需要运行 Notebook，可额外安装：

```bash
pip install jupyter
```

`clean_data.py` 使用了 `nltk.stem.WordNetLemmatizer`，首次运行前需要下载 NLTK 词典：

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## 推荐运行流程

1. 准备原始数据到 `data/train.json` 和 `data/test.json`
2. 运行数据清洗：

```bash
python clean_data.py
```

3. 如需查看数据分布，运行：

```bash
python eda_data.py
```

4. 训练传统机器学习模型并生成提交文件：

```bash
python model.py
```

5. 如需尝试深度学习与集成方案，运行：

```bash
python wild_deep.py
```
