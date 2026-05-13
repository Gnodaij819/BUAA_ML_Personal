# ML_Personal

一个基于食材列表进行菜系分类的机器学习项目。当前目录包含数据清洗、探索性分析、传统机器学习建模、Wide & Deep 尝试，以及多次提交结果，整体看起来是围绕类似 Kaggle "What's Cooking?" 这类任务推进的实验仓库。

## 项目目标

输入一道菜的配料列表（`ingredients`），预测它所属的菜系（`cuisine`）。

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
├── try.ipynb                   # 额外实验/草稿笔记本
├── best_cuisine_model.h5       # 已训练模型权重
└── README.md
```

## 主要脚本说明

- `clean_data.py`
  - 读取 `data/train.json` 和 `data/test.json`
  - 对配料文本做标准化、去噪、词形还原
  - 过滤配料数过少的训练样本
  - 生成 `train_cleaned.json` 与 `test_cleaned.json`

- `eda_data.py`
  - 查看菜系分布、常见配料分布
  - 统计非 ASCII 字符、短词、低配料样本等
  - 适合做数据理解和清洗前分析

- `model.py`
  - 基于清洗后的文本使用 `TF-IDF + SVM + OneVsRestClassifier`
  - 执行交叉验证
  - 输出预测文件到 `submissions/submission.csv`

- `wild_deep.py`
  - 使用宽模型（TF-IDF）和深模型（Embedding + Pooling）组合
  - 再与逻辑回归结果做集成
  - 属于更偏实验性的深度学习尝试

## 环境依赖

建议使用 Python 3.10+。项目里没有现成的 `requirements.txt`，按现有脚本推断，至少需要以下依赖：

```bash
pip install pandas numpy scikit-learn nltk tensorflow matplotlib seaborn h5py
```

`clean_data.py` 使用了 `nltk.stem.WordNetLemmatizer`，首次运行前还需要下载 NLTK 词典：

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## 推荐运行流程

1. 准备原始数据到 `data/train.json` 和 `data/test.json`
2. 运行数据清洗：

```bash
python clean_data.py
```

3. 如需先看数据分布，运行：

```bash
python eda_data.py
```

4. 训练传统机器学习模型并生成提交文件：

```bash
python model.py
```

5. 如需尝试深度学习/集成方案，运行：

```bash
python wild_deep.py
```

## 当前仓库特点

- 同时包含脚本版流程和 Notebook 实验过程
- `submissions/` 中保留了多次历史提交结果，便于回看迭代
- `best_cuisine_model.h5` 是训练产物，不一定是复现流程的必要输入

## 上传到 GitHub 前的建议

- 优先提交源码、笔记本和说明文档
- 训练生成的中间文件、模型权重、IDE 配置建议忽略
- 如果 `data/` 中的数据来自竞赛或外部数据源，建议在仓库中说明来源，必要时只保留目录结构，不直接上传数据文件

## 后续可补充项

- `requirements.txt` 或 `environment.yml`
- 训练结果指标对比表
- 更清晰的实验记录
- 命令行参数化，避免路径和超参数硬编码

