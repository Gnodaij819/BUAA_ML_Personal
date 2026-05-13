import pandas as pd
import numpy as np
import re
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, callbacks, regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # 引入 LR

# ==========================================
# 🔧 GPU 路径修复 (保持不动)
# ==========================================
conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix:
    try:
        os.add_dll_directory(os.path.join(conda_prefix, 'Library', 'bin'))
    except:
        pass

# ==========================================
# 1. 配置与初始化
# ==========================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==========================================
# 2. 数据加载与高级清洗
# ==========================================
print("\n[Step 1] Loading Data & Cleaning...")
train_df = pd.read_json('data/train.json')
test_df = pd.read_json('data/test.json')

# 使用你之前那套高级清洗逻辑 (这里简化展示，请保留你的 final_clean_pipeline)
# 假设你已经运行过清洗逻辑，或者这里用简单的代替（建议用你那个高级版的）
# -----------------------------------------------------------------
# ⚠️ 请确保 train_df['clean_text'] 是你用 nltk 清洗后的结果
# 这里为了代码完整性，我放一个简易版，你最好替换回你的高级版
STOP_WORDS = set(['oz', 'lb', 'tsp', 'tbsp', 'cup'])  # 核心噪音手动加进去


def simple_clean(ingredients):
    # 【核心修复】判断输入是否为列表
    # 如果是列表 ['salt', 'pepper'] -> 变成字符串 "salt pepper"
    if isinstance(ingredients, list):
        text = ' '.join(ingredients)
    else:
        # 如果万一已经是字符串了，直接用
        text = str(ingredients)

    # 现在 text 肯定是字符串了，可以安全地调用 .lower()
    text = text.lower()

    # 正则替换：只保留小写字母和空格
    text = re.sub(r'[^a-z\s]', '', text)

    # 再次过滤停用词，并重新拼接
    return ' '.join([w for w in text.split() if w not in STOP_WORDS and len(w) > 2])


# 应用修复后的函数
print("正在清洗文本...")
train_df['clean_text'] = train_df['ingredients'].apply(simple_clean)
test_df['clean_text'] = test_df['ingredients'].apply(simple_clean)

# -----------------------------------------------------------------

# ==========================================
# 3. 特征工程 (关键优化)
# ==========================================
print("\n[Step 2] Feature Engineering...")

# --- A. Wide: TF-IDF (更严格的过滤) ---
print("构建 Wide 特征 (min_df=10)...")
tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2),  # 保持 N-gram
    min_df=10,  # 【狠招】少于10次的词直接扔掉，杀掉 'red', 'white' 这种噪音
    binary=True
)
X_wide_all = tfidf.fit_transform(train_df['clean_text']).toarray()
X_wide_test = tfidf.transform(test_df['clean_text']).toarray()
REAL_WIDE_DIM = X_wide_all.shape[1]

# --- B. Deep: Sequences ---
print("构建 Deep 特征...")
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(train_df['clean_text'])
vocab_size = len(tokenizer.word_index) + 1

train_seqs = tokenizer.texts_to_sequences(train_df['clean_text'])
test_seqs = tokenizer.texts_to_sequences(test_df['clean_text'])
max_len = 50  # 稍微放宽一点

X_deep_all = pad_sequences(train_seqs, maxlen=max_len, padding='post', truncating='post')
X_deep_test = pad_sequences(test_seqs, maxlen=max_len, padding='post', truncating='post')

# --- C. Labels ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(train_df['cuisine'])
y_categorical = to_categorical(y_encoded)
num_classes = y_categorical.shape[1]

# 划分验证集
X_deep_train, X_deep_val, X_wide_train, X_wide_val, y_train, y_val = train_test_split(
    X_deep_all, X_wide_all, y_categorical, test_size=0.15, random_state=SEED, stratify=y_encoded
)


# ==========================================
# 4. 构建模型 (Max Pooling 升级版)
# ==========================================
def build_model():
    # --- Deep ---
    input_deep = Input(shape=(max_len,), name='deep_input')
    d = layers.Embedding(vocab_size, 48)(input_deep)

    # 【核心升级】从 Average 改为 Max Pooling
    # 就像雷达一样，只探测“最强特征”，忽略平庸的水分
    d = layers.GlobalMaxPooling1D()(d)

    d = layers.Dense(64, activation='relu')(d)
    d = layers.Dropout(0.5)(d)
    deep_out = layers.Dense(32, activation='relu')(d)

    # --- Wide ---
    input_wide = Input(shape=(REAL_WIDE_DIM,), name='wide_input')
    # 加入 L2 正则化，防止过拟合
    w = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(input_wide)
    w = layers.Dropout(0.5)(w)
    wide_out = layers.Dense(32, activation='relu')(w)

    # --- Output ---
    combined = layers.concatenate([deep_out, wide_out])
    c = layers.Dense(64, activation='relu')(combined)
    c = layers.Dropout(0.5)(c)
    output = layers.Dense(num_classes, activation='softmax', dtype='float32')(c)

    model = Model(inputs=[input_deep, input_wide], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


print("\n[Step 3] Training Wide & Deep Model...")
model = build_model()
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)

model.fit(
    x=[X_deep_train, X_wide_train], y=y_train,
    epochs=25, batch_size=256,  # GPU 加速
    validation_data=([X_deep_val, X_wide_val], y_val),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ==========================================
# 5. 终极必杀：集成学习 (Ensemble)
# ==========================================
print("\n[Step 4] Ensemble Strategy (NN + LR)...")

# 1. 获取 NN 的预测
nn_probs = model.predict([X_deep_test, X_wide_test])

# 2. 训练一个强力的 LR 模型
print("正在训练辅助 LR 模型...")
# C=0.8 是经验参数，稍强正则化
lr = LogisticRegression(C=0.8, solver='liblinear', max_iter=1000, random_state=SEED)
lr.fit(X_wide_all, y_encoded)  # 使用全部数据训练 LR
lr_probs = lr.predict_proba(X_wide_test)

# 3. 融合！(通常 NN 看得深，LR 看得准，0.5/0.5 或 0.6/0.4)
final_probs = 0.55 * nn_probs + 0.45 * lr_probs

y_pred_indices = final_probs.argmax(axis=1)
y_pred_text = label_encoder.inverse_transform(y_pred_indices)

# 生成提交
submission = pd.DataFrame({'id': test_df['id'], 'cuisine': y_pred_text})
filename = 'submission_final_ensemble.csv'
submission.to_csv(filename, index=False)

print(f"\n🏆 最终融合完成！结果已保存: {filename}")
print(submission.head())