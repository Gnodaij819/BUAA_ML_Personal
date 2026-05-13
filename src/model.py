from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"

if __name__ == "__main__":
	# 读取清洗后的数据集
	train_df = pd.read_json(DATA_DIR / 'train_cleaned.json', orient='records')
	test_df = pd.read_json(DATA_DIR / 'test_cleaned.json', orient='records')

	vectorizer = make_pipeline(
		# 使用 TF-IDF 将清洗后的文本转换为数值向量
		TfidfVectorizer(sublinear_tf=True, stop_words='english'),
		# TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=3),
		# 使用函数转换器将生成的稀疏矩阵转换为 float32 类型
		FunctionTransformer(lambda x: x.astype('float32'), validate=False),
	)

	# 提取训练集文本特征
	x_train = vectorizer.fit_transform(train_df['ingredients_clean'])
	# 对生成的稀疏矩阵内部存储的列索引按升序排列
	x_train.sort_indices()
	# 使用在训练集上学习到的词汇表和 IDF 权重来转换测试集文本
	x_test = vectorizer.transform(test_df['ingredients_clean'].values)
	# 对 cuisine 列进行编码
	label_encoder = LabelEncoder()
	y_train = label_encoder.fit_transform(train_df['cuisine'])

	# 定义支持向量分类器
	estimator = SVC(
		C=5,
		kernel='rbf',
		gamma='scale',
		tol=0.001,
		cache_size=1000,
	)

	# 定义多分类器模型
	classifier = OneVsRestClassifier(
		estimator,
		n_jobs=-1,
		verbose=5
	)

	scores = cross_validate(
		classifier,
		x_train,
		y_train,
		cv=3,
		verbose=5
	)
	scores['test_score'].mean()

	# 训练模型
	classifier.fit(x_train, y_train)
	# 对训练集进行预测，并把预测结果由数字转换为菜系名称
	y_pred = label_encoder.inverse_transform(classifier.predict(x_train))
	# 训练集实际结果
	y_true = label_encoder.inverse_transform(y_train)

	print(f'训练集预测准确率: {accuracy_score(y_true, y_pred):.2%}')

	# 对测试集进行预测
	y_pred = label_encoder.inverse_transform(classifier.predict(x_test))
	test_df['cuisine'] = y_pred
	SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
	test_df[['id', 'cuisine']].to_csv(SUBMISSIONS_DIR / 'submission.csv', index=False)
	print("预测结果已输出到 submissions/submission.csv")

	# # 网格搜索最优参数
	# base_svc = SVC(
	# 	kernel='rbf',
	# 	C=5,
	# 	tol=0.001,
	# 	cache_size=1000
	# )
	#
	# classifier = OneVsRestClassifier(
	# 	base_svc,
	# 	n_jobs=1,
	# 	verbose=3
	# )
	#
	# param_grid = {
	# 	'estimator__gamma': ['scale', 'auto', 0.01, 0.1, 1, 2]
	# }
	#
	# grid_search = GridSearchCV(
	# 	estimator=classifier,
	# 	param_grid=param_grid,
	# 	scoring='accuracy',
	# 	cv=3,
	# 	n_jobs=-1,
	# 	verbose=3
	# )
	#
	# grid_search.fit(x_train, y_train)
	#
	# print("最优参数：")
	# print(grid_search.best_params_)
	#
	# print(f"交叉验证最优准确率: {grid_search.best_score_:.4f}")
