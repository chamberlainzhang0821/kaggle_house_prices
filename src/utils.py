import pandas as pd
import numpy as np

def load_and_clean_data():
    # 1. 读取数据
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    
    # 2. 提取标签 y (SalePrice) 并做对数转换（线性模型非常需要目标值接近正态分布）
    y_train = np.log1p(train['SalePrice'])
    
    # 3. 合并特征进行统一清洗
    X_train_raw = train.drop(['Id', 'SalePrice'], axis=1)
    X_test_raw = test.drop(['Id'], axis=1)
    
    # 记录训练集的行数，方便后面切分
    num_train = len(X_train_raw)
    
    combined = pd.concat([X_train_raw, X_test_raw], axis=0)
    
    # 4. 极简清洗：只保留数值型特征（线性回归无法直接处理文本特征）
    combined_numeric = combined.select_dtypes(include=[np.number])
    
    # 5. 极简填充缺失值：用中位数填充（线性回归不能输入 NaN）
    combined_clean = combined_numeric.fillna(combined_numeric.median())
    
    # 6. 切分回训练集特征和测试集特征
    X_train = combined_clean.iloc[:num_train]
    X_test = combined_clean.iloc[num_train:]
    
    return X_train, y_train, X_test, test['Id']