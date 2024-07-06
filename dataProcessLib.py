import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def checkZScore(combined_features_array):
    # 計算統計數據
    means = np.mean(combined_features_array, axis=0)
    std_devs = np.std(combined_features_array, axis=0)

    # 檢查標準差是否為零
    zero_std_mask = std_devs == 0

    # 對於標準差不為零的特徵，計算 z 分數
    if not zero_std_mask.all():
        z_scores = np.abs((combined_features_array[:, ~zero_std_mask] - means[~zero_std_mask]) / std_devs[~zero_std_mask])
        outliers = np.where(z_scores > 3)
        if len(outliers[0]) > 0:
            print("存在異常值")
        else:
            print("沒有異常值")
    else:
        print("所有特徵的標準差都為零，無法計算 z 分數")

def visualData(combined_features_array):
    plt.figure(figsize=(10, 5))

    # 繪製每個特徵的直方圖
    for i in range(combined_features_array.shape[1]):
        plt.subplot(1, combined_features_array.shape[1], i+1)
        plt.hist(combined_features_array[:, i], bins=20)
        plt.title(f'Feature {i+1}')

    plt.tight_layout()
    plt.show()

    # 繪製散點圖
    plt.scatter(range(len(combined_features_array)), combined_features_array[:, 0])  # 假設我們只視覺化第一個特徵
    plt.xlabel('Sample index')
    plt.ylabel('Feature value')
    plt.title('Scatter Plot of Feature 1')
    plt.show()

def doPCA1(combined_features):
    # 假設 combined_features 是您所有特徵的 numpy 陣列
    # combined_features.shape 應該是 (n_samples, n_features)

    # 標準化特徵
    scaler = StandardScaler()
    combined_features_scaled = scaler.fit_transform(combined_features)

    # 初始化PCA模型，並設定要降至的維度數，例如降至2維
    pca = PCA(n_components=2)
    # 對特徵數據進行降維
    combined_features_reduced = pca.fit_transform(combined_features_scaled)

    # 現在 combined_features_reduced 是降維後的數據，其形狀應該是 (n_samples, 2)

    # 繪製降維後的數據
    plt.figure(figsize=(10, 8))
    plt.scatter(combined_features_reduced[:, 0], combined_features_reduced[:, 1], alpha=0.7)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of Combined Features')
    plt.grid(True)
    plt.show()

def doPCAWeight(all_features):
    pca = PCA(n_components='mle')  # 'mle'可以自动选择组件数
    pca.fit(all_features)

    explained_variance = pca.explained_variance_ratio_
    components = pca.components_
    # 输出每个主成分的解释方差比例
    print("Explained variance ratio:", explained_variance)
    # 输出每个主成分的系数
    print("PCA components:\n", components)
    return explained_variance,components[0]

def doPCA2(instructor_features, student_features):
    # 結合教練和學生的特徵
    combined_features = np.vstack((instructor_features, student_features))

    # 標準化特徵
    scaler = StandardScaler()
    combined_features_scaled = scaler.fit_transform(combined_features)

    # 初始化PCA模型，並設定要降至的維度數，例如降至2維
    pca = PCA(n_components=2)   
    # 對特徵數據進行降維
    combined_features_reduced = pca.fit_transform(combined_features_scaled)

    # 分離降維後的教練和學生數據
    instructor_data_reduced = combined_features_reduced[:len(instructor_features), :]
    student_data_reduced = combined_features_reduced[len(instructor_features):, :]

    # 繪製降維後的數據
    plt.figure(figsize=(10, 8))
    plt.scatter(instructor_data_reduced[:, 0], instructor_data_reduced[:, 1], alpha=0.7, label='Instructor')
    plt.scatter(student_data_reduced[:, 0], student_data_reduced[:, 1], alpha=0.7, label='Student', marker='x')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of Combined Features')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualData2(combined_features_array):
    for i in range(combined_features_array.shape[1]):
        plt.figure(figsize=(10, 2))
        plt.scatter(range(len(combined_features_array)), combined_features_array[:, i])
        plt.title(f"Feature {i+1}")
        plt.xlabel("Sample index")
        plt.ylabel("Value")
        plt.show()
