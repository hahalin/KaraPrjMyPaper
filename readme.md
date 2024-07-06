# 結合電腦視覺和機器學習於運動動作時序資料分析：以空手道「型」之學習為例

本專案實現了一個結合電腦視覺與機器學習技術的系統，用於分析與評估空手道 Kata 動作。透過動態時間扭曲（Dynamic Time Warping, DTW）算法，比對學生與教練的動作相似度，提供客觀的評價標準。

## 專案介紹

本專案的目的是使用電腦視覺技術（如MediaPipe）提取人體關鍵點資訊，並結合機器學習技術進行動作評估，主要步驟如下：

1. **資料收集**：收集教練與學生的Kata動作影片。
2. **資料處理**：使用MediaPipe提取關鍵點，並計算動作特徵向量。
3. **特徵分析**：使用主成分分析（PCA）進行特徵權重計算。
4. **動作比對**：使用DTW算法計算動作相似度，並進行視覺化展示。

## 系統需求

- Python 3.10+
- OpenCV
- Matplotlib
- NumPy
- fastdtw
- PIL (Pillow)
- scipy

## 安裝

請確保已安裝上述依賴庫，可使用以下指令安裝：

```bash
pip install opencv-python matplotlib numpy fastdtw pillow scipy
```

## 使用說明
1. 收集資料
確保教練與學生的Kata影片已經準備好，並放置於指定的目錄中。例如：
```
c:\video\kata\TT1.mp4 # 教練影片
c:\video\kata\SF.mp4  # 學生影片
```
2. 執行kataMain.py

## 結果展示

運行程式後，將顯示教練與學生之間的動作對齊路徑與相似度，透過視覺化圖表了解學生與教練動作的差異。

![image]([https://raw.githubusercontent.com/hahalin/KaraPrjMyPaper/main/images/1-4.png)
![image]([https://user-images.githubusercontent.com/2748761/166239060-917e486c-9113-4cec-9192-bbc45c94544e.png](https://raw.githubusercontent.com/hahalin/KaraPrjMyPaper/main/images/dtwD.png)

## 本專案文獻

請參考論文《結合電腦視覺和機器學習於運動動作時序資料分析：以空手道「型」之學習為例》，了解更多技術細節與研究成果。

感謝您對本專案的關注，期待您的貢獻與建議！

希望這份README文件能夠幫助您將程式碼順利上傳至GitHub並與他人分享。如有任何問題或需要進一步協助，請隨時告知。




