# RL Gridworld: Value Iteration vs. Random Policy

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)

本專案為強化學習（Reinforcement Learning）基礎作業，實現了一個基於 **Flask** 後端與 **NumPy** 運算的網格世界 (Gridworld) 互動系統。使用者可以透過網頁介面即時設定地圖，並觀察不同演算法下的價值函數收斂與策略導向。

## 🌟 功能特點
* **互動式地圖編輯**：支援 5x5 至 9x9 網格生成，可手動放置 **起點 (START)**、**終點 (END)** 與 **障礙物 (Wall)**。
* **HW1-2: 隨機策略評估 (Random Policy Evaluation)**：
    * 模擬代理人在地圖中隨機移動（上下左右機率各 25%）。
    * 觀察在低效率路徑下，價值函數如何因撞牆懲罰與長距離移動而呈現較低數值。
* **HW1-3: 價值迭代 (Value Iteration)**：
    * 採用 Bellman Optimality Equation 進行迭代計算。
    * 自動推導出「最優策略 (Optimal Policy)」，並以箭頭即時顯示最短路徑。

## 📊 實驗結果分析
| 演算法 | 箭頭表現 | 價值分佈 (Value Distribution) |
| :--- | :--- | :--- |
| **隨機策略評估** | 方向隨機亂跳，無固定邏輯 | 數值普遍偏低（多為負數），反映出隨機碰撞的代價。 |
| **價值迭代** | 精確指向終點，並自動避開障礙物 | 數值向終點遞增（梯度明顯），展現出收斂後的最優路徑。 |

## 🛠️ 如何執行
1. **安裝環境依賴**：
   使用 `requirements.txt` 安裝所需的 Python 套件：
   ```bash
   pip install -r requirements.txt
2. **啟動伺服器**：
   ```bash
   python app.py
   ```
3. **開啟網頁**：
   在瀏覽器中訪問 `http://127.0.0.1:5000`。
   (也可以使用render進行demo:https://drl-hw1-75wv.onrender.com)

## 📁 專案架構
app.py: Flask Web 伺服器，處理前端請求與 API 銜接。

logic.py: 核心演算法實現（包含 Value Iteration 與 Policy Evaluation）。

templates/index.html: 前端互動介面與網格渲染邏輯。

requirements.txt: 專案套件清單。