# GNN 節點分類 — GraphSAGE & GAT

基於 PyTorch Geometric 從零實作兩種經典圖神經網路架構 — **GraphSAGE** 與 **GAT** — 並在 CORA 引用網路資料集上進行節點分類評測。

兩個層皆直接繼承 PyG 的 `MessagePassing` 基礎類別實作，未使用任何內建的 conv 封裝。

---

## 專案目的

本專案的核心目標是**深入理解圖神經網路的訊息傳遞機制（Message Passing）**，而非僅呼叫現成函式庫。

具體而言：

- 手動實作 `message()`、`aggregate()`、`forward()` 三個核心函數，了解節點如何從鄰居聚合資訊
- 比較兩種不同的聚合策略：GraphSAGE 的**平均聚合 + skip connection**，與 GAT 的**注意力加權聚合**
- 以 CORA 節點分類任務作為標準 benchmark，驗證兩種實作的正確性與效果
- 提供可複用的模組化架構，方便日後插入新的 GNN 層進行實驗

---

## 模型介紹

### GraphSAGE（[Hamilton et al., 2017](https://arxiv.org/abs/1706.02216)）

透過平均聚合鄰居嵌入，並加入 skip connection（殘差連接）：

$$h_v^{(l)} = W_l \cdot h_v^{(l-1)} + W_r \cdot \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} h_u^{(l-1)}$$

每層輸出後進行 L-2 正規化。

### GAT（[Veličković et al., 2018](https://arxiv.org/abs/1710.10903)）

透過共享前饋網路學習邊的注意力權重，支援多頭注意力機制：

$$\alpha_{ij} = \text{softmax}_j\!\Big(\text{LeakyReLU}\!\Big(\vec{a}_l^\top W_l \vec{h}_i + \vec{a}_r^\top W_r \vec{h}_j\Big)\Big)$$

$$\vec{h}_i' = \|_{k=1}^{K}\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} W_r^{(k)} \vec{h}_j$$

---

## 專案結構

```
.
├── models/
│   ├── graphsage.py      # GraphSAGE 卷積層
│   ├── gat.py            # GAT 卷積層（多頭注意力）
│   ├── gnn_stack.py      # 通用 GNN 堆疊（可插拔層）
│   └── __init__.py
├── utils/
│   ├── trainer.py        # 訓練迴圈與評估
│   ├── optimizer.py      # 優化器 / 排程器工廠
│   └── __init__.py
├── train.py              # CLI 訓練入口
├── SMM_Assignment_2.py   # 原始單檔版本（保留參考用）
├── requirements.txt
└── .gitignore
```

---

## 安裝

```bash
# 1. 複製專案
git clone https://github.com/<your-username>/gnn-node-classification.git
cd gnn-node-classification

# 2. 安裝 PyTorch 相關套件
#    請確認 CUDA 版本與 torch-scatter / torch-sparse 的 wheel 一致
#    以下以 CUDA 12.1 + PyTorch 2.2 為例：
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-geometric

# 3. 安裝其他依賴
pip install -r requirements.txt
```

---

## 使用方式

### 同時訓練並比較兩個模型

```bash
python train.py --model all --epochs 500
```

### 單獨訓練指定模型

```bash
python train.py --model GraphSage
python train.py --model GAT
```

### 主要參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model` | `all` | 選擇模型：`GraphSage` / `GAT` / `all` |
| `--dataset` | `cora` | 資料集名稱（目前僅支援 `cora`） |
| `--epochs` | `500` | 訓練總 epoch 數 |
| `--hidden_dim` | `32` | 隱藏層維度 |
| `--num_layers` | `2` | GNN 層數 |
| `--dropout` | `0.6` | Dropout 比例 |
| `--lr` | `0.01` | 學習率 |
| `--opt` | `adam` | 優化器（`adam` / `sgd` / `rmsprop` / `adagrad`） |
| `--save_plot` | `results.png` | 結果圖片儲存路徑 |

---

## 資料集

**CORA** — 學術論文引用網路：
- 節點 = 論文（2,708 篇）
- 邊 = 無向引用關係（5,429 條）
- 節點特徵 = 詞袋表示（1,433 維）
- 類別 = 7 個研究主題

首次執行時會透過 `torch_geometric.datasets.Planetoid` 自動下載。

---

## 預期結果

| 模型 | 驗證集準確率 |
|------|-------------|
| GraphSAGE | ~81% |
| GAT（2 heads） | ~82% |

| 比較面向 | 說明 |
|----------|------|
| 收斂速度 | GraphSAGE 通常收斂較快，GAT 前期震盪較大 |
| 最終準確率 | GAT 憑藉注意力機制通常略優於 GraphSAGE |
| 過擬合傾向 | Dropout 0.6 有效抑制，兩者訓練曲線皆穩定下降 |
