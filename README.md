# EM-algorithm
## 概要
指定した平均，共分散から自動で3次元のデータ点をプロットし，このデータ群のクラスタリングをおこなうプログラム．  
クラスタを100個に設定しており，クラスタの平均位置はランダムにしている．  
EM法を用いることでクラスタ数を減らしていき，クラスタの中心位置の変更が閾値以下になった場合自動で動作が終了しプロットをおこなう．  

## Parameter
|変数名|意味|初期状態|
|-----|-----|----|
| k |初期クラスタ数 | 100|
| threshold | クラスタ削除の閾値 | 0.01|
| tol | クラスタ中心の移動量の閾値 | 1e-3| 
| stable_limit | 収束判定に必要な連続回数（Δmu < tol） | 3 |

## 実行例
```
git clone https://github.com/flyzll/propablistic-robotics
cd propablistic-robotics
python3 EM.py
```
## 出力例
EMalgorithmの試行回数(iter)と残存したクラスタの中心の数(K)が表示され3Dのプロットがおこなわれる．  
```
iter 114: K = 5   
```
<img width="634" height="615" alt="Image" src="https://github.com/user-attachments/assets/f8a4d4c3-ce6d-4e7a-801d-7c8c14ca4387" />

