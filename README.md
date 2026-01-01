# EM-algorithm
## 概要
指定した平均，共分散から自動で3次元のデータ点をプロットし，このデータ群のクラスタリングをおこなうプログラム．    

## パラメータ
|変数名|意味|初期状態|
|-----|-----|----|
| K | 初期クラスタ数 | 3 |
| mu | クラスタの混合係数 |  |
| Sigma | クラスタの平均の重み |  |
| N | 点群のデータ数 | 300 |
| pi |  |  |
| Nk | 任意のデータがk番目のクラスタに属する確率 |  |

## プログラムの内容
1. Estep 
次の式より任意の点nが任意のクラスタkに属する事後分布γnkを計算する.  
<img width="213" height="50" alt="Image" src="https://github.com/user-attachments/assets/7e0cc076-a08b-4524-9df3-93e93a11745e" />


2. Mstep  
Estepで求めたγを使いクラスタに含まれるデータ数であるNkを求める．  
<img width="95" height="53" alt="Image" src="https://github.com/user-attachments/assets/e4060751-1f3a-4b4c-a1f6-1f490461ac66" />
Nkを利用し各クラスタの大きさ，平均値，共分散行列を求める．
<img width="67" height="38" alt="Image" src="https://github.com/user-attachments/assets/6209f60f-b6da-4193-aa0a-1fffa8b5ef52" /> 
<img width="144" height="53" alt="Image" src="https://github.com/user-attachments/assets/440dda7a-223f-4e61-8468-164a307d9589" />

<img width="285" height="53" alt="Image" src="https://github.com/user-attachments/assets/1ed5096a-e331-4fe8-902b-4b050c882b94" />

1,2を繰り返すことにより，クラスタがそれぞれの点群データのまとまりによるため，クラスタリングが可能となる.  

## 実行例
```
git clone https://github.com/flyzll/propablistic-robotics
cd propablistic-robotics
python3 EM.py
```
## 出力例
元データの平均，共分散の値とクラスタリングによって推定された平均，共分散の値が表示され，点群のプロットされた3Dグラフが表示される．
```
[True Cluster 0]
Mean (mu):
[2 2 2]
Covariance (Sigma):
[[0.3 0.1 0. ]
 [0.1 0.3 0. ]
 [0.  0.  0.3]]

[True Cluster 1]
Mean (mu):
[ 2 10  3]
Covariance (Sigma):
[[0.3 0.  0. ]
 [0.  0.3 0. ]
 [0.  0.  0.3]]

[True Cluster 2]
Mean (mu):
[10 14 15]
Covariance (Sigma):
[[0.3 0.  0. ]
 [0.  0.3 0. ]
 [0.  0.  0.3]]
 ESTIMATED PARAMETERS (EM)

[Estimated Cluster 0]
Mean (mu):
[1.99759794 2.05910253 2.01317875]
Covariance (Sigma):
[[0.31338306 0.12594718 0.05183585]
 [0.12594718 0.29118478 0.01052965]
 [0.05183585 0.01052965 0.34005566]]

[Estimated Cluster 1]
Mean (mu):
[ 2.00374584 10.0011836   2.99060717]
Covariance (Sigma):
[[0.32420717 0.0338489  0.00692252]
 [0.0338489  0.30951406 0.01289545]
 [0.00692252 0.01289545 0.29932963]]

[Estimated Cluster 2]
Mean (mu):
[ 9.99857077 13.99902366 14.98500867]
Covariance (Sigma):
[[ 0.40756215  0.02513629 -0.04712495]
 [ 0.02513629  0.38794651  0.03372123]
 [-0.04712495  0.03372123  0.30021336]]
```
<img width="700" height="700" alt="Image" src="https://github.com/user-attachments/assets/89f53841-471b-47ca-9553-fdb4f0b7f88e" />

