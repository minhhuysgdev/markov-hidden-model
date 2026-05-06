# Dự đoán giá cổ phiếu FPT ngắn hạn với Hidden Markov Model

Dữ liệu cổ phiếu **FPT Corporation** crawl từ API VCI qua thư viện `vnstock`,
làm đầu vào cho mô hình **Discrete HMM** dự đoán xu hướng giá ngày tiếp theo.

---

## 1. Dataset

### Nguồn dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| Mã cổ phiếu | `FPT` |
| Nguồn chính | VCI (fallback: TCBS) |
| Thư viện | `vnstock` 3.5.0 |
| Khoảng thời gian | 2020-01-01 - 2025-03-28 |
| So phien giao dich | 1,369 ngay |
| Tham chieu thi truong | VN-Index (VNINDEX.csv) |

### Cac file CSV

**FPT.csv** - Gia & khoi luong (OHLCV)

| Cot | Kieu | Mo ta |
|---|---|---|
| `time` | date | Ngay giao dich |
| `open` | float | Gia mo cua (VND) |
| `high` | float | Gia cao nhat |
| `low` | float | Gia thap nhat |
| `close` | float | Gia dong cua |
| `volume` | int | Khoi luong giao dich |
| `return` | float | Loi suat ngay: close(t)/close(t-1) - 1 |

**FPT_balance_sheet.csv** - Bang can doi ke toan (theo quy)

| Cot chinh | Y nghia |
|---|---|
| `LIABILITIES (Bn. VND)` | Tong no phai tra |
| `OWNER'S EQUITY(Bn.VND)` | Von chu so huu |
| `Common shares (Bn. VND)` | Gia tri co phan (= so CP x menh gia 10,000 VND) |

**FPT_income_statement.csv** - Bao cao KQKD (theo quy)

| Cot chinh | Y nghia |
|---|---|
| `Revenue YoY (%)` | Tang truong doanh thu so voi cung ky |
| `Net Sales` | Doanh thu thuan |
| `Gross Profit` | Loi nhuan gop |
| `Net Profit For the Year` | Loi nhuan rong |

---

## 2. Mo hinh: Discrete Hidden Markov Model

### Y tuong

Thi truong chung khoan khong bieu hien trang thai noi tai mot cach truc tiep --
moi phien giao dich la bien hien cua mot **che do thi truong an** (hidden state):
tang manh, tich luy, di ngang, hoac sup giam. HMM mo hinh hoa chinh xac dieu nay.

```
Hidden:   Uptrend -> Uptrend -> Sideways -> Downtrend -> Downtrend -> ...
               |          |          |            |             |
Observed:   [obs]      [obs]      [obs]        [obs]         [obs]
                     (10 nhan to da roi rac hoa)
```

### Roi rac hoa quan sat

10 nhan to lien tuc duoc roi rac hoa thanh **ky hieu quan sat roi rac** (M ky hieu)
qua K-Means clustering hoac chia theo quantile:

- Moi ngay giao dich duoc anh xa thanh 1 ky hieu `o_t` trong {0, 1, ..., M-1}
- HMM hoc phan phoi phat sinh `B[i,k] = P(obs=k | state=i)` cho tung trang thai

### Tham so mo hinh

| Ky hieu | Y nghia | Kich thuoc |
|---|---|---|
| `pi[i]` | Xac suat trang thai ban dau | N |
| `A[i,j]` | Transition matrix: P(state_{t+1}=j | state_t=i) | N x N |
| `B[i,k]` | Emission matrix: P(obs=k | state=i) | N x M |

### Cac thuat toan tu implement

| Thuat toan | Mo ta |
|---|---|
| Scaled Forward | Tinh alpha_hat[t,i] voi he so ti le c[t] -- tranh numerical underflow |
| Scaled Backward | Tinh beta_hat[t,i] dung cung he so c[t] tu forward |
| Baum-Welch (EM) | E-step: tinh gamma, xi -- M-step: cap nhat pi, A, B |
| Viterbi | Log-domain -- tim chuoi trang thai an co xac suat cao nhat |

### Huan luyen (Baum-Welch)

```
E-step:  gamma[t,i]   = P(q_t=i | O, lambda)
         xi[t,i,j]    = P(q_t=i, q_{t+1}=j | O, lambda)

M-step:  pi[i]   <- gamma[0, i]
         A[i,j]  <- sum_t xi[t,i,j] / sum_t gamma[t,i]
         B[i,k]  <- sum_{t: obs_t=k} gamma[t,i] / sum_t gamma[t,i]
```

Lap den khi hoi tu (`n_iter=100`, `tol=1e-4`).
Tren tap train: Chia 80/20 theo thoi gian (tren 1,095 phien dau).

### Du doan xu huong gia ngay tiep theo

**Direct Mapping** -- dung xac suat tang/giam thuc nghiem cua state hien tai:

```
P(Up | ngay t) = P(return_{t+1} > 0 | state_t)    (hoc tu tap train)
```

**Transition Prediction** -- nhan transition matrix de nhin ve ngay mai:

```
P(Up | ngay t+1) = sum_j  A[state_t, j] x P(Up | state_j)
```

### Giai thich cac trang thai an

Sau khi huan luyen, moi hidden state tu nhien tuong ung voi mot che do thi truong:

| State | P(return > 0) | Dac trung |
|---|---|---|
| S0 | Thap (~30%) | Che do giam -- FPT mat gia lien tuc |
| S1 | Cao (~70%) | Che do tang -- FPT hoi phuc manh |
| S2 | Trung binh (~50%) | Di ngang, bien dong nho |
| S3 | Rat cao (>75%) | Che do bung pha -- khoi luong lon, momentum cao |

> Cac gia tri tren la uoc tinh sau huan luyen. Ket qua thuc te phu thuoc vao du lieu.

---

## 4. Ket qua ky vong

Chia train/test theo thoi gian (80/20):

| Phuong phap | Accuracy | ROC-AUC |
|---|---|---|
| Baseline (luon du doan Tang) | ~55% | 0.50 |
| Direct Mapping (threshold=0.5) | ~60-65% | ~0.60-0.65 |
| Transition Prediction | ~60-65% | ~0.60-0.65 |

> Du bao ngan han tren thi truong chung khoan la bai toan kho. HMM cung cap tin hieu xu huong (regime detection) hon la du bao chinh xac tung phien.

---

## 5. Cau truc thu muc

```
.
|-- crawl_vn_data.ipynb          # Notebook crawl & tinh nhan to
|-- weather_hmm.ipynb            # Notebook HMM du doan
|-- FPT.csv                      # Gia OHLCV (1,369 phien)
|-- FPT_balance_sheet.csv        # Bang can doi ke toan (theo quy)
|-- FPT_income_statement.csv     # Bao cao KQKD (theo quy)
|-- FPT_factors.csv              # 10 nhan to (theo ngay)
|-- VNINDEX.csv                  # VN-Index tham chieu (1,369 phien)
`-- README.md
```

## 6. Cai dat & chay

```bash
pip install vnstock scikit-learn matplotlib seaborn pandas numpy
jupyter notebook crawl_vn_data.ipynb   # Buoc 1: crawl du lieu
jupyter notebook weather_hmm.ipynb     # Buoc 2: huan luyen & du doan
```

Notebook crawl tu dong bo qua neu file CSV da ton tai.
HMM duoc implement tu dau bang NumPy -- khong can cai `hmmlearn`.
