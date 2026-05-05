# Dự đoán thời tiết với Hidden Markov Model

Bài tập lớn môn học — dự đoán mưa ngày mai (`RainTomorrow`) từ dữ liệu thời tiết Úc sử dụng mô hình Hidden Markov Model (HMM).

## Dataset

| Thuộc tính | Giá trị |
|---|---|
| Nguồn | weatherAUS.csv |
| Số quan sát | 142,193 dòng |
| Số địa điểm | 49 địa điểm tại Úc |
| Khoảng thời gian | 2008 – 2017 |
| Mục tiêu | `RainTomorrow` (Yes / No) |

## Phương pháp: Gaussian Hidden Markov Model

### Ý tưởng

Thời tiết thực tế tuân theo một chuỗi Markov ẩn — mỗi ngày nằm trong một **chế độ thời tiết tiềm ẩn** (hidden state) không quan sát trực tiếp được, nhưng tạo ra các quan sát đo lường được (nhiệt độ, độ ẩm, áp suất...).

```
Hidden:   Dry → Dry → Mild → Humid → Stormy → ...
             ↓     ↓     ↓      ↓       ↓
Observed: [T,H,P] [T,H,P] [T,H,P] [T,H,P] [T,H,P]
```

### Cấu trúc mô hình

- **Hidden states** (4 trạng thái): Dry / Mild / Humid / Stormy — model tự học từ dữ liệu
- **Observations**: 11 feature liên tục → dùng **Gaussian HMM** (mỗi state có một phân phối Gaussian đa chiều)
- **Tham số học được**:
  - `π` — xác suất trạng thái ban đầu
  - `A` — transition matrix: `A[i,j] = P(state_t+1 = j | state_t = i)`
  - `μ, Σ` — mean và covariance của Gaussian emission cho mỗi state

### Features sử dụng

```
MinTemp, MaxTemp, Rainfall,
Humidity9am, Humidity3pm,
Pressure9am, Pressure3pm,
Temp9am, Temp3pm,
WindSpeed9am, WindSpeed3pm
```

### Huấn luyện: Baum-Welch (EM Algorithm)

Không dùng gradient descent hay epoch như neural network. Baum-Welch là một dạng **EM algorithm** đảm bảo log-likelihood tăng sau mỗi vòng lặp:

- **E-step**: Tính xác suất trạng thái ẩn (forward-backward algorithm)
- **M-step**: Cập nhật `π`, `A`, `μ`, `Σ` để tối đa hoá log-likelihood
- Lặp đến khi hội tụ (`n_iter=100`, `tol=1e-2`)

### Dự đoán

**Phương pháp 1 — Direct Mapping**:

```
P(Rain | ngày t) = P(Rain | state_t)
```

Dùng trực tiếp xác suất mưa thực nghiệm của trạng thái hiện tại.

**Phương pháp 2 — Transition Prediction** *(chính)*:

```
P(Rain | ngày t+1) = Σ_j  A[state_t, j] × P(Rain | state_j)
```

Nhân transition matrix với xác suất mưa của từng trạng thái kế tiếp — phản ánh đúng bản chất chuỗi Markov.

### Giải mã chuỗi trạng thái: Viterbi

Tìm chuỗi trạng thái ẩn có xác suất cao nhất cho toàn bộ chuỗi quan sát:

```
state_1*, state_2*, ..., state_T* = argmax P(states | observations)
```

## Kết quả

| Phương pháp | Accuracy | ROC-AUC |
|---|---|---|
| Direct Mapping (threshold=0.5) | ~77% | ~0.67 |
| Transition Predict (threshold=0.5) | ~77% | ~0.67 |
| Transition Predict (threshold tối ưu) | ~78% | ~0.67 |

## Cấu trúc thư mục

```
.
├── weatherAUS.csv       # Dataset gốc
├── weather_hmm.ipynb    # Notebook chính
└── README.md
```

## Chạy notebook

```bash
pip install hmmlearn scikit-learn matplotlib seaborn pandas numpy
jupyter notebook weather_hmm.ipynb
```
