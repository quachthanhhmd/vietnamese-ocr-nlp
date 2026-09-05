# Vietnamese OCR — hợp phiếu PaddleOCR + VietOCR + TrOCR

Nhận dạng chữ Việt trên ảnh, sử dụng voting trên 3 models

| ID | Framework | Checkpoint | Bộ giải mã |
|----|-----------|------------|------------|
| M1 | PaddleOCR | `latin_PP-OCRv5_mobile_rec` fine-tune | CTC, song song |
| M2 | VietOCR | `vgg_transformer` fine-tune | tự hồi quy, mức ký tự |
| M3 | TrOCR | `microsoft/trocr-base-printed` fine-tune | tự hồi quy, mức token BPE |

## Kết quả trên `rec_test.txt` (n = 3.003)

| Hệ thống | acc | norm_edit_dis | CER | WER |
|---|---:|---:|---:|---:|
| M1 PaddleOCR | 0.6577 | 0.9778 | 0.0195 | 0.0667 |
| M2 VietOCR | 0.6617 | 0.9771 | 0.0195 | 0.0624 |
| M3 TrOCR | 0.6307 | 0.9757 | 0.0209 | 0.0683 |
| Hợp phiếu mức dòng | 0.6757 | 0.9789 | 0.0186 | 0.0606 |
| **Hợp phiếu mức ký tự** | **0.6780** | **0.9799** | **0.0171** | **0.0573** |

`acc` là so khớp nguyên dòng sau khi chuẩn hoá NFC và bỏ khoảng trắng
(`ignore_space = True`). Đo trên Kaggle, 2× NVIDIA T4, seed 2026.

## Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps vietocr==0.3.13      # tránh vietocr ghì Pillow về bản cũ
pip install paddlepaddle-gpu==3.0.0        # hoặc paddlepaddle==3.0.0 nếu chạy CPU
```

Yêu cầu Python ≥ 3.10.

**Tốt nhất nên chạy bằng python của venv** để tránh lỗi version với phiên bản hiện tại trên máy:

```bash
.venv/bin/python run_eval.py
```
## Dữ liệu và checkpoint

Tải về và trỏ đường dẫn trong `configs/default.yaml`:

```
data/vi_rec_100k/       rec_train.txt (93.997)  rec_val.txt (3.000)
                        rec_test.txt  (3.003)   vi_dict.txt (233 ký tự)
                        train/  val/  test/

checkpoints/paddleocr-custom-det/   model.pdparams   (M1)
checkpoints/vietocr_finetune/       vietocr_finetune.zip finetune.yaml    (M2)
checkpoints/trocr_model/            config.json, model.safetensors, tokenizer.json (M3)
```

* Dữ liệu: do giảng viên cấp
* Checkpoint: `<ĐIỀN LINK GOOGLE DRIVE / HUGGINGFACE>`

Nhãn có dạng `<đường_dẫn_ảnh>\t<text>`, đường dẫn tính tương đối từ `data_dir`.

## Chạy

**Nhận dạng một ảnh bất kỳ** 

```bash
python predict.py duong/dan/anh.jpg
python predict.py thu_muc_anh/ --save results/demo.jsonl
python predict.py anh.jpg --models VietOCR TrOCR      # bỏ qua model 1 vài model
```

In ra dự đoán của từng model và của cả hai cách hợp phiếu, kèm lý do chốt phiếu.

**Đánh giá toàn tập test** — sinh mọi bảng và hình cho báo cáo:

```bash
python run_eval.py                      # ba model + hợp phiếu
python run_eval.py --baseline           # thêm PaddleOCR gốc (mục 2.1)
python run_eval.py --limit 200          # chạy thử nhanh
python run_eval.py --from-predictions results   # chấm lại từ JSONL đã lưu
```

**Chấm một file dự đoán**:

```bash
python eval.py results/pred_test_voting_char.jsonl
python eval.py results/pred_test_baseline.jsonl --keep-space
python eval.py results/*.jsonl --errors
```

## Tóm tắt

| Mục | Nội dung | Vị trí|
|-----|----------|-------|
| 2.1 | Baseline PaddleOCR gốc trên `rec_test.txt` | `run_eval.py --baseline` → `results/comparison_test.csv` |
| 2.2 | Fine-tune, ra checkpoint chạy được | `notebooks/paddle-ocr-nlp-ck.ipynb` |
| 2.3 | Bảng so sánh duy nhất, cùng tập test | `results/comparison_test.csv` |
| 2.4 | Ablation đổi đúng 1 yếu tố: rộng ảnh 640 vs 960 | `notebooks/paddle-ocr-nlp-ck.ipynb` |
| 2.5 | 20 dòng sai, ba loại lỗi kèm tỉ lệ | `results/error_samples.csv`, `results/error_types.csv` |
| 4 | `pred_test.jsonl` cho cả model gốc và fine-tune | `results/pred_test_*.jsonl` |
| 4 | `README.md`, `requirements.txt`, `eval.py`, `results/` | |
| 5 | config tham số | `configs/default.yaml`, mục dưới |

## Cấu trúc

```
vietnamese-ocr/
├── eval.py                  chấm một file dự đoán JSONL
├── predict.py               nhận dạng ảnh bất kỳ bằng cả ba model + hợp phiếu
├── run_eval.py              đánh giá toàn tập test, sinh bảng và hình
├── configs/default.yaml     mọi đường dẫn và siêu tham số
├── viocr/
│   ├── config.py            bảng ký tự 252, cấu hình VietOCR, hằng số PaddleOCR
│   ├── data.py              đọc file nhãn
│   ├── metrics.py           acc / NED / CER / WER, phân loại ba lỗi
│   ├── voting.py            hợp phiếu mức dòng, mức ký tự (ROVER), từ điển
│   ├── report.py            bảng và biểu đồ
│   └── recognizers/         paddle_rec.py, vietocr_rec.py, trocr_rec.py
├── notebooks/               5 notebook Kaggle đã dùng để train và đo
└── results/                 đầu ra
```