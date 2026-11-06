# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** GPU (CUDA)

**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   - Record này nói về **cả bức ảnh** (image-level classification). Tác vụ Image Classification gán nhãn tổng thể ngữ cảnh/chủ đề chính cho toàn bộ khung hình, đầu ra không chứa thông tin tọa độ vị trí (bounding box) hay phân tách riêng lẻ từng đối tượng trong ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   - Danh sách tên lớp đi kèm taxonomy cố định được định nghĩa trước khi huấn luyện checkpoint (ở đây là bộ taxonomy ImageNet-1K gồm 1.000 lớp đi cùng checkpoint `yolo11n-cls.pt`), mô hình không tự nghĩ ra tên lớp.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   - Vì mỗi taxonomy có quy ước và không gian nhãn riêng. Cùng một `class_name` hoặc cùng một `class_id` có thể mang ý nghĩa khác nhau giữa các bộ dữ liệu (ví dụ COCO-80 so với ImageNet-1K). Việc lưu đủ cả `taxonomy_name`, `class_id` và `class_name` đảm bảo tính toàn vẹn dữ liệu (provenance), tránh nhầm lẫn khi tích hợp và tái lập pipeline.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   - Guideline cần quy định rõ tiêu chí ưu tiên đối tượng chính (dominant subject rule): xác định theo diện tích chiếm phần lớn khung hình, vị trí trung tâm, tiêu cự lấy nét (in-focus), hoặc bảng phân cấp ưu tiên chủ đề/hành vi rõ ràng khi có nhiều đối tượng cùng xuất hiện.
6. Vì sao không được lấy `score` của model làm ground truth?
   - `score` chỉ là xác suất suy luận (độ tin cậy toán học) của mô hình dựa trên trọng số đã học, có thể bị ảo giác (hallucination), bias hoặc dự đoán sai. Ground truth phải là sự thật chuẩn xác do con người kiểm chứng và gắn nhãn theo bộ quy tắc thống nhất (guideline).

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   - Hộp bao quanh đối tượng người (`person`) có tọa độ góc trên-trái tại điểm `(x_min=385.33 px, y_min=69.24 px)` tính từ gốc tọa độ `(0, 0)` ở góc trên-cùng bên trái bức ảnh, và góc dưới-phải tại điểm `(x_max=498.92 px, y_max=348.92 px)`. Chiều rộng của hộp là 113.58 pixel và chiều cao là 279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   - Tại ngưỡng `threshold = 0.20`: có 17 vật thể được phát hiện (`person`, `bowl`, `bowl`, `oven`, `oven`, `person`, `bowl`, `bowl`, `cup`, `cup`, `bowl`, `spoon`, `potted plant`, `spoon`, `dining table`, `spoon`, `bottle`).
   - Tại ngưỡng `threshold = 0.60`: có 6 vật thể được phát hiện (`person`, `bowl`, `bowl`, `oven`, `oven`, `person`).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   - Reviewer phải tốn nhiều thời gian và công sức hơn để kiểm tra, lọc và xóa các dự đoán dương tính giả (false positives), các hộp bao trùng lặp (duplicate boxes), hoặc các hộp nhiễu có độ tin cậy thấp.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   - Quy tắc bao khít (tight bounding box): Hộp chữ nhật phải bao bọc sát toàn bộ phần nhìn thấy được của vật thể (không chừa quá 3-5 pixel biên trống), và không được cố tình cắt phạm vào thân vật thể.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   - Ví dụ chiếc thìa (`spoon`) hoặc bát (`bowl`) bị che khuất một phần bởi tay người/bàn bếp, hoặc người đứng ở mép trái (`person` ở tọa độ `x_min=0.12` bị cắt mép ảnh): Trường hợp này nên viết rõ thành quy tắc trong guideline (nêu rõ tỷ lệ nhìn thấy tối thiểu, ví dụ thấy trên 20-30% thì bao phần nhìn thấy được, gắn cờ thuộc tính `occluded=True` hoặc `truncated=True`). Chỉ chuyển lên hỏi Lab Coach (escalation) khi gặp vật thể quá mơ hồ, không thể nhận diện được hình dạng hoặc mâu thuẫn trực tiếp với phân loại của guideline.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   - Polygon cung cấp đường biên chính xác đến từng điểm ảnh theo hình dáng thực tế của đối tượng (đường cong cơ thể, viền cong của bát đĩa, cánh tay), loại bỏ hoàn toàn phần diện tích phông nền bị thừa bên trong góc của bounding box hình chữ nhật.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   - `instance_id` là mã định danh duy nhất dùng để phân biệt các cá thể/vật thể riêng biệt trong cùng một bức ảnh (dù chúng có chung lớp). Nó không phải `class_id` vì các vật thể cùng lớp vẫn có `instance_id` khác nhau; và không phải `tracking_id` vì không liên kết theo dõi đối tượng xuyên suốt qua chuỗi khung hình video thời gian.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   - Đường viền polygon phải bám sát mép ngoài nhìn thấy của đối tượng với sai số không vượt quá 2 pixel; nếu đối tượng có lỗ rỗng xuyên thấu hoặc khoảng trống nhìn xuyên qua nền thì phải tạo polygon trừ (donut hole / negative polygon).
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   - Vị trí chiếc bát chạm nhau trên mặt bếp hoặc tay người cầm dụng cụ bị che khuất một phần: Trường hợp này dùng guideline bằng cách quy định ranh giới tiếp xúc (shared boundary: không vẽ đè mask lên nhau, gán ranh giới rõ ràng cho vật ở lớp trên). Nếu vùng tiếp xúc quá tối, nhòe chuyển động hoặc không thể phân biệt nổi ranh giới vật thể bằng mắt thường thì mới chuyển lên escalation để xin ý kiến quyết định chuẩn từ Coach.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn duy nhất cho toàn bộ ảnh: `class_id`, `class_name`, `taxonomy_name` | Ảnh `traffic` có cả xe buýt, ô tô, người đi bộ nhưng model lại đoán top-1 là `cab` (0.51) do taxonomy ImageNet-1K không có lớp bao quát tổng thể giao thông. | Đọc ảnh, đối chiếu quy tắc chọn chủ thể chính trong guideline để gán đúng 1 nhãn chuẩn. | Kiểm tra nhãn có bám sát taxonomy và tiêu chí chủ thể chính của guideline không, trả về rework nếu sai quy tắc. |
| Phát hiện vật thể | Danh sách các hộp tọa độ: `[class_id, x_min, y_min, x_max, y_max]` cho từng vật thể | Ở ảnh `kitchen`, tại threshold 0.20 xuất hiện tới 17 vật thể, nhiều vật bị nhận diện nhầm hoặc nhiễu (`spoon`, `bottle`), hai `oven` bị phân mảnh. | Vẽ bounding box ôm sát các vật thể nhìn rõ theo định nghĩa trong taxonomy COCO-80. | Kiểm tra độ khít của box, đảm bảo không bỏ sót đối tượng (FN) và không gán nhầm vật thể nhiễu (FP). |
| Instance segmentation | Danh sách các polygon điểm tọa độ viền và `instance_id` riêng cho từng cá thể | Tại ảnh `kitchen`, các vật dụng đặt sát nhau hoặc bóng đổ trên mặt bàn khiến viền mask của model dễ lẹm vào nền hoặc dính hai vật thể. | Vẽ các điểm đa giác polygon bao bọc tỉ mỉ từng cá thể riêng biệt, tách biệt các vật thể tiếp xúc nhau. | Soi chi tiết độ chính xác đường biên mask ở mức pixel, kiểm tra tính duy nhất của `instance_id` và tính toàn vẹn đa giác. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   - Tuân thủ phạm vi sử dụng ảnh mẫu công khai được cấp phép, tuyệt đối không tải lên dữ liệu cá nhân (họ tên, MSSV, thông tin liên lạc) hay dữ liệu nhạy cảm nội bộ vào mã nguồn, artifact và báo cáo công khai.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   - Dừng ngay thao tác xử lý dữ liệu và báo cáo ngay lập tức cho Lab Coach / Giảng viên hướng dẫn để được xử lý và cô lập dữ liệu an toàn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS

