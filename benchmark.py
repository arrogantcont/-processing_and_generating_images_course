from ultralytics import YOLO

# model = YOLO("processing_and_generating_images_course/yolov8_default.yaml")


# model.train(
#     data="/media/yura/edbf3aee-91a2-4ee8-98ca-fb2ac8d9fc21/processing_gan/processing_and_generating_images_course/CSGO-TRAIN-YOLO-V5-5/data.yaml",
#     batch=64,
#     name="yolov8n_default_params",
#     workers=8,
#     exist_ok=True,
# )

from ultralytics.utils.benchmarks import benchmark

# Benchmark on GPU
benchmark(
    model="/media/yura/edbf3aee-91a2-4ee8-98ca-fb2ac8d9fc21/processing_gan/processing_and_generating_images_course/runs/detect/yolov8n_custom_params/weights/last.pt",
    data="/media/yura/edbf3aee-91a2-4ee8-98ca-fb2ac8d9fc21/processing_gan/processing_and_generating_images_course/CSGO-TRAIN-YOLO-V5-5/data.yaml",
    imgsz=640,
    half=False,
    device=0,
)

benchmark(
    model="/media/yura/edbf3aee-91a2-4ee8-98ca-fb2ac8d9fc21/processing_gan/processing_and_generating_images_course/runs/detect/runs/detect/yolov8n_default_params/weights/last.pt",
    data="/media/yura/edbf3aee-91a2-4ee8-98ca-fb2ac8d9fc21/processing_gan/processing_and_generating_images_course/CSGO-TRAIN-YOLO-V5-5/data.yaml",
    imgsz=640,
    half=False,
    device=0,
)
