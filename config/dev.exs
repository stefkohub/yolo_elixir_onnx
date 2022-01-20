import Config

config :yolo_elixir_onnx,
  model_directory: "/home/stefano/ElixirML/YoloWeights",
  classes_filename: "/home/stefano/ElixirML/YoloWeights/coco.names",
  supported_models: ["yolov3-tiny", "yolov4-tiny", "yolov3"],
  iou_threshold: 0.4,
  prob_threshold: 0.5
