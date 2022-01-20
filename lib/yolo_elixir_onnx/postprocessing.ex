defmodule YoloElixirOnnx.Postprocessing do
  import Nx.Defn

  Nx.Defn.default_options(compiler: EXLA)
  EXLA.set_preferred_defn_options([:tpu, :cuda, :rocm, :host])

  def saveImage(img, imgpath \\ nil) do
    # TODO: Add checks on imgpath
    (imgpath && CImg.save(img, imgpath)) || raise ArgumentError, "empty imgpath"
  end

  def scale_bbox(
        x,
        y,
        height,
        width,
        class_id,
        confidence,
        im_h,
        im_w,
        resized_im_h \\ 640,
        resized_im_w \\ 640
      ) do
    # gain  = old / new
    gain = min(resized_im_w / im_w, resized_im_h / im_h)
    # wh padding
    [pad_0, pad_1] = [(resized_im_w - im_w * gain) / 2, (resized_im_h - im_h * gain) / 2]
    x = trunc((x - pad_0) / gain)
    y = trunc((y - pad_1) / gain)

    w = trunc(width / gain)
    h = trunc(height / gain)

    xmin = max(0, trunc(x - w / 2))
    ymin = max(0, trunc(y - h / 2))
    xmax = min(im_w, trunc(xmin + w))
    ymax = min(im_h, trunc(ymin + h))

    %{
      xmin: xmin,
      xmax: xmax,
      ymin: ymin,
      ymax: ymax,
      class_id: Nx.to_number(class_id),
      confidence: Nx.to_number(confidence)
    }
  end

  defn create_predictions_tensor(blob) do
    1.0 / (1.0 + Nx.exp(-blob))
  end

  defn class_id_confidence(class_probabilities, object_probability) do
    class_id =
      class_probabilities
      |> Nx.dot(object_probability)
      |> Nx.argmax()

    {class_id, Nx.dot(class_probabilities[class_id], object_probability)}
  end

  @doc """
    blob must be an Nx tensor
    predictions must be an Nx tensor
  """
  def parse_yolo_region(
        blob,
        resized_image_shape,
        original_im_shape,
        params,
        threshold \\ 0.5,
        yolo_type \\ "yolov3-tiny"
      ) do
    # ------------------------------------------ Validating output parameters ------------------------------------------
    {_, out_blob_c, out_blob_h, out_blob_w} = blob.shape
    predictions = create_predictions_tensor(blob)
    # IO.puts "DOPO: "<>inspect(predictions)
    if out_blob_w != out_blob_h do
      raise "Invalid size of output blob. It sould be in NCHW layout and height should be equal to width. Current height = #{out_blob_h}, current width = #{out_blob_w}"
    end

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    {orig_im_h, orig_im_w} = original_im_shape
    {resized_image_h, resized_image_w} = resized_image_shape

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    # 4+1+num_classes
    bbox_size = trunc(out_blob_c / params.num)

    objects =
      for row <- 0..(elem(params.side, 0) - 1),
          col <- 0..(elem(params.side, 1) - 1),
          n <- 0..(params.num - 1) do
        bbox = predictions[[0, (n * bbox_size)..((n + 1) * bbox_size - 1), row, col]]
        [x, y, width, height, object_probability] = Nx.to_flat_list(bbox[0..4])
        {last_elem} = bbox.shape
        class_probabilities = bbox[5..(last_elem - 1)]

        if object_probability >= threshold do
          x = (2 * x - 0.5 + col) * (resized_image_w / out_blob_w)
          y = (2 * y - 0.5 + row) * (resized_image_h / out_blob_h)

          idx =
            cond do
              # 80x80
              trunc(resized_image_w / out_blob_w) == 8 and
                  trunc(resized_image_h / out_blob_h) == 8 ->
                0

              # 40x40
              trunc(resized_image_w / out_blob_w) == 16 and
                  trunc(resized_image_h / out_blob_h) == 16 ->
                1

              # 20x20
              trunc(resized_image_w / out_blob_w) == 32 and
                  trunc(resized_image_h / out_blob_h) == 32 ->
                2

              # 20x20
              trunc(resized_image_w / out_blob_w) == 64 and
                  trunc(resized_image_h / out_blob_h) == 64 ->
                3

              # 20x20
              trunc(resized_image_w / out_blob_w) == 128 and
                  trunc(resized_image_h / out_blob_h) == 128 ->
                4

              true ->
                raise("Wrong resized image size")
            end

          {width, height} =
            if yolo_type == "yolov4-p5" or yolo_type == "yolov4-p6" or yolo_type == "yolov4-p7" do
              {
                :math.pow(2 * width, 2) * elem(params.anchors, idx * 8 + 2 * n),
                :math.pow(2 * height, 2) * elem(params.anchors, idx * 8 + 2 * n + 1)
              }
            else
              {
                :math.pow(2 * width, 2) * elem(params.anchors, idx * 6 + 2 * n),
                :math.pow(2 * height, 2) * elem(params.anchors, idx * 6 + 2 * n + 1)
              }
            end

          {class_id, confidence} = class_id_confidence(class_probabilities, object_probability)

          scale_bbox(
            x,
            y,
            height,
            width,
            class_id,
            confidence,
            orig_im_h,
            orig_im_w,
            resized_image_h,
            resized_image_w
          )
        else
          nil
        end
      end

    Enum.filter(objects, fn x -> x != nil end)
  end

  def intersection_over_union(box_1, box_2) do
    width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin)
    height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin)

    area_of_overlap =
      ((width_of_overlap_area < 0 or height_of_overlap_area < 0) &&
         0) ||
        width_of_overlap_area * height_of_overlap_area

    box_1_area = (box_1.ymax - box_1.ymin) * (box_1.xmax - box_1.xmin)
    box_2_area = (box_2.ymax - box_2.ymin) * (box_2.xmax - box_2.xmin)
    area_of_union = box_1_area + box_2_area - area_of_overlap
    (area_of_union == 0 && 0) || area_of_overlap / area_of_union
  end
end
