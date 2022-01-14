defmodule YoloElixirOnnx.Preprocessing do

  import Nx.Defn

  def letterbox(img, [w, h] \\ [640, 640], color \\ [114, 114, 114], auto \\ True, scaleFill \\ False, scaleup \\ True) do
    {:ok, {height, width, _channels}}=OpenCV.Mat.shape(img)
    shape = [h: height, w: width]  # current shape [height, width]

    ## Scale ratio (new / old)
    r = min(h / shape[:h], w / shape[:w])
    r = if !scaleup do  # only scale down, do not scale up (for better test mAP)
      min(r, 1.0)
    else
      r
    end
    # Compute padding
    new_unpad = {trunc(round(shape[:h] * r)), trunc(round(shape[:w] * r))}
    {dw, dh}={w - elem(new_unpad,0), h - elem(new_unpad,1)}
    {dw, dh, new_unpad} = if scaleFill do  # stretch
      {0.0, 0.0, [ h: w, w: h ]}
    else
      {dw, dh} = (auto == true && {rem(dw, 64), rem(dh, 64)} || {dw, dh})
      {dw, dh, [ h: (round(shape[:w] * r)), w: (round(shape[:h] * r)) ]}
    end

    img = if Enum.reverse(shape) != new_unpad do  # resize
      IO.puts "resize"<>inspect([img, new_unpad])
      {:ok, img}=OpenCV.resize(img, [new_unpad[:h], new_unpad[:w]], interpolation: OpenCV.cv_inter_linear) 
      img
    else
      img
    end
    {top, bottom} = {trunc(round(dh - 0.1)), trunc(round(dh + 0.01))}
    {left, right} = {trunc(round(dw - 0.1)), trunc(round(dw + 0.01))}
    {:ok, img} = OpenCV.copymakeborder(img, top, bottom, left, right, OpenCV.cv_border_constant, value: color)
    top2 = bottom2 = left2 = right2 = 0
    {:ok, {img_shape_0, img_shape_1, _channels}}=OpenCV.Mat.shape(img)
    img = cond do
      img_shape_0 != h ->
        top2 = div(h - img_shape_0,2)
        bottom2 = top2
        {:ok, img}=OpenCV.copymakeborder(img, top2, bottom2, left2, right2, OpenCV.cv_border_constant, value: color)  # add border
        img
      img_shape_1 != w ->
        left2 = div(w - img_shape_1, 2)
        right2 = left2
        {:ok, img}=OpenCV.copymakeborder(img, top2, bottom2, left2, right2, OpenCV.cv_border_constant, value: color)  # add border
        img
      true -> img
      end
    img
  end

  def saveImage(img, imgpath \\ nil) do
    # TODO: Add checks on imgpath
    imgpath && CImg.save(img, imgpath) || raise ArgumentError, "empty imgpath"
  end

  def scale_bbox(x, y, height, width, class_id, confidence, im_h, im_w, resized_im_h \\ 640, resized_im_w \\ 640) do
    gain = min(resized_im_w / im_w, resized_im_h / im_h)  # gain  = old / new
    [ pad_0, pad_1] = [ (resized_im_w - im_w * gain) / 2, (resized_im_h - im_h * gain) / 2]  # wh padding
    x = trunc((x - pad_0)/gain)
    y = trunc((y - pad_1)/gain)

    w = trunc(width/gain)
    h = trunc(height/gain)

    xmin = max(0, trunc(x - w / 2))
    ymin = max(0, trunc(y - h / 2))
    xmax = min(im_w, trunc(xmin + w))
    ymax = min(im_h, trunc(ymin + h))
    
    %{ xmin: xmin, xmax: xmax, ymin: ymin, ymax: ymax, class_id: class_id, confidence: confidence}
  end

  def entry_index(side, coord, classes, location, entry) do
    side_power_2 = trunc(:math.pow(side, 2))
    n = Integer.floor_div(location, side_power_2)
    loc = rem(location , side_power_2)
    trunc(side_power_2 * (n * (coord + classes + 1) + entry) + loc)
  end

  defn create_predictions_tensor(blob) do
    (1.0 / (1.0+Nx.exp(-blob)))
  end

  @doc """
    blob must be an Nx tensor
    predictions must be an Nx tensor
  """
  def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold \\ 0.5, yolo_type \\ "yolov3-tiny") do
    # ------------------------------------------ Validating output parameters ------------------------------------------
    { _, out_blob_c, out_blob_h, out_blob_w } = blob.shape
    IO.puts "PRIMA: "<>inspect(blob)
    predictions = create_predictions_tensor(blob) #Nx.divide(1,blob|>Nx.negate|>Nx.exp|>Nx.add(1))
    IO.puts "DOPO: "<>inspect(predictions)
    if out_blob_w != out_blob_h do
     raise "Invalid size of output blob. It sould be in NCHW layout and height should be equal to width. Current height = #{out_blob_h}, current width = #{out_blob_w}" \
    end
    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    {orig_im_h, orig_im_w} = original_im_shape
    {resized_image_h, resized_image_w} = resized_image_shape
    # side_square = elem(params.side,1) * elem(params.side,0)

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    bbox_size = trunc(out_blob_c/params.num) #4+1+num_classes
    # IO.puts("bbox_size = " <> inspect(bbox_size))
    # IO.puts("predictions = " <> inspect(predictions))

    objects = for row <- 0..elem(params.side,0)-1, col <- 0..elem(params.side,1)-1, n <- 0..params.num-1 do
      # IO.puts "Coordinate predictions: 0,#{inspect(n*bbox_size..(n+1)*bbox_size)}, #{row}, #{col}, #{n}"
      bbox = predictions[[0,n*bbox_size..((n+1)*bbox_size)-1,row,col]]
      IO.puts "[ x, y, width, height, object_probability ]="<>inspect(Nx.to_flat_list(bbox[0..4]))
      [ x, y, width, height, object_probability ] = Nx.to_flat_list(bbox[0..4])
      { last_elem } = bbox.shape 
      class_probabilities = bbox[5..last_elem-1]
      if object_probability >= threshold do
        x = (2*x - 0.5 + col)*(resized_image_w/out_blob_w)
        y = (2*y - 0.5 + row)*(resized_image_h/out_blob_h)

        idx = cond do
          trunc(resized_image_w/out_blob_w) == 8 and trunc(resized_image_h/out_blob_h) == 8 -> 0     #80x80
          trunc(resized_image_w/out_blob_w) == 16 and trunc(resized_image_h/out_blob_h) == 16 -> 1   #40x40
          trunc(resized_image_w/out_blob_w) == 32 and trunc(resized_image_h/out_blob_h) == 32 -> 2   #20x20
          trunc(resized_image_w/out_blob_w) == 64 and trunc(resized_image_h/out_blob_h) == 64 -> 3   #20x20
          trunc(resized_image_w/out_blob_w) == 128 and trunc(resized_image_h/out_blob_h) == 128 -> 4 #20x20
          true -> raise("Wrong resized image size")
        end
        {width, height}=if yolo_type == "yolov4-p5" or yolo_type == "yolov4-p6" or yolo_type == "yolov4-p7" do
          # Controllare il formato di params
          { 
            :math.pow(2*width,2)* Enum.at(params.anchors,idx * 8 + 2 * n),
            :math.pow(2*height,2) * Enum.at(params.anchors,idx * 8 + 2 * n + 1)
          }
        else
          {
            :math.pow(2*width,2) * Enum.at(params.anchors,idx * 6 + 2 * n),
            :math.pow(2*height,2) * Enum.at(params.anchors,idx * 6 + 2 * n + 1)
          }
        end
        # class_id = Nx.argmax(Nx.dot(class_probabilities, object_probability))|>Nx.to_number
        class_id = 
          class_probabilities
          |> Nx.dot(object_probability)
          |> Nx.argmax
          |> Nx.to_number
        confidence = Nx.to_number(class_probabilities[class_id])*object_probability
        scale_bbox(x, y, height, width, class_id, confidence, orig_im_h, orig_im_w, resized_image_h, resized_image_w)
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
      (width_of_overlap_area < 0 or height_of_overlap_area < 0) 
        && 0 
        || (width_of_overlap_area * height_of_overlap_area)
    box_1_area = (box_1.ymax - box_1.ymin) * (box_1.xmax - box_1.xmin)
    box_2_area = (box_2.ymax - box_2.ymin) * (box_2.xmax - box_2.xmin)
    area_of_union = box_1_area + box_2_area - area_of_overlap
    # IO.puts "area_of_union=#{area_of_overlap}, #{area_of_union}"
    area_of_union == 0 && 0 || area_of_overlap / area_of_union
  end

  def yoloParams(side, yolo_type, param \\ []) do
    coords = Keyword.get(param, :coords, 4)
    classes = Keyword.get(param, :classes, 80)

    {num, anchors} = cond do
      yolo_type == "yolov4" ->
        {3,[12.0,16.0, 19.0,36.0, 40.0,28.0, 36.0,75.0, 76.0,55.0, 72.0,146.0, 142.0,110.0, 192.0,243.0, 459.0,401.0]}
      yolo_type == "yolov4-p5" ->
        {4, [13.0,17.0, 31.0,25.0, 24.0,51.0, 61.0,45.0, 48.0,102.0, 119.0,96.0, 97.0,189.0, 217.0,184.0, 171.0,384.0, 324.0,451.0, 616.0,618.0, 800.0,800.0]}
      yolo_type == "yolov4-p6" ->
        {4, [13.0,17.0, 31.0,25.0, 24.0,51.0, 61.0,45.0, 61.0,45.0, 48.0,102.0, 119.0,96.0, 97.0,189.0, 97.0,189.0, 217.0,184.0, 171.0,384.0, 324.0,451.0, 324.0,451.0, 545.0,357.0, 616.0,618.0, 1024.0,1024.0]}
      yolo_type == 'yolov4-p7' ->
        {5,[13.0,17.0,  22.0,25.0,  27.0,66.0,  55.0,41.0, 57.0,88.0,  112.0,69.0,  69.0,177.0,  136.0,138.0, 136.0,138.0,  287.0,114.0,  134.0,275.0,  268.0,248.0, 268.0,248.0,  232.0,504.0,  445.0,416.0,  640.0,640.0, 812.0,393.0,  477.0,808.0,  1070.0,908.0,  1408.0,1408.0]}
      true ->
        {3,[10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0]}
    end
    %{coords: coords, classes: classes, side: side, num: num, anchors: anchors}
  end
end
