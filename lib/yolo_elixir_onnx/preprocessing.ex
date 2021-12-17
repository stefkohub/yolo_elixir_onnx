defmodule YoloElixirOnnx.Preprocessing do

  def letterbox(img, [w, h] \\ [640, 640], color \\ [114, 114, 114], auto \\ True, scaleFill \\ False, scaleup \\ True) do
    shape = [h: img.height, w: img.width]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(h / shape[:h], w / shape[:w])
    IO.puts "shape, r, scaleup="<>inspect([shape, r, scaleup])
    r = if !scaleup do  # only scale down, do not scale up (for better test mAP)
      r = min(r, 1.0)
    else
      r
    end
    # Compute padding
    new_unpad = if scaleFill do  # stretch
      [ h: w, w: h ]
    else
      [ h: (round(shape[:w] * r)), w: (round(shape[:h] * r)) ]
    end

    img = if Enum.reverse(shape) != new_unpad do  # resize
      resizeImg(img, new_unpad)
    else
      img
    end
    copyMakeBorder(img, new_unpad, color) # add border
    img
  end

  def resizeImg(img, new_unpad, interpolation \\ :INTER_LINEAR) do
    IO.puts "Ridimensiono immagine a: "<>inspect(new_unpad)
    img |> Mogrify.resize_to_limit(Integer.to_string(new_unpad[:h])<>"x"<>Integer.to_string(new_unpad[:w]))
  end

  def copyMakeBorder(img, new_unpad, color) do
    hexColor = "#" <> (for c <- color do Integer.to_string(c, 16) end|>Enum.join)
    img|>
    Mogrify.custom("background", hexColor)|>
    Mogrify.gravity("center")|>
    Mogrify.extent(Integer.to_string(new_unpad[:h])<>"x"<>Integer.to_string(new_unpad[:w]))
  end

  def openImage(imgpath) do
    # TODO: Add sanity checks...
    Mogrify.open(imgpath)|>Mogrify.verbose
  end
  
  def identifyImage(imgpath) do
    info = Mogrify.identify(imgpath)
    IO.inspect(info)
    info
  end

  def saveImage(img, imgpath \\ :in_place) do
    # TODO: Add checks on imgpath
    IO.puts "Dovrei salvare in: "<>imgpath
    img|>Mogrify.save(path: imgpath)
  end

  def scale_bbox(x, y, height, width, class_id, confidence, im_h, im_w, resized_im_h \\ 640, resized_im_w \\ 640) do
    gain = min(resized_im_w / im_w, resized_im_h / im_h)  # gain  = old / new
    pad = [ (resized_im_w - im_w * gain) / 2, (resized_im_h - im_h * gain) / 2]  # wh padding
    x = trunc((x - pad[0])/gain)
    y = trunc((y - pad[1])/gain)

    w = trunc(width/gain)
    h = trunc(height/gain)

    xmin = max(0, trunc(x - w / 2))
    ymin = max(0, trunc(y - h / 2))
    xmax = min(im_w, trunc(xmin + w))
    ymax = min(im_h, trunc(ymin + h))
    
    %{ xmin: xmin, xmax: xmax, ymin: ymin, ymax: ymax, class_id: class_id, confidence: confidence}
  end

  def entry_index(side, coord, classes, location, entry) do
    side_power_2 = :math.pow(side, 2)
    n = Integer.floor_div(location, side_power_2)
    loc = rem(location , side_power_2)
    side_power_2 * (n * (coord + classes + 1) + entry) + loc
  end

  @doc """
    blob must be an Nx tensor
    predictions must be an Nx tensor
  """
  def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold, yolo_type) do
    # ------------------------------------------ Validating output parameters ------------------------------------------
    [ out_blob_n, out_blob_c, out_blob_h, out_blob_w ] = blob.shape
    predictions = Nx.divide(1,blob|>Nx.negate|>Nx.exp|>Nx.add(1))
    if out_blob_w != out_blob_h do
     raise "Invalid size of output blob. It sould be in NCHW layout and height should be equal to width. Current height = [#{out_blob_h}, current width = #{out_blob_w}" \
    end
    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    [orig_im_h, orig_im_w] = original_im_shape
    [resized_image_h, resized_image_w] = resized_image_shape
    # TODO: Devo vedere questo params come è fatto
    side_square = Enum.at(params.side,1) * Enum.at(params.side,0)

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    bbox_size = trunc(out_blob_c/params.num) #4+1+num_classes
    IO.puts("bbox_size = " <> inspect(bbox_size))
    IO.puts("bbox_size = " <> inspect(bbox_size))

    objects = for row <- Enum.at(params.side,0), col <- Enum.at(params.side,1), n <- params.num do
      bbox = predictions[0][n*bbox_size..(n+1)*bbox_size][row][col]
      [ x, y, width, height, object_probability ] = Nx.to_flat_list(bbox[0..4])
      { last_elem } = bbox.shape 
      last_elem = last_elem - 1
      class_probabilities = bbox[5..last_elem]
      if object_probability >= threshold do
        IO.puts("resized_image_w = " <>inspect(resized_image_w))
        IO.puts("out_blob_w = " <>inspect(out_blob_w))
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
        if yolo_type == "yolov4-p5" or yolo_type == "yolov4-p6" or yolo_type == "yolov4-p7" do
          # Controllare il formato di params
          width = :math.pow(2*width,2)* params.anchors[idx * 8 + 2 * n]
          height = :math.pow(2*height,2) * params.anchors[idx * 8 + 2 * n + 1]
        else
          width = :math.pow(2*width,2) * params.anchors[idx * 6 + 2 * n]
          height = :math.pow(2*height,2) * params.anchors[idx * 6 + 2 * n + 1]
        end
        class_id = Nx.argmax(Nx.dot(class_probabilities, object_probability))
        confidence = Nx.dot(class_probabilities[class_id], object_probability)
        scale_bbox(x, y, height, width, class_id, confidence, orig_im_h, orig_im_w, resized_image_h, resized_image_w)
      end
    end
  end

  def intersection_over_union(box_1, box_2) do
    width_of_overlap_area = min(box_1["xmax"], box_2["xmax"]) - max(box_1["xmin"], box_2["xmin"])
    height_of_overlap_area = min(box_1["ymax"], box_2["ymax"]) - max(box_1["ymin"], box_2["ymin"])
    area_of_overlap = width_of_overlap_area < 0 or height_of_overlap_area < 0 && 0 || width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1["ymax"] - box_1["ymin"]) * (box_1["xmax"] - box_1["xmin"])
    box_2_area = (box_2["ymax"] - box_2["ymin"]) * (box_2["xmax"] - box_2["xmin"])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    area_of_union == 0 && 0 || area_of_overlap / area_of_union
  end


  def preprocess(img) do
    boxed_img = letterbox(img, Enum.reverse([416, 416]))

  end
end
