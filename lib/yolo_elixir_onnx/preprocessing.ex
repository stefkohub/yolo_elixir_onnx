defmodule YoloElixirOnnx.Preprocessing do
  def letterbox(
        img,
        [w, h] \\ [640, 640],
        color \\ [114, 114, 114],
        auto \\ True,
        scaleFill \\ False,
        scaleup \\ True
      ) do
    {:ok, {height, width, _channels}} = OpenCV.Mat.shape(img)
    # current shape [height, width]
    shape = [h: height, w: width]

    ## Scale ratio (new / old)
    r = min(h / shape[:h], w / shape[:w])
    # only scale down, do not scale up (for better test mAP)
    r =
      if !scaleup do
        min(r, 1.0)
      else
        r
      end

    # Compute padding
    new_unpad = {trunc(round(shape[:h] * r)), trunc(round(shape[:w] * r))}
    {dw, dh} = {w - elem(new_unpad, 0), h - elem(new_unpad, 1)}
    # stretch
    {dw, dh, new_unpad} =
      if scaleFill do
        {0.0, 0.0, [h: w, w: h]}
      else
        {dw, dh} = (auto == true && {rem(dw, 64), rem(dh, 64)}) || {dw, dh}
        {dw, dh, [h: round(shape[:w] * r), w: round(shape[:h] * r)]}
      end

    # resize
    img =
      if Enum.reverse(shape) != new_unpad do
        {:ok, img} =
          OpenCV.resize(img, [new_unpad[:h], new_unpad[:w]],
            interpolation: OpenCV.cv_INTER_LINEAR()
          )

        img
      else
        img
      end

    {top, bottom} = {trunc(round(dh - 0.1)), trunc(round(dh + 0.01))}
    {left, right} = {trunc(round(dw - 0.1)), trunc(round(dw + 0.01))}

    {:ok, img} =
      OpenCV.copyMakeBorder(img, top, bottom, left, right, OpenCV.cv_BORDER_CONSTANT(),
        value: color
      )

    top2 = bottom2 = left2 = right2 = 0
    {:ok, {img_shape_0, img_shape_1, _channels}} = OpenCV.Mat.shape(img)

    img =
      cond do
        img_shape_0 != h ->
          top2 = div(h - img_shape_0, 2)
          bottom2 = top2
          # add border
          {:ok, img} =
            OpenCV.copyMakeBorder(img, top2, bottom2, left2, right2, OpenCV.cv_BORDER_CONSTANT(),
              value: color
            )

          img

        img_shape_1 != w ->
          left2 = div(w - img_shape_1, 2)
          right2 = left2
          # add border
          {:ok, img} =
            OpenCV.copyMakeBorder(img, top2, bottom2, left2, right2, OpenCV.cv_BORDER_CONSTANT(),
              value: color
            )

          img

        true ->
          img
      end

    img
  end

  def entry_index(side, coord, classes, location, entry) do
    side_power_2 = trunc(:math.pow(side, 2))
    n = Integer.floor_div(location, side_power_2)
    loc = rem(location, side_power_2)
    trunc(side_power_2 * (n * (coord + classes + 1) + entry) + loc)
  end

  def yoloParams(side, yolo_type, param \\ []) do
    coords = Keyword.get(param, :coords, 4)
    classes = Keyword.get(param, :classes, 80)

    {num, anchors} =
      cond do
        yolo_type == "yolov4" ->
          {3,
           {12.0, 16.0, 19.0, 36.0, 40.0, 28.0, 36.0, 75.0, 76.0, 55.0, 72.0, 146.0, 142.0, 110.0,
            192.0, 243.0, 459.0, 401.0}}

        yolo_type == "yolov4-p5" ->
          {4,
           {13.0, 17.0, 31.0, 25.0, 24.0, 51.0, 61.0, 45.0, 48.0, 102.0, 119.0, 96.0, 97.0, 189.0,
            217.0, 184.0, 171.0, 384.0, 324.0, 451.0, 616.0, 618.0, 800.0, 800.0}}

        yolo_type == "yolov4-p6" ->
          {4,
           {13.0, 17.0, 31.0, 25.0, 24.0, 51.0, 61.0, 45.0, 61.0, 45.0, 48.0, 102.0, 119.0, 96.0,
            97.0, 189.0, 97.0, 189.0, 217.0, 184.0, 171.0, 384.0, 324.0, 451.0, 324.0, 451.0,
            545.0, 357.0, 616.0, 618.0, 1024.0, 1024.0}}

        yolo_type == 'yolov4-p7' ->
          {5,
           {13.0, 17.0, 22.0, 25.0, 27.0, 66.0, 55.0, 41.0, 57.0, 88.0, 112.0, 69.0, 69.0, 177.0,
            136.0, 138.0, 136.0, 138.0, 287.0, 114.0, 134.0, 275.0, 268.0, 248.0, 268.0, 248.0,
            232.0, 504.0, 445.0, 416.0, 640.0, 640.0, 812.0, 393.0, 477.0, 808.0, 1070.0, 908.0,
            1408.0, 1408.0}}

        true ->
          {3,
           {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0,
            156.0, 198.0, 373.0, 326.0}}
      end

    %{coords: coords, classes: classes, side: side, num: num, anchors: anchors}
  end
end
