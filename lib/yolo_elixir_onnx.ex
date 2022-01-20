defmodule YoloElixirOnnx do
  @moduledoc """
  Documentation for `YoloElixirOnnx`.
  """

  require Axon
  require Logger

  alias YoloElixirOnnx.Preprocessing
  alias YoloElixirOnnx.Postprocessing

  EXLA.set_preferred_defn_options([:tpu, :cuda, :rocm, :host])

  @doc """

  """
  def main(imgPath \\ nil, in_stream \\ nil, model_type \\ "yolov3-tiny") do
    # parsing parameters
    config = Map.new(Application.get_all_env(:yolo_elixir_onnx))

    unless model_type in config.supported_models do
      raise ArgumentError,
            "Unsupported model #{model_type}. Supported models are: #{config.supported_models}"
    end

    {out_stream, out_filename, frame, image_height, image_width} =
      cond do
        imgPath != nil ->
          unless File.exists?(imgPath) do
            raise ArgumentError, "Cannot find image file #{imgPath}"
          end

          IO.puts("Opening image file...")
          {:ok, image} = OpenCV.imread(imgPath)
          {:ok, {image_height, image_width, _image_channels}} = OpenCV.Mat.shape(image)
          out_filename = "#{Path.rootname(imgPath)}-#{model_type}-#{Time.utc_now()}.jpeg"
          {nil, out_filename, image, image_height, image_width}

        in_stream != nil ->
          unless File.exists?(in_stream) do
            raise ArgumentError, "Cannot find video file #{in_stream}"
          end

          IO.puts("Opening video file...")
          input_stream = (in_stream == "cam" && 0) || in_stream
          {:ok, cap} = OpenCV.VideoCapture.videoCapture(in_stream)
          {:ok, in_fps} = OpenCV.VideoCapture.get(cap, OpenCV.cv_CAP_PROP_FPS())
          {:ok, in_fourcc} = OpenCV.VideoCapture.get(cap, OpenCV.cv_CAP_PROP_FOURCC())

          {:ok, number_input_frames} =
            OpenCV.VideoCapture.get(cap, OpenCV.cv_CAP_PROP_FRAME_COUNT())

          number_input_frames =
            (number_input_frames != -1 and number_input_frames < 0 && 1) ||
              trunc(number_input_frames)

          {image_height, image_width} =
            if number_input_frames != 1 do
              # {:ok, res } = OpenCV.VideoCapture.read(cap)
              {:ok, image_height} =
                OpenCV.VideoCapture.get(cap, OpenCV.cv_CAP_PROP_FRAME_HEIGHT())

              {:ok, image_width} = OpenCV.VideoCapture.get(cap, OpenCV.cv_CAP_PROP_FRAME_WIDTH())
              {trunc(image_height), trunc(image_width)}
            else
              raise "Error in reading the video file"
            end

          in_stream_extension = Path.extname(in_stream)

          out_filename =
            "#{Path.rootname(in_stream)}-#{model_type}-#{Time.utc_now()}#{in_stream_extension}"

          {:ok, out_stream} =
            OpenCV.VideoWriter.videoWriter(out_filename, trunc(in_fourcc), in_fps, [
              image_width,
              image_height
            ])

          video_stream_map = %{
            :input_stream => input_stream,
            :cap => cap,
            :in_fps => in_fps,
            :in_fourcc => trunc(in_fourcc),
            :out_stream => out_stream
          }

          {video_stream_map, out_filename, nil, image_height, image_width}
      end

    {h, w, onnx_model_filename} =
      case model_type do
        "yolov3-tiny" -> {416, 416, "#{config.model_directory}/yolov3-tiny-416.onnx"}
        "yolov4-tiny" -> {416, 416, "#{config.model_directory}/yolov4-tiny-416.onnx"}
        "yolov3" -> {608, 608, "#{config.model_directory}/yolov3-608.onnx"}
      end

    start_time = Time.utc_now()
    IO.puts("Loading #{model_type} model...")
    {model_tuple, model_params} = AxonOnnx.Deserialize.__import__(onnx_model_filename)
    loading_time = Time.diff(Time.utc_now(), start_time, :millisecond) / 1000
    IO.puts("Time to load: #{loading_time}s")

    IO.puts("Loading class names...")

    {:ok, labels_map} =
      File.open(config.classes_filename, [:read], fn file ->
        names = IO.read(file, :all) |> String.split("\n") |> Enum.with_index()
        for {n, i} <- names, n != "", into: %{}, do: {i, n}
      end)

    # Generate pseudo-random colors for boxes
    colors = Nx.random_uniform({Enum.count(labels_map), 3}, 0, 255)

    IO.puts("Preparing inputs...")

    yolo_layer_params =
      for node <- Tuple.to_list(model_tuple), into: %{} do
        layer_name = node.name
        shape = Tuple.delete_at(node.parent.output_shape, 0)
        shape = Tuple.delete_at(shape, 0)
        yolo_params = Preprocessing.yoloParams(shape, model_type, node.opts)
        {layer_name, [shape, yolo_params]}
      end

    output_layer_names = Map.keys(yolo_layer_params)

    IO.puts("Entering input loop...")

    input_loop(
      out_stream,
      out_filename,
      imgPath,
      h,
      w,
      frame,
      image_height,
      image_width,
      model_tuple,
      model_params,
      labels_map,
      colors,
      yolo_layer_params,
      output_layer_names,
      config,
      0,
      1
    )

    IO.puts("Done.")
  end

  def input_loop(
        video_stream_map,
        out_filename,
        imgPath,
        h,
        w,
        frame,
        image_height,
        image_width,
        model_tuple,
        model_params,
        labels_map,
        colors,
        yolo_layer_params,
        output_layer_names,
        config,
        cur_request_id,
        request_id
      ) do
    if request_id != cur_request_id do
      # se è un'immagine importo request_id=cur_request_id
      # se è un video request_id cambia per ogni frame nuovo
      {frame, request_id} =
        cond do
          imgPath != nil ->
            IO.puts("Preprocessing input image...")
            {frame, cur_request_id}

          video_stream_map != nil ->
            IO.puts(".")

            frame =
              case OpenCV.VideoCapture.read(video_stream_map.cap) do
                {:ok, frame} -> frame
                :error -> nil
              end

            request_id =
              (OpenCV.VideoCapture.isOpened(video_stream_map.cap) != :ok && cur_request_id) ||
                request_id

            {frame, request_id}
        end

      if frame != nil do
        {:ok, blob} =
          OpenCV.dnn_blobFromImage(
            frame,
            scalefactor: 1 / 255.0,
            swapRB: true,
            mean: [0, 0, 0],
            size: [w, h]
          )

        in_frame = OpenCV.Nx.to_nx(blob)
        IO.puts("Starting inference...")
        start_time = Time.utc_now()
        output = Axon.predict(model_tuple, model_params, in_frame, compiler: EXLA)
        parsing_time = Time.diff(Time.utc_now(), start_time, :millisecond) / 1000
        IO.puts("Time to output: #{parsing_time}s")

        start_time = Time.utc_now()

        objects =
          for id <- 0..(Enum.count(output_layer_names) - 1) do
            layer_name = Enum.at(output_layer_names, id)
            out_blob = elem(output, id)
            [_shape, yolo_params] = yolo_layer_params[layer_name]
            {_, _, h, w} = in_frame.shape

            Postprocessing.parse_yolo_region(
              out_blob,
              {h, w},
              {image_height, image_width},
              yolo_params
            )
          end

        objects =
          objects
          |> Enum.concat()
          |> Enum.filter(fn x -> x != nil || x != [] end)
          |> Enum.sort_by(& &1.confidence, :desc)
          |> Enum.group_by(& &1.class_id)

        objects =
          for {_class, output} <- objects do
            class_len = Enum.count(output)
            outp = Enum.with_index(output)

            to_update =
              for {a, i} <- outp, {b, j} <- Enum.slice(outp, i + 1, class_len) do
                test_val = Postprocessing.intersection_over_union(a, b)
                (test_val > config.iou_threshold && [j, j, 0.0]) || [-1, -1, -1]
              end
              |> Enum.filter(fn [_, _, e] -> e == 0.0 end)
              |> Enum.uniq()

            new_outp =
              Enum.map(outp, fn {e, i} ->
                confidence =
                  Enum.find_value(to_update, e.confidence, fn x -> x == [i, i, 0.0] && 0.0 end)

                %{e | confidence: confidence}
              end)

            Enum.filter(new_outp, fn o ->
              o.confidence >= config.prob_threshold
            end)
          end
          |> List.flatten()

        parsing_time = Time.diff(Time.utc_now(), start_time, :millisecond) / 1000
        IO.puts("Time to classify: #{parsing_time}s")
        # and args.raw_output_message:
        if Enum.count(objects) > 0 do
          IO.puts("\nDetected boxes for batch 1")
          IO.puts(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")
        end

        output_img =
          Enum.reduce(objects, frame, fn obj, out_img ->
            if obj.xmax > image_width or obj.ymax > image_height or obj.xmin < 0 or obj.ymin < 0 do
              IO.puts("")
            else
              color = Nx.to_flat_list(colors[obj.class_id])

              det_label =
                if obj.class_id in Map.keys(labels_map),
                  do: labels_map[obj.class_id],
                  else: Integer.to_string(obj.class_id)

              :io.format("~9.9s | ~10.6f | ~4.4w | ~4.4w | ~4.4w | ~4.4w | ~13.13w\n", [
                det_label,
                obj.confidence,
                obj.xmin,
                obj.ymin,
                obj.xmax,
                obj.ymax,
                color
              ])

              {:ok, out_img} =
                OpenCV.rectangle(out_img, [obj.xmin, obj.ymin], [obj.xmax, obj.ymax], color,
                  thickness: 2
                )

              {:ok, out_img} =
                OpenCV.putText(
                  out_img,
                  "\##{det_label} #{round(obj.confidence * 100)}%",
                  [obj.xmin, obj.ymin - 7],
                  # , 1)
                  OpenCV.cv_FONT_HERSHEY_COMPLEX(),
                  0.8,
                  color
                )

              out_img
            end
          end)

        if video_stream_map != nil,
          do: OpenCV.VideoWriter.write(video_stream_map.out_stream, output_img)

        input_loop(
          video_stream_map,
          out_filename,
          imgPath,
          h,
          w,
          output_img,
          image_height,
          image_width,
          model_tuple,
          model_params,
          labels_map,
          colors,
          yolo_layer_params,
          output_layer_names,
          config,
          cur_request_id,
          request_id
        )
      end
    else
      IO.puts("Exiting from loop.")

      if video_stream_map != nil do
        OpenCV.VideoWriter.release(video_stream_map.out_stream)
        IO.puts("Saved video file: #{out_filename}")
      else
        OpenCV.imwrite(out_filename, frame)
        IO.puts("Saved image file: #{out_filename}")
      end
    end
  end

  def old_main(imgPath, model_type \\ "yolov3-tiny") do
    # parsing parameters
    config = Map.new(Application.get_all_env(:yolo_elixir_onnx))

    unless File.exists?(imgPath) do
      raise ArgumentError, "Cannot find #{imgPath}"
    end

    unless model_type in config.supported_models do
      raise ArgumentError,
            "Unsupported model #{model_type}. Supported models are: #{config.supported_models}"
    end

    {h, w, onnx_model_filename} =
      case model_type do
        "yolov3-tiny" -> {416, 416, "#{config.model_directory}/yolov3-tiny-416.onnx"}
        "yolov4-tiny" -> {416, 416, "#{config.model_directory}/yolov4-tiny-416.onnx"}
        "yolov3" -> {608, 608, "#{config.model_directory}/yolov3-608.onnx"}
      end

    start_time = Time.utc_now()
    IO.puts("Loading ONNX model...")
    {model_tuple, model_params} = AxonOnnx.Deserialize.__import__(onnx_model_filename)
    loading_time = Time.diff(Time.utc_now(), start_time, :millisecond) / 1000
    IO.puts("Time to load: #{loading_time}s")

    IO.puts("Loading class names...")

    {:ok, labels_map} =
      File.open(config.classes_filename, [:read], fn file ->
        names = IO.read(file, :all) |> String.split("\n") |> Enum.with_index()
        for {n, i} <- names, n != "", into: %{}, do: {i, n}
      end)

    # Generate pseudo-random colors for boxes
    colors = Nx.random_uniform({Enum.count(labels_map), 3}, 0, 255)

    IO.puts("Preparing inputs...")

    yolo_layer_params =
      for node <- Tuple.to_list(model_tuple), into: %{} do
        layer_name = node.name
        shape = Tuple.delete_at(node.parent.output_shape, 0)
        shape = Tuple.delete_at(shape, 0)
        yolo_params = Preprocessing.yoloParams(shape, model_type, node.opts)
        {layer_name, [shape, yolo_params]}
      end

    output_layer_names = Map.keys(yolo_layer_params)

    IO.puts("Preprocessing input image...")
    {:ok, image} = OpenCV.imread(imgPath)
    {:ok, {image_height, image_width, _image_channels}} = OpenCV.Mat.shape(image)

    {:ok, blob} =
      OpenCV.dnn_blobFromImage(
        image,
        scalefactor: 1 / 255.0,
        swapRB: true,
        mean: [0, 0, 0],
        size: [w, h]
      )

    in_frame = OpenCV.Nx.to_nx(blob)

    IO.puts("Starting inference...")
    start_time = Time.utc_now()
    output = Tuple.to_list(Axon.predict(model_tuple, model_params, in_frame, compiler: EXLA))
    parsing_time = Time.diff(Time.utc_now(), start_time, :millisecond) / 1000
    IO.puts("Time to output: #{parsing_time}s")

    start_time = Time.utc_now()

    objects =
      for id <- 0..(Enum.count(output_layer_names) - 1) do
        layer_name = Enum.at(output_layer_names, id)
        out_blob = Enum.at(output, id)
        [_shape, yolo_params] = yolo_layer_params[layer_name]
        {_, _, h, w} = in_frame.shape

        Postprocessing.parse_yolo_region(
          out_blob,
          {h, w},
          {image_height, image_width},
          yolo_params
        )
      end

    objects =
      objects
      |> Enum.concat()
      |> Enum.filter(fn x -> x != nil || x != [] end)
      |> Enum.sort_by(& &1.confidence, :desc)
      |> Enum.group_by(& &1.class_id)

    objects =
      for {_class, output} <- objects do
        class_len = Enum.count(output)
        outp = Enum.with_index(output)

        to_update =
          for {a, i} <- outp, {b, j} <- Enum.slice(outp, i + 1, class_len) do
            test_val = Postprocessing.intersection_over_union(a, b)
            (test_val > config.iou_threshold && [j, j, 0.0]) || [-1, -1, -1]
          end
          |> Enum.filter(fn [_, _, e] -> e == 0.0 end)
          |> Enum.uniq()

        new_outp =
          Enum.map(outp, fn {e, i} ->
            confidence =
              Enum.find_value(to_update, e.confidence, fn x -> x == [i, i, 0.0] && 0.0 end)

            %{e | confidence: confidence}
          end)

        Enum.filter(new_outp, fn o ->
          o.confidence >= config.prob_threshold
        end)
      end
      |> List.flatten()

    parsing_time = Time.diff(Time.utc_now(), start_time, :millisecond) / 1000
    IO.puts("Time to classify: #{parsing_time}s")
    # and args.raw_output_message:
    if Enum.count(objects) > 0 do
      IO.puts("\nDetected boxes for batch 1")
      IO.puts(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")
    end

    output_img =
      Enum.reduce(objects, image, fn obj, out_img ->
        if obj.xmax > image_width or obj.ymax > image_height or obj.xmin < 0 or obj.ymin < 0 do
          IO.puts("")
        else
          color = Nx.to_flat_list(colors[obj.class_id])

          det_label =
            if obj.class_id in Map.keys(labels_map),
              do: labels_map[obj.class_id],
              else: Integer.to_string(obj.class_id)

          :io.format("~9.9s | ~10.6f | ~4.4w | ~4.4w | ~4.4w | ~4.4w | ~13.13w\n", [
            det_label,
            obj.confidence,
            obj.xmin,
            obj.ymin,
            obj.xmax,
            obj.ymax,
            color
          ])

          {:ok, out_img} =
            OpenCV.rectangle(out_img, [obj.xmin, obj.ymin], [obj.xmax, obj.ymax], color,
              thickness: 2
            )

          {:ok, out_img} =
            OpenCV.putText(
              out_img,
              "\##{det_label} #{round(obj.confidence * 100)}%",
              [obj.xmin, obj.ymin - 7],
              # , 1)
              OpenCV.cv_FONT_HERSHEY_COMPLEX(),
              0.8,
              color
            )

          out_img
        end
      end)

    out_filename = "#{Path.rootname(imgPath)}-#{model_type}-#{Time.utc_now()}.jpeg"
    OpenCV.imwrite(out_filename, output_img)
    objects
  end
end
