defmodule YoloElixirOnnx do
  @moduledoc """
  Documentation for `YoloElixirOnnx`.
  """

  require Axon
  require Logger

  alias YoloElixirOnnx.Preprocessing

  EXLA.set_preferred_defn_options([:tpu, :cuda, :rocm, :host])

  @iou_threshold 0.4
  @prob_threshold 0.5
  @classes_filename "../YoloWeights/coco.names"
  @onnx_model_filename "../YoloWeights/yolov3-tiny-416.onnx"
  # @onnx_model_filename "../tensorrt_demos/yolo/yolov3-tiny-416.onnx"
  
  @doc """

  """
  def main(imgPath) do
    IO.puts "Loading ONNX model..."
    {model, starting_params} = AxonOnnx.Deserialize.__import__(@onnx_model_filename)
    model_signature = for m <- model, do: Axon.get_model_signature(m)
    # model_tuple = List.to_tuple(Enum.map(model, fn m -> Axon.freeze(m) end))
    model_tuple = List.to_tuple(model)
    # mah...
    #{n,c,h,w} = Enum.at(model_signature, 1)|>Tuple.to_list|>Enum.at(0)|>Tuple.to_list|>Enum.at(0)
    {{n,c,h,w}, _} = Enum.at(model_signature, 0)

    IO.puts "Loading class names..."
    {:ok, labels_map} = File.open(@classes_filename, [:read], fn file -> 
      names = IO.read(file, :all)|>String.split("\n")|>Enum.with_index
      for {n, i} <- names, n != "", into: %{}, do: {i,n}
    end)

    IO.puts "Preparing inputs..."
    yolo_layer_params = for node <- model, into: %{} do
      layer_name = node.name
      shape = Tuple.delete_at(node.parent.output_shape, 0)
      shape = Tuple.delete_at(shape, 0)
      yolo_params = Preprocessing.yoloParams(shape, "yolov3-tiny", node.opts)
      {layer_name, [shape, yolo_params]}
    end
    IO.puts inspect(yolo_layer_params)
    output_layer_names = Map.keys(yolo_layer_params)

    IO.puts "Initializing Axon model..."
    model_params = Axon.init(model_tuple, compiler: EXLA)
    #
    # TODO: Add labels...

    IO.puts "Preprocessing input image..."
    {:ok, image} = OpenCV.imread(imgPath)
    {:ok, image_type} = OpenCV.Mat.type(image)
    {:ok, {image_height, image_width, image_channels}}=OpenCV.Mat.shape(image)
    
    in_frame = 
      OpenCV.resize(image, [w, h])
      |> case do
        {:ok, res} -> res #Preprocessing.letterbox(res, [w, h])
        _ -> raise "Error in transpose"
      end
      |> Preprocessing.letterbox([w, h])
      |> OpenCV.transpose(axes: [2, 0, 1])
      |> case do
        {:ok, res} -> OpenCV.Mat.to_binary(res)
        _ -> raise "Error in transpose"
      end
      |> case do 
        {:ok, res} -> Nx.from_binary(res, image_type)
        _ -> raise "Error in Mat.to_binary"
      end
      |> Nx.reshape({n, c, h, w})
      # |> Nx.divide(255.0)

    IO.puts "Getting output values from #{inspect(in_frame)}..."
    start_time = Time.utc_now
    #model_params = %{ model_params | "016_convolutional" => Map.merge(model_params["016_convolutional"], %{ "bias" => starting_params["016_convolutional_bias"] })}
    #model_params = %{ model_params | "023_convolutional" => Map.merge(model_params["023_convolutional"], %{ "bias" => starting_params["023_convolutional_bias"] })}
    #kernels = Map.keys(starting_params)|>Enum.filter(fn x -> String.match?(x, ~r/.*_kernel.*/) end)
    #betas = Map.keys(starting_params)|>Enum.filter(fn x -> String.match?(x, ~r/.*_beta.*/) end)
    #model_params=for {k, _v} <- model_params do
    #  kname = k<>"_kernel"
    #  if kname in kernels,
    #    do: { k, Map.merge(model_params[k], %{ "kernel" => starting_params[kname] })},
    #    else: { k, model_params[k]}
    #  end 
    #model_params=for {b, v} <- model_params do
    #    bname = b<>"_beta"
    #    if bname in betas,
    #      do: { b, Map.merge(v, %{ "beta" => starting_params[bname] })},
    #      else: { b, v}
    #  end |> Enum.into(%{})

    model_params = for {name,nmap} <- model_params, into: %{} do
      { name, nmap
        |> Enum.map(fn {k,_val} ->
               sp_name = "#{name}_#{k}"
               { k , starting_params[sp_name]}
        end) 
        |> Map.new
      }
    end
    output = Tuple.to_list(Axon.predict(model_tuple, model_params, in_frame, compiler: EXLA))

    parsing_time = Time.diff(Time.utc_now, start_time)
    IO.puts "Tempo per output: #{parsing_time}s"
    
    objects = 
      # for {layer_name, out_blob} <- output do
      for id <- 0..Enum.count(output_layer_names)-1 do
        layer_name = Enum.at(output_layer_names, id)
        out_blob = Enum.at(output, id)
        [shape, yolo_params] = yolo_layer_params[layer_name]
        {_,_,h,w} = in_frame.shape
        # in_frame_shape = {h,w}
        # image_shape = {image.height, image.width}
        IO.inspect([out_blob, {h,w}, {image_height, image_width}, yolo_params])
        Preprocessing.parse_yolo_region(out_blob, {h,w}, {image_height, image_width}, yolo_params)
      end
    parsing_time = Time.diff(Time.utc_now, start_time)
    IO.puts "Tempo per inferenza: #{parsing_time}s"
    IO.puts(inspect(objects))
    objects = 
      objects
      |> Enum.concat
      |> Enum.filter(fn x -> x != nil || x !=  [] end)
      |> Enum.sort_by(& &1.confidence) 
      |> Enum.group_by(& &1.class_id)

    IO.puts("1 Adesso objects=#{Enum.count(objects)}")
    IO.puts("1 Adesso objects=")
    Enum.map(objects, fn {class_id, class} -> 
      Enum.each(class, fn i -> 
        IO.puts("class_id: #{class_id}, confidence: #{i.confidence}") 
      end) 
    end)

    start_time = Time.utc_now
    objects = 
      for {class, output} <- objects do
        class_len = Enum.count(output)
        outp = Enum.with_index(output)
        to_update =
          for {a, i} <- outp, {b, j} <- Enum.slice(outp, i+1, class_len) do
            test_val = YoloElixirOnnx.Preprocessing.intersection_over_union(a, b)
            test_val > @iou_threshold && [j,j,0.0] || [i,i,test_val]
          end
          |> Enum.filter(fn [_,_,e] -> e == 0.0 end)
          |> Enum.uniq
        Enum.map(outp, fn {e, i} ->
          confidence = Enum.find_value(to_update, e.confidence, fn x -> x == [i, i, 0.0] && 0.0 end)
          %{e | confidence: confidence}
        end)
        |> Enum.filter(fn o -> 
          if o.confidence != 0.0, do: IO.inspect("#{labels_map[o.class_id]} confidence=#{o.confidence}") 
          o.confidence >= @prob_threshold 
        end)
      end
      
    parsing_time = Time.diff(Time.utc_now, start_time)
    IO.puts "Tempo per loop: #{parsing_time}s"
    IO.puts("2 Adesso objects=#{Enum.count(objects)}")
    if Enum.count(objects)>0 do   #and args.raw_output_message:
      IO.puts("\nDetected boxes for batch 1")
      IO.puts(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")
      Enum.each(objects, fn o ->
        IO.puts(" #{o.class_id} | #{o.confidence} | #{o.xmin} | #{o.ymin o.xmax}|#{o.ymax} | #{o.color}")
      end)
    end

    objects
  end
end
