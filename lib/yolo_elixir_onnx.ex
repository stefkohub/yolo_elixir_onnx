defmodule YoloElixirOnnx do
  @moduledoc """
  Documentation for `YoloElixirOnnx`.
  """

  require Axon

  alias YoloElixirOnnx.Preprocessing

  @iou_threshold 0.4
  @prob_threshold 0.5
  @classes_filename "../YoloWeights/coco.names"

  @doc """
  Hello world.

  ## Examples

      iex> YoloElixirOnnx.hello()
      :world

  """
  def hello do
    :world
  end

  def main(imgPath) do
    IO.puts "Loading ONNX model..."
    {model, starting_params} = AxonOnnx.Deserialize.__import__("../YoloWeights/yolov3-tiny-416.onnx")
    model_signature = for m <- model, do: Axon.get_model_signature(m)
    model_tuple = List.to_tuple(model)
    # mah...
    {n,c,h,w} = Enum.at(model_signature, 1)|>Tuple.to_list|>Enum.at(0)|>Tuple.to_list|>Enum.at(0)

    IO.puts "Loading class names..."
    {:ok, labels_map} = File.open(@classes_filename, [:read], fn file -> 
      names = IO.read(file, :all)|>String.split("\n")|>Enum.with_index
      for {n, i} <- names, into: %{}, do: {i,n}
    end)

    IO.puts "Preparing inputs..."
    yolo_layer_params = for node <- model, into: %{} do
      layer_name = node.name
      shape = Tuple.delete_at(node.parent.output_shape, 0)
      shape = Tuple.delete_at(shape, 0)
      yolo_params = Preprocessing.yoloParams(shape, "yolov3-tiny", node.opts)
      {layer_name, [shape, yolo_params]}
    end
    output_layer_names = Map.keys(yolo_layer_params)

    IO.puts "Initializing Axon model..."
    model_params = Axon.init(model_tuple, compiler: EXLA)
    #
    # TODO: Add labels...

    IO.puts "Preprocessing input image..."
    image = Mogrify.open(imgPath)|>Mogrify.verbose
    {_, _, _, imgData} = Preprocessing.image_preprocess(image, [416, 416])
    # Change data layout from HWC to CHW
    in_frame = 
      imgData
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n, c, h, w})
      # |> Nx.transpose(axes: [0,1,3,2])
      # |> Nx.divide(255)
    IO.puts "Getting output values..."
    start_time = Time.utc_now
    output = Enum.zip(output_layer_names, Tuple.to_list(Axon.predict(model_tuple, model_params, in_frame, compiler: EXLA)))

    parsing_time = Time.diff(Time.utc_now, start_time)
    IO.puts "Tempo per output: #{parsing_time}s"
    
    objects = 
      for {layer_name, out_blob} <- output do
        [shape, yolo_params] = yolo_layer_params[layer_name]
        {_,_,h,w} = in_frame.shape
        # in_frame_shape = {h,w}
        # image_shape = {image.height, image.width}
        Preprocessing.parse_yolo_region(out_blob, {h,w}, {image.height, image.width}, yolo_params)
      end
    parsing_time = Time.diff(Time.utc_now, start_time)
    IO.puts "Tempo per inferenza: #{parsing_time}s"
    objects = 
      objects
      |> Enum.concat
      |> Enum.filter(fn x -> x != nil end)
      |> Enum.sort_by(& &1.confidence) #, fn a, b -> a>b end)
      |> Enum.group_by(& &1.class_id)

    IO.puts("1 Adesso objects=#{Enum.count(objects)}")

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
    end

    objects
  end
end
