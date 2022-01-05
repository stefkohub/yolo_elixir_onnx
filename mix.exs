defmodule YoloElixirOnnx.MixProject do
  use Mix.Project

  def project do
    [
      app: :yolo_elixir_onnx,
      version: "0.1.0",
      elixir: "~> 1.12",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
      # {:imagineer, git: "https://github.com/tyre/imagineer.git", tag: "master"}
      {:mogrify, "~> 0.9.1"},
      # {:imgutils, "~> 0.1.1"},
      # {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon"},
      # {:evision, "~> 0.1.0-dev", github: "cocoa-xu/evision", branch: "main"},
      # {:nx_evision, "~> 0.1.0-dev", github: "zeam-vm/nx_evision", branch: "main"},
      {:axon, git: "https://github.com/elixir-nx/axon", tag: "main", override: true},
      {:axon_onnx, git: "https://github.com/stefkohub/axon_onnx", override: true},
      {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla"},
      {:torchx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "torchx"},
      {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true}
    ]
  end
end
