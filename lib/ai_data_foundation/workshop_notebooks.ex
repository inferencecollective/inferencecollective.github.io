defmodule AiDataFoundation.WorkshopNotebooks do
  @moduledoc "Download, preview, and Colab links for workshop notebooks."
  @repository "inferencecollective/inferencecollective.github.io"

  def download_url(id), do: "/aidatafoundation/notebooks/#{id}.ipynb"

  def preview_url(id) do
    "https://github.com/#{@repository}/blob/#{revision()}/priv/static/aidatafoundation/notebooks/#{id}.ipynb"
  end

  def colab_url("mlx"), do: nil

  def colab_url(id) do
    "https://colab.research.google.com/github/#{@repository}/blob/#{revision()}/priv/static/aidatafoundation/notebooks/#{id}.ipynb"
  end

  defp revision,
    do: Application.get_env(:ai_data_foundation, :notebook_revision, "codex/ml-workshop-tracks")
end
