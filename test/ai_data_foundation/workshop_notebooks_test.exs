defmodule AiDataFoundation.WorkshopNotebooksTest do
  use ExUnit.Case, async: true
  alias AiDataFoundation.WorkshopNotebooks

  test "downloads and previews point to ipynb files" do
    for id <- ~w(ml-foundations pytorch vision mlx) do
      assert WorkshopNotebooks.download_url(id) == "/aidatafoundation/notebooks/#{id}.ipynb"

      assert WorkshopNotebooks.preview_url(id) =~
               "github.com/inferencecollective/inferencecollective.github.io/blob/"

      assert WorkshopNotebooks.preview_url(id) =~ "/#{id}.ipynb"
    end
  end

  test "only portable tracks launch in hosted Colab" do
    assert is_nil(WorkshopNotebooks.colab_url("mlx"))

    for id <- ~w(ml-foundations pytorch vision) do
      url = URI.parse(WorkshopNotebooks.colab_url(id))
      assert url.host == "colab.research.google.com"
      assert url.path =~ "/#{id}.ipynb"
    end
  end
end
