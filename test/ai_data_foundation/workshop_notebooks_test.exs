defmodule AiDataFoundation.WorkshopNotebooksTest do
  use ExUnit.Case, async: false
  alias AiDataFoundation.WorkshopNotebooks

  setup do
    before = Application.get_env(:ai_data_foundation, :livebook_url)

    Application.put_env(
      :ai_data_foundation,
      :livebook_url,
      "https://notebooks.example.test/livebook"
    )

    on_exit(fn -> Application.put_env(:ai_data_foundation, :livebook_url, before) end)
  end

  test "launch link imports the actual notebook on the configured server" do
    uri = URI.parse(WorkshopNotebooks.launch_url("mlx"))
    assert uri.host == "notebooks.example.test"
    assert uri.path == "/livebook/import"
    assert URI.decode_query(uri.query)["url"] =~ "/notebooks/mlx.livemd"
  end

  test "session embeds accept only the configured origin and session paths" do
    assert {:ok, _} =
             WorkshopNotebooks.trusted_session(
               "https://notebooks.example.test/livebook/sessions/abc123"
             )

    for url <- [
          "javascript:alert(1)",
          "https://evil.example/livebook/sessions/abc",
          "https://notebooks.example.test.evil/livebook/sessions/abc",
          "https://notebooks.example.test/livebook/sessions/abc?token=secret",
          "https://notebooks.example.test/livebook/sessions/../settings",
          "https://notebooks.example.test/livebook/settings"
        ] do
      assert :error == WorkshopNotebooks.trusted_session(url)
    end
  end
end
