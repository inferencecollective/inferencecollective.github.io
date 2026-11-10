defmodule AiDataFoundationWeb.WorkshopRoomLiveTest do
  use AiDataFoundationWeb.ConnCase
  import Phoenix.LiveViewTest

  @room String.duplicate("a", 32)

  test "participants see each other's progress and help requests", %{conn: conn} do
    path = "/workshops/vision/room?room=#{@room}"
    {:ok, first, _} = live(conn, path)
    {:ok, second, _} = live(build_conn(), path)
    first |> form("#join-room", name: "Alice") |> render_submit()
    second |> form("#join-room", name: "Bob") |> render_submit()
    first |> element("#room-step-1") |> render_click()
    first |> element("#room-help") |> render_click()
    # Request a fresh server render after the Presence diff arrives.
    assert eventually(fn -> has_element?(second, ".room-person", "Needs help") end)
    assert has_element?(second, ".room-person", "Alice")
    assert has_element?(second, ".room-person", "1 / 2 labs")
    assert has_element?(first, "#room-help[aria-pressed=true]")
  end

  test "rooms are isolated and invalid progress is ignored", %{conn: conn} do
    {:ok, one, _} = live(conn, "/workshops/vision/room?room=#{@room}")
    {:ok, two, _} = live(build_conn(), "/workshops/vision/room?room=#{String.duplicate("b", 32)}")
    one |> form("#join-room", name: "Private group") |> render_submit()
    render_click(one, "toggle_step", %{"step" => "999"})
    refute has_element?(two, ".room-person", "Private group")
    refute has_element?(one, "#room-step-1[aria-pressed=true]")
  end

  test "trusted Livebook sessions can be shared and embedded", %{conn: conn} do
    previous = Application.get_env(:ai_data_foundation, :livebook_url)
    Application.put_env(:ai_data_foundation, :livebook_url, "https://notebooks.example.test")
    on_exit(fn -> Application.put_env(:ai_data_foundation, :livebook_url, previous) end)
    {:ok, view, _} = live(conn, "/workshops/vision/room?room=#{@room}")
    view |> form("#join-room", name: "Instructor") |> render_submit()
    view |> form("#share-notebook", url: "https://evil.example/sessions/test") |> render_submit()
    assert has_element?(view, "#room-error")

    view
    |> form("#share-notebook", url: "https://notebooks.example.test/sessions/abc123")
    |> render_submit()

    view |> element(".room-person button") |> render_click()
    assert has_element?(view, "iframe[src='https://notebooks.example.test/sessions/abc123']")
  end

  defp eventually(check, attempts \\ 50)
  defp eventually(check, 0), do: check.()

  defp eventually(check, attempts) do
    if check.(), do: true, else: eventually(check, attempts - 1)
  end
end
