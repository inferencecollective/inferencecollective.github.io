defmodule AiDataFoundationWeb.WorkshopControllerTest do
  use AiDataFoundationWeb.ConnCase

  alias AiDataFoundation.{Foundation, PublicPages}

  test "catalog links every workshop and exports it", %{conn: conn} do
    document = conn |> get(~p"/workshops") |> html_response(200) |> LazyHTML.from_document()

    for track <- Foundation.workshops() do
      path = "/workshops/#{track["id"]}"
      assert path in PublicPages.paths()
      assert LazyHTML.query(document, "#track-#{track["id"]} a[href='#{path}']") |> Enum.any?()
    end
  end

  test "each track includes setup and sequential runnable labs", %{conn: conn} do
    for track <- Foundation.workshops() do
      document =
        conn |> get("/workshops/#{track["id"]}") |> html_response(200) |> LazyHTML.from_document()

      assert LazyHTML.query(document, "#setup pre code") |> Enum.any?()
      assert LazyHTML.query(document, "#finish") |> Enum.any?()

      for {step, index} <- Enum.with_index(track["steps"], 1) do
        assert LazyHTML.query(document, "#step-#{index} a[href='/labs/#{step["guide"]}']")
               |> Enum.any?()

        assert LazyHTML.query(document, "input[data-workshop-step='#{index}']") |> Enum.any?()
      end
    end
  end

  test "unknown workshop returns 404", %{conn: conn} do
    conn = get(conn, "/workshops/missing")
    assert conn.status == 404
  end
end
