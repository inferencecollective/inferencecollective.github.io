defmodule AiDataFoundationWeb.WorkshopRoomLive do
  use AiDataFoundationWeb, :live_view
  alias AiDataFoundation.{Foundation, WorkshopNotebooks, WorkshopPresence}

  def mount(%{"id" => id}, _session, socket) do
    case Foundation.workshop(id) do
      nil ->
        {:ok, push_navigate(socket, to: "/workshops")}

      track ->
        {:ok,
         assign(socket,
           track: track,
           page_title: "Workshop room · " <> track["title"],
           room: nil,
           topic: nil,
           participant_id: random_id(),
           joined?: false,
           name: "",
           completed: [],
           help?: false,
           session_url: nil,
           participant_count: 0,
           active_notebook: nil,
           share_url: nil,
           join_form: to_form(%{"name" => ""}),
           notebook_form: to_form(%{"url" => ""}),
           error: nil
         )
         |> stream(:participants, [])}
    end
  end

  def handle_params(%{"room" => room}, uri, socket) do
    if Regex.match?(~r/\A[a-f0-9]{32}\z/, room) do
      topic = "workshop:#{socket.assigns.track["id"]}:#{room}"

      if connected?(socket) and socket.assigns.topic != topic do
        if socket.assigns.topic do
          if socket.assigns.joined?,
            do:
              WorkshopPresence.untrack(
                self(),
                socket.assigns.topic,
                socket.assigns.participant_id
              )

          Phoenix.PubSub.unsubscribe(AiDataFoundation.PubSub, socket.assigns.topic)
        end

        Phoenix.PubSub.subscribe(AiDataFoundation.PubSub, topic)
      end

      socket =
        if socket.assigns.topic != topic,
          do:
            assign(socket,
              joined?: false,
              completed: [],
              help?: false,
              session_url: nil,
              active_notebook: nil
            ),
          else: socket

      {:noreply, socket |> assign(room: room, topic: topic, share_url: uri) |> refresh()}
    else
      {:noreply, push_patch(socket, to: room_path(socket.assigns.track["id"], random_id()))}
    end
  end

  def handle_params(_, _, socket) do
    {:noreply, push_patch(socket, to: room_path(socket.assigns.track["id"], random_id()))}
  end

  def handle_event("join", %{"name" => raw}, socket) when is_binary(raw) do
    name = raw |> String.trim() |> String.slice(0, 40)

    cond do
      socket.assigns.joined? ->
        {:noreply, socket}

      name == "" ->
        {:noreply, assign(socket, error: "Enter a display name to join.")}

      true ->
        socket = assign(socket, name: name, joined?: true, error: nil)

        {:ok, _} =
          WorkshopPresence.track(
            self(),
            socket.assigns.topic,
            socket.assigns.participant_id,
            metadata(socket)
          )

        {:noreply, refresh(socket)}
    end
  end

  def handle_event("toggle_step", %{"step" => step}, socket) do
    with true <- socket.assigns.joined?,
         {index, ""} <- Integer.parse(step),
         true <- index in 1..length(socket.assigns.track["steps"]) do
      completed = socket.assigns.completed

      completed =
        if index in completed, do: List.delete(completed, index), else: [index | completed]

      {:noreply, socket |> assign(completed: completed) |> update_presence()}
    else
      _ -> {:noreply, socket}
    end
  end

  def handle_event("toggle_help", _, socket) do
    if socket.assigns.joined? do
      {:noreply, socket |> assign(help?: not socket.assigns.help?) |> update_presence()}
    else
      {:noreply, socket}
    end
  end

  def handle_event("share_notebook", %{"url" => url}, socket) do
    case {socket.assigns.joined?, WorkshopNotebooks.trusted_session(url)} do
      {true, {:ok, url}} ->
        {:noreply, socket |> assign(session_url: url, error: nil) |> update_presence()}

      _ ->
        {:noreply,
         assign(socket,
           error:
             "Use a session URL from the configured Livebook server, without a token or query string."
         )}
    end
  end

  def handle_event("show_notebook", %{"participant" => participant}, socket) do
    url =
      WorkshopPresence.list(socket.assigns.topic)
      |> Map.get(participant, %{metas: []})
      |> Map.get(:metas)
      |> Enum.find_value(& &1[:session_url])

    case WorkshopNotebooks.trusted_session(url) do
      {:ok, url} -> {:noreply, assign(socket, active_notebook: url)}
      :error -> {:noreply, assign(socket, error: "That shared notebook is no longer available.")}
    end
  end

  def handle_event("close_notebook", _, socket),
    do: {:noreply, assign(socket, active_notebook: nil)}

  def handle_info(%Phoenix.Socket.Broadcast{event: "presence_diff"}, socket),
    do: {:noreply, refresh(socket)}

  defp refresh(socket) do
    participants =
      WorkshopPresence.list(socket.assigns.topic)
      |> Enum.map(fn {id, %{metas: [meta | _]}} -> Map.put(meta, :id, id) end)
      |> Enum.sort_by(&{&1.name, &1.id})

    socket
    |> assign(participant_count: length(participants))
    |> stream(:participants, participants, reset: true)
  end

  defp metadata(socket), do: Map.take(socket.assigns, [:name, :completed, :help?, :session_url])

  defp update_presence(socket) do
    {:ok, _} =
      WorkshopPresence.update(
        self(),
        socket.assigns.topic,
        socket.assigns.participant_id,
        metadata(socket)
      )

    refresh(socket)
  end

  defp random_id, do: :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)
  defp room_path(id, room), do: "/workshops/#{id}/room?room=#{room}"
end
