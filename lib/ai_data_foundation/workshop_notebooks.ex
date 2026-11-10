defmodule AiDataFoundation.WorkshopNotebooks do
  @moduledoc "Notebook launch links and trusted shared-session validation."

  def download_url(id), do: "/aidatafoundation/notebooks/#{id}.livemd"

  def source_url(id) do
    base =
      Application.get_env(
        :ai_data_foundation,
        :notebook_base_url,
        "https://inferencecollective.github.io/aidatafoundation/notebooks"
      )

    String.trim_trailing(base, "/") <> "/#{id}.livemd"
  end

  def launch_url(id) do
    case Application.get_env(:ai_data_foundation, :livebook_url) do
      nil ->
        "https://livebook.dev/run/?" <> URI.encode_query(%{"url" => source_url(id)})

      base ->
        String.trim_trailing(base, "/") <>
          "/import?" <> URI.encode_query(%{"url" => source_url(id)})
    end
  end

  def room_url(id) do
    case Application.get_env(:ai_data_foundation, :workshop_live_url) do
      nil -> nil
      base -> String.trim_trailing(base, "/") <> "/workshops/#{id}/room"
    end
  end

  def trusted_session(url) when is_binary(url) and byte_size(url) <= 2048 do
    with base when is_binary(base) <- Application.get_env(:ai_data_foundation, :livebook_url),
         %URI{} = trusted <- URI.parse(base),
         %URI{} = candidate <- URI.parse(url),
         true <- candidate.scheme in ["http", "https"],
         true <-
           {candidate.scheme, candidate.host, candidate.port} ==
             {trusted.scheme, trusted.host, trusted.port},
         nil <- candidate.userinfo,
         nil <- candidate.query,
         nil <- candidate.fragment,
         prefix = String.trim_trailing(trusted.path || "", "/"),
         true <- String.starts_with?(candidate.path || "", prefix <> "/sessions/"),
         id = String.replace_prefix(candidate.path, prefix <> "/sessions/", ""),
         true <- Regex.match?(~r/\A[a-zA-Z0-9_-]+\z/, id) do
      {:ok, URI.to_string(candidate)}
    else
      _ -> :error
    end
  end

  def trusted_session(_), do: :error
end
