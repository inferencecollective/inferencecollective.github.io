defmodule AiDataFoundation.PublicPages do
  @moduledoc "Route inventory shared by the Pages exporter and sitemap generator."
  alias AiDataFoundation.Foundation

  def paths do
    [
      "/",
      "/tools",
      "/models",
      "/labs",
      "/workshops",
      "/blog",
      "/roadmap",
      "/about",
      "/security"
    ] ++
      Enum.map(Foundation.workshops(), &"/workshops/#{&1["id"]}") ++
      Enum.map(Foundation.labs(), &"/labs/#{&1["id"]}") ++
      Enum.map(Foundation.posts(), &"/blog/#{&1["id"]}")
  end
end
