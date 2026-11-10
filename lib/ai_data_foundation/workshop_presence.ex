defmodule AiDataFoundation.WorkshopPresence do
  use Phoenix.Presence,
    otp_app: :ai_data_foundation,
    pubsub_server: AiDataFoundation.PubSub
end
