// Vercel serverless proxy to the read-only Supabase results API.
// Only whitelisted api_* RPC functions are reachable; the Supabase key stays
// in Vercel environment variables and is never sent to the browser.

const ENDPOINTS = {
  stations: { fn: "api_stations", params: [] },
  champion: { fn: "api_champion_model", params: [] },
  models: { fn: "api_model_history", params: ["p_limit"] },
  submissions: { fn: "api_submissions", params: ["p_limit"] },
  accuracy: { fn: "api_accuracy_summary", params: ["p_days"] },
  status: { fn: "api_pipeline_status", params: [] },
  demand: { fn: "api_station_demand", params: ["p_station_id", "p_hours"] },
  predictions: {
    fn: "api_predictions_vs_actuals",
    params: ["p_station_id", "p_from", "p_to"],
  },
};

export default async function handler(req, res) {
  if (req.method !== "GET") {
    res.setHeader("Allow", "GET");
    return res.status(405).json({ error: "Método no permitido" });
  }

  const endpoint = ENDPOINTS[req.query.endpoint];
  if (!endpoint) {
    return res.status(404).json({
      error: "Endpoint desconocido",
      available: Object.keys(ENDPOINTS),
    });
  }

  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_PUBLISHABLE_KEY;
  if (!url || !key) {
    return res.status(500).json({ error: "Faltan SUPABASE_URL o SUPABASE_PUBLISHABLE_KEY en Vercel" });
  }

  const body = {};
  for (const name of endpoint.params) {
    const value = req.query[name.replace(/^p_/, "")];
    if (typeof value === "string" && value !== "") body[name] = value;
  }

  try {
    const response = await fetch(`${url.replace(/\/$/, "")}/rest/v1/rpc/${endpoint.fn}`, {
      method: "POST",
      headers: {
        apikey: key,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok) {
      return res.status(response.status).json({ error: data.message || "Error de Supabase" });
    }
    res.setHeader("Cache-Control", "s-maxage=60, stale-while-revalidate=300");
    return res.status(200).json(data);
  } catch {
    return res.status(502).json({ error: "No se pudo contactar a Supabase" });
  }
}
