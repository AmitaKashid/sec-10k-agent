const FASTAPI_URL = process.env.FASTAPI_URL ?? "http://localhost:8000";

export async function GET(): Promise<Response> {
  try {
    const upstream = await fetch(`${FASTAPI_URL}/health`, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    if (!upstream.ok) {
      return Response.json({ error: `Backend ${upstream.status}` }, { status: 502 });
    }
    const data = await upstream.json();
    return Response.json(data);
  } catch {
    return Response.json({ error: "Could not reach backend" }, { status: 502 });
  }
}
