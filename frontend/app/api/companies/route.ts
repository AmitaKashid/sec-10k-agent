import { auth } from "@clerk/nextjs/server";
import type { Company } from "@/lib/types";

const FASTAPI_URL = process.env.FASTAPI_URL ?? "http://localhost:8000";
const API_KEY = process.env.API_KEY ?? "";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(): Promise<Response> {
  // Forward the user's Clerk JWT so the backend resolves them as the
  // user tier. Anon callers omit the Bearer header and the backend
  // returns 401 auth_required — the page CTA handles that.
  // X-API-Key is kept as a fallback so admin/eval scripts that go
  // through this proxy keep working until they're migrated to Clerk.
  const headers: Record<string, string> = {
    "X-API-Key": API_KEY,
    Accept: "application/json",
  };
  try {
    const { getToken } = await auth();
    const token = await getToken();
    if (token) headers.Authorization = `Bearer ${token}`;
  } catch {
    // No Clerk session — proceed; backend will 401 anon-tier.
  }

  try {
    const upstream = await fetch(`${FASTAPI_URL}/companies`, {
      headers,
      cache: "no-store",
    });
    if (upstream.status === 401) {
      return Response.json({ error: "auth_required" }, { status: 401 });
    }
    if (!upstream.ok) {
      return Response.json({ error: `Backend ${upstream.status}` }, { status: 502 });
    }
    const data = (await upstream.json()) as Company[];
    return Response.json(data);
  } catch {
    return Response.json({ error: "Could not reach backend" }, { status: 502 });
  }
}
