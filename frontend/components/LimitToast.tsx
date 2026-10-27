"use client";

// Lightweight toast for signed-in users who hit their per-minute cap.
// Auto-dismisses after window_s seconds (mirrors the backend's
// Retry-After). Anonymous users get the LimitModal instead.
import { useEffect } from "react";
import type { LimitErrorPayload } from "@/lib/types";

type Props = {
  payload: LimitErrorPayload | null;
  onDismiss: () => void;
};

export function LimitToast({ payload, onDismiss }: Props) {
  useEffect(() => {
    if (!payload) return;
    const ms = Math.max(2000, payload.limit.window_s * 1000);
    const t = setTimeout(onDismiss, ms);
    return () => clearTimeout(t);
  }, [payload, onDismiss]);

  if (!payload) return null;

  const isRequest = payload.reason === "authed_req_limit";
  const label = isRequest ? "request" : "token";

  return (
    <div style={toast} role="status" aria-live="polite">
      <div style={dot} />
      <div>
        <div style={title}>
          You hit your per-minute {label} limit
        </div>
        <div style={body}>
          {payload.limit.used} / {payload.limit.budget}. Try again in {payload.limit.window_s}s.
        </div>
      </div>
      <button type="button" onClick={onDismiss} style={dismissBtn} aria-label="Dismiss">
        ×
      </button>
    </div>
  );
}

const toast: React.CSSProperties = {
  position: "fixed",
  bottom: 24,
  left: "50%",
  transform: "translateX(-50%)",
  background: "#15161a",
  color: "#e5e7eb",
  border: "1px solid #2d2f36",
  borderRadius: 8,
  padding: "10px 14px",
  zIndex: 90,
  display: "flex",
  alignItems: "center",
  gap: 12,
  maxWidth: 420,
  boxShadow: "0 12px 30px rgba(0,0,0,0.4)",
};

const dot: React.CSSProperties = {
  width: 8,
  height: 8,
  borderRadius: "50%",
  background: "#f59e0b",
  flexShrink: 0,
};

const title: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 500,
};

const body: React.CSSProperties = {
  fontSize: 12,
  color: "#9ca3af",
};

const dismissBtn: React.CSSProperties = {
  background: "transparent",
  color: "#6b7280",
  border: "none",
  fontSize: 18,
  cursor: "pointer",
  padding: 0,
  marginLeft: 4,
};
