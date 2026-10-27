"use client";

// Soft gate: rendered when the backend's anon tier returns 413/429.
// The pending question is preserved in the page's input (we don't touch
// it here); the user signs in via Clerk's hosted modal and re-submits.
//
// Uses Clerk's <SignInButton mode="modal"> to delegate the actual
// sign-in UI to Clerk — avoids embedding <SignIn routing="virtual">,
// which has more edge cases with our existing layout.
import { useEffect } from "react";
import { SignInButton, useUser } from "@clerk/nextjs";
import type { LimitErrorPayload } from "@/lib/types";

type Props = {
  payload: LimitErrorPayload | null;
  pendingQuestion: string;
  onDismiss: () => void;
};

export function LimitModal({ payload, pendingQuestion, onDismiss }: Props) {
  const { isSignedIn } = useUser();

  // Auto-close once Clerk reports a signed-in session. The user's
  // pending question stays in the input, ready to resubmit.
  useEffect(() => {
    if (payload && isSignedIn) onDismiss();
  }, [payload, isSignedIn, onDismiss]);

  if (!payload) return null;

  const isInputTooLarge = payload.reason === "input_too_large";
  const isLifetime = payload.reason === "anonymous_lifetime_limit";
  const isTickerBlocked = payload.reason === "anon_ticker_not_allowed";

  let headline: string;
  let subline: string;
  if (isInputTooLarge) {
    headline = "Your question is too long";
    subline = `${payload.limit.used} tokens — max ${payload.limit.budget} on the anonymous tier.`;
  } else if (isLifetime) {
    headline = `You've used your ${payload.limit.budget} free questions`;
    subline = "Sign in to keep asking. Signed-in users get the full company universe and a much higher budget.";
  } else if (isTickerBlocked) {
    headline = "Sign in to ask about this company";
    subline = "The free preview only covers AAPL, MSFT, GOOGL, AMZN, and NVDA. Sign in to ask about any of the 50 covered companies.";
  } else {
    headline = "You've hit the free tier limit";
    subline = "Sign in to keep asking. Authenticated users get a much higher per-minute budget.";
  }

  return (
    <div style={overlay} role="dialog" aria-modal="true" aria-labelledby="limit-modal-title">
      <div style={card}>
        <h2 id="limit-modal-title" style={title}>{headline}</h2>
        <p style={body}>{subline}</p>
        {pendingQuestion && (
          <div style={pendingBox}>
            <div style={pendingLabel}>Your question</div>
            <div style={pendingText}>{pendingQuestion}</div>
          </div>
        )}
        <div style={actions}>
          {!isInputTooLarge && (
            <SignInButton mode="modal">
              <button type="button" style={primaryBtn}>Sign in to continue</button>
            </SignInButton>
          )}
          <button type="button" onClick={onDismiss} style={secondaryBtn}>
            {isInputTooLarge ? "Edit my question" : "Maybe later"}
          </button>
        </div>
      </div>
    </div>
  );
}

const overlay: React.CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "rgba(0,0,0,0.55)",
  zIndex: 100,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  padding: 16,
  backdropFilter: "blur(2px)",
};

const card: React.CSSProperties = {
  background: "#15161a",
  color: "#e5e7eb",
  border: "1px solid #2d2f36",
  borderRadius: 12,
  padding: 24,
  maxWidth: 440,
  width: "100%",
  boxShadow: "0 18px 50px rgba(0,0,0,0.4)",
};

const title: React.CSSProperties = {
  margin: 0,
  marginBottom: 8,
  fontSize: 20,
  fontWeight: 600,
};

const body: React.CSSProperties = {
  margin: 0,
  marginBottom: 16,
  color: "#9ca3af",
  fontSize: 14,
  lineHeight: 1.5,
};

const pendingBox: React.CSSProperties = {
  background: "#1f2024",
  borderRadius: 8,
  padding: 12,
  marginBottom: 16,
};

const pendingLabel: React.CSSProperties = {
  fontSize: 11,
  letterSpacing: 0.6,
  textTransform: "uppercase",
  color: "#6b7280",
  marginBottom: 4,
};

const pendingText: React.CSSProperties = {
  fontSize: 14,
  color: "#e5e7eb",
};

const actions: React.CSSProperties = {
  display: "flex",
  gap: 8,
  justifyContent: "flex-end",
};

const primaryBtn: React.CSSProperties = {
  background: "#3b82f6",
  color: "white",
  border: "none",
  borderRadius: 6,
  padding: "8px 14px",
  fontSize: 14,
  fontWeight: 500,
  cursor: "pointer",
};

const secondaryBtn: React.CSSProperties = {
  background: "transparent",
  color: "#9ca3af",
  border: "1px solid #2d2f36",
  borderRadius: 6,
  padding: "8px 14px",
  fontSize: 14,
  cursor: "pointer",
};
