"use client";

// Minimal top-right auth control. Step 6 will replace this with an
// in-flow LimitModal triggered when the backend returns
// `auth_required: true`. For now it's the only visible sign-in entry
// point so the JWT-forwarding wiring can be exercised end-to-end.
import { SignedIn, SignedOut, SignInButton, SignUpButton, UserButton } from "@clerk/nextjs";

export function AuthControl() {
  return (
    <div
      style={{
        position: "fixed",
        top: 12,
        right: 12,
        zIndex: 50,
        display: "flex",
        gap: 8,
        alignItems: "center",
      }}
    >
      <SignedOut>
        <SignInButton mode="modal">
          <button type="button" style={authBtn}>Sign in</button>
        </SignInButton>
        <SignUpButton mode="modal">
          <button type="button" style={authBtnGhost}>Sign up</button>
        </SignUpButton>
      </SignedOut>
      <SignedIn>
        <UserButton afterSignOutUrl="/" />
      </SignedIn>
    </div>
  );
}

const authBtn: React.CSSProperties = {
  background: "#1f2024",
  color: "#e5e7eb",
  border: "1px solid #2d2f36",
  borderRadius: 6,
  padding: "6px 10px",
  fontSize: 13,
  cursor: "pointer",
};

const authBtnGhost: React.CSSProperties = {
  ...authBtn,
  background: "transparent",
};
