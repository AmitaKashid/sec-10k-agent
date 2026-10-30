"use client";

import Link from "next/link";
import { useEffect, useState, useSyncExternalStore } from "react";
import { useUser } from "@clerk/nextjs";
import type { ActiveRoute } from "@/lib/types";

type IconProps = { size?: number };

function IconPanelLeft({ size = 18 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <path d="M9 3v18" />
    </svg>
  );
}
function IconPanelLeftClose({ size = 18 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <path d="M9 3v18" />
      <path d="M16 15l-3-3 3-3" />
    </svg>
  );
}
function IconCirclePlus({ size = 20 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <path d="M8 12h8M12 8v8" />
    </svg>
  );
}
function IconBuilding2({ size = 20 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M6 22V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v18Z" />
      <path d="M6 12H4a2 2 0 0 0-2 2v8h4" />
      <path d="M18 9h2a2 2 0 0 1 2 2v11h-4" />
      <path d="M10 6h4M10 10h4M10 14h4M10 18h4" />
    </svg>
  );
}

function subscribeViewport(onChange: () => void) {
  const query = window.matchMedia("(max-width: 480px)");
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

type SidebarProps = {
  activeRoute?: ActiveRoute;
  companyCount?: number;
  healthOk?: boolean;
  onNewThread?: () => void;
};

export default function Sidebar({
  activeRoute = "thread",
  companyCount = 0,
  healthOk = true,
  onNewThread,
}: SidebarProps) {
  const desktop = useSyncExternalStore(
    subscribeViewport,
    () => !window.matchMedia("(max-width: 480px)").matches,
    () => true,
  );
  const [manualOpen, setOpen] = useState<boolean | null>(null);
  const open = manualOpen ?? desktop;
  const { isSignedIn } = useUser();

  useEffect(() => {
    document.body.classList.toggle("is-open", open);
  }, [open]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "b") {
        e.preventDefault();
        setOpen((v) => !(v ?? desktop));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [desktop]);

  // On mobile, tapping a nav link should close the drawer so the user
  // sees the page they navigated to.
  const closeOnMobile = () => {
    if (typeof window !== "undefined" && window.matchMedia("(max-width: 480px)").matches) {
      setOpen(false);
    }
  };

  return (
    <>
      {open && (
        <div
          className="sidebar-backdrop"
          aria-hidden="true"
          onClick={() => setOpen(false)}
        />
      )}
      {!open && (
        <button
          className="sidebar-open-btn"
          aria-label="Open sidebar"
          onClick={() => setOpen(true)}
        >
          <IconPanelLeft />
        </button>
      )}
      <aside className="sidebar">
      <button
        className="toggle-btn"
        aria-label={open ? "Collapse sidebar" : "Open sidebar"}
        aria-expanded={open}
        title={open ? "Collapse sidebar (⌘B)" : "Open sidebar (⌘B)"}
        onClick={() => setOpen((v) => !(v ?? desktop))}
      >
        {open ? <IconPanelLeftClose /> : <IconPanelLeft />}
      </button>

      <div className="brand">
        <div className="brand-name">
          SEC <b>Analyst</b>
        </div>
      </div>

      <nav className="nav">
        <Link
          href="/"
          className={`nav-item ${activeRoute === "thread" ? "is-active" : ""}`}
          data-tip="New thread"
          aria-label="New thread"
          onClick={(event) => {
            if (onNewThread) {
              event.preventDefault();
              onNewThread();
            }
            closeOnMobile();
          }}
        >
          <span className="ic"><IconCirclePlus /></span>
          <span className="text">New thread</span>
        </Link>
        {isSignedIn && (
          <Link
            href="/companies"
            className={`nav-item ${activeRoute === "companies" ? "is-active" : ""}`}
            data-tip="Companies"
            aria-label="Companies"
            onClick={closeOnMobile}
          >
            <span className="ic"><IconBuilding2 /></span>
            <span className="text">Companies</span>
            {companyCount > 0 && <span className="badge">{companyCount}</span>}
          </Link>
        )}
      </nav>

      <div className="status">
        <span className={`status-dot ${healthOk ? "" : "is-down"}`}></span>
        <span className="status-text">
          {healthOk ? "All systems normal" : "Service degraded"}
        </span>
      </div>
      </aside>
    </>
  );
}
