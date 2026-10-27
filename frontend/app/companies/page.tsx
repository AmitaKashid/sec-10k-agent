"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { useUser, SignInButton } from "@clerk/nextjs";
import Sidebar from "@/components/Sidebar";
import type { Company, CompanyStats, CoverageStatus, Health } from "@/lib/types";

type IconProps = { size?: number };

function IconSearch({ size = 14 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="11" cy="11" r="7" />
      <path d="M21 21l-4.3-4.3" />
    </svg>
  );
}
function IconArrowRight({ size = 10 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M5 12h14M13 5l7 7-7 7" />
    </svg>
  );
}

function CoverageDot({ status }: { status?: CoverageStatus | null }) {
  if (!status) return <span className="cov is-empty">—</span>;
  const cls = status === "complete" ? "is-ok" : status === "pending" ? "is-warn" : "is-empty";
  return (
    <span className={`cov ${cls}`}>
      <span className="cov-dot" aria-hidden="true" />
      {status}
    </span>
  );
}

function Stat({
  label,
  val,
  sub,
  hideOnMobile,
}: {
  label: string;
  val: string;
  sub?: string;
  hideOnMobile?: boolean;
}) {
  return (
    <div className={`stat ${hideOnMobile ? "is-desktop-only" : ""}`}>
      <span className="stat-label">{label}</span>
      <span className="stat-val">
        {val}
        {sub && <span className="stat-sub">{sub}</span>}
      </span>
    </div>
  );
}

export default function CompaniesPage() {
  const router = useRouter();
  const { isSignedIn, isLoaded } = useUser();
  const [companies, setCompanies] = useState<Company[]>([]);
  const [stats, setStats] = useState<CompanyStats | null>(null);
  const [healthOk, setHealthOk] = useState<boolean>(true);
  const [query, setQuery] = useState<string>("");
  const [activeSector, setActiveSector] = useState<string>("All");

  useEffect(() => {
    if (!isSignedIn) return; // anon: page renders the sign-in CTA instead
    let cancelled = false;
    void (async () => {
      try {
        const res = await fetch("/api/companies");
        if (!cancelled && res.ok) {
          const data = (await res.json()) as Company[];
          setCompanies(data);
        }
      } catch {}
      try {
        const res = await fetch("/api/companies/stats");
        if (!cancelled && res.ok) {
          const data = (await res.json()) as CompanyStats;
          setStats(data);
        }
      } catch {}
      try {
        const res = await fetch("/api/health");
        if (!cancelled && res.ok) {
          const data = (await res.json()) as Health;
          setHealthOk(data?.status === "ok");
        }
      } catch {
        if (!cancelled) setHealthOk(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [isSignedIn]);

  const formatRefresh = (iso: string | null | undefined): string => {
    if (!iso) return "—";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "—";
    const diffMs = Date.now() - d.getTime();
    const diffMin = Math.floor(diffMs / 60000);
    if (diffMin < 1) return "just now";
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24) return `${diffHr}h ago`;
    const diffDay = Math.floor(diffHr / 24);
    if (diffDay < 30) return `${diffDay}d ago`;
    return d.toISOString().slice(0, 10);
  };

  const fmtNum = (n: number | null | undefined): string =>
    n == null ? "—" : n.toLocaleString();

  const sectorOptions = useMemo<string[]>(() => {
    const seen = new Set<string>();
    for (const c of companies) if (c.sector) seen.add(c.sector);
    return ["All", ...Array.from(seen).sort()];
  }, [companies]);

  const filtered = useMemo<Company[]>(() => {
    const q = query.trim().toLowerCase();
    return companies.filter((c) => {
      if (activeSector !== "All" && c.sector !== activeSector) return false;
      if (!q) return true;
      return (
        c.ticker.toLowerCase().includes(q) ||
        c.name.toLowerCase().includes(q) ||
        (c.sector ?? "").toLowerCase().includes(q)
      );
    });
  }, [companies, query, activeSector]);

  const goToThread = (ticker: string) => {
    router.push(`/?ticker=${encodeURIComponent(ticker)}`);
  };

  if (isLoaded && !isSignedIn) {
    return (
      <div className="app">
        <Sidebar activeRoute="companies" companyCount={0} healthOk={true} />
        <section className="main">
          <div className="universe" style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "60vh" }}>
            <div style={{ textAlign: "center", maxWidth: 480 }}>
              <h1>Sign in to browse the universe</h1>
              <p style={{ marginTop: 12, marginBottom: 24, opacity: 0.8 }}>
                The full list of 50 covered companies is available to signed-in users.
                Anonymous visitors can still chat about Apple, Microsoft, Google, Amazon, and Nvidia.
              </p>
              <SignInButton mode="modal">
                <button className="btn-primary">Sign in</button>
              </SignInButton>
            </div>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="app">
      <Sidebar activeRoute="companies" companyCount={companies.length} healthOk={healthOk} />

      <section className="main">
        <div className="topbar">
          <div className="eyebrow-label">Universe</div>
        </div>

        <div className="universe">
          <div className="universe-head">
            <h1>Universe</h1>
            <p>
              Every claim grounded in these companies&apos; 10-K filings. Click any row to start
              a thread for that company.
            </p>
          </div>

          <div className="stats">
            <Stat label="Issuers" val={fmtNum(stats?.issuers ?? companies.length)} />
            <Stat
              label="Filings indexed"
              val={fmtNum(stats?.filings_indexed)}
              sub="10-K · 10-K/A"
            />
            <Stat label="XBRL facts" val={fmtNum(stats?.xbrl_facts)} />
            <Stat
              label="Coverage"
              val={
                stats
                  ? `${stats.coverage_complete}/${stats.coverage_total}`
                  : "—"
              }
              hideOnMobile
            />
            <Stat label="Last refresh" val={formatRefresh(stats?.last_refresh)} hideOnMobile />
          </div>

          <div className="filter-row">
            <div className="filter-search">
              <span className="search-ic"><IconSearch /></span>
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Filter ticker, name, or sector…"
                aria-label="Filter companies"
              />
            </div>
            {sectorOptions.length > 1 && (
              <div className="sector-pills">
                {sectorOptions.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => setActiveSector(s)}
                    className={`pill ${s === activeSector ? "is-active" : ""}`}
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="table-wrap">
            <table className="universe-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Name</th>
                  <th className="col-hide-mobile">Sector</th>
                  <th className="col-hide-mobile">Latest 10-K</th>
                  <th className="t-right">Filings</th>
                  <th className="col-hide-mobile">Coverage</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {filtered.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="t-empty">
                      {companies.length === 0 ? "Loading…" : "No matches"}
                    </td>
                  </tr>
                ) : (
                  filtered.map((c) => {
                    const period = c.latest_10k_period ? `FY${c.latest_10k_period}` : "—";
                    const filed = c.latest_10k_filed_at ?? "";
                    const filings = c.filings_count != null ? String(c.filings_count) : "—";
                    return (
                      <tr key={c.ticker} onClick={() => goToThread(c.ticker)} className="row">
                        <td><span className="ticker">{c.ticker}</span></td>
                        <td className="t-name">{c.name}</td>
                        <td className="t-sector col-hide-mobile">{c.sector ?? "—"}</td>
                        <td className="t-mono col-hide-mobile">
                          {period}
                          {filed && <span className="t-faint"> · {filed}</span>}
                        </td>
                        <td className="t-right t-mono">{filings}</td>
                        <td className="t-mono col-hide-mobile"><CoverageDot status={c.coverage_status} /></td>
                        <td className="t-action">
                          <span className="ask">Ask <IconArrowRight /></span>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>

          <div className="scope-note">
            <b>Scope is finite by design.</b>{" "}
            The agent only answers about these {companies.length || "—"} issuers, and only from
            10-K filings 2015 onward. Out-of-scope questions are refused with a citation-grounded
            explanation.
          </div>
        </div>
      </section>
    </div>
  );
}
