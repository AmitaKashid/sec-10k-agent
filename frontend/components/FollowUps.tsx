"use client";

type Props = {
  items: string[];
  onPick: (q: string) => void;
};

function IconLink2({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M9 17H7A5 5 0 0 1 7 7h2" />
      <path d="M15 7h2a5 5 0 1 1 0 10h-2" />
      <line x1="8" y1="12" x2="16" y2="12" />
    </svg>
  );
}

export function FollowUps({ items, onPick }: Props) {
  if (!items.length) return null;
  return (
    <div className="followups">
      <div className="followups-head">
        <IconLink2 />
        <span>Related</span>
      </div>
      <div className="followups-list">
        {items.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => onPick(q)}
            className="followup-item"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
