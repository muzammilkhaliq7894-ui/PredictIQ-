import type { ReactNode } from "react";
import { AlertTriangle, CheckCircle2, LoaderCircle, Radio, ServerCrash } from "lucide-react";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <section className={`motion-card rounded-2xl border border-line bg-panel/80 shadow-glow ${className}`}>{children}</section>;
}

export function SectionTitle({ eyebrow, title, detail }: { eyebrow?: string; title: string; detail?: string }) {
  return (
    <div className="mb-5">
      {eyebrow && <p className="mb-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-cyan">{eyebrow}</p>}
      <h2 className="text-xl font-semibold text-slate-100">{title}</h2>
      {detail && <p className="mt-1 text-sm text-slate-400">{detail}</p>}
    </div>
  );
}

export function StatusPill({ status, label }: { status: "ok" | "warn" | "error" | "loading"; label: string }) {
  const styles = {
    ok: "border-lime/30 bg-lime/10 text-lime",
    warn: "border-amber/30 bg-amber/10 text-amber",
    error: "border-danger/30 bg-danger/10 text-danger",
    loading: "border-cyan/30 bg-cyan/10 text-cyan"
  };
  const Icon = status === "ok" ? CheckCircle2 : status === "error" ? ServerCrash : status === "loading" ? LoaderCircle : AlertTriangle;
  return <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs ${styles[status]}`}><Icon size={13} className={status === "loading" ? "animate-spin" : ""} />{label}</span>;
}

export function ErrorState({ message, onRetry, title = "Unable to load this module" }: { message: string; onRetry?: () => void; title?: string }) {
  return (
    <div className="rounded-xl border border-danger/30 bg-danger/10 p-4 text-sm text-danger">
      <div className="flex items-start gap-3"><AlertTriangle size={18} /><div><p className="font-medium">{title}</p><p className="mt-1 text-danger/80">{message}</p>{onRetry && <button className="mt-3 rounded-lg border border-danger/40 px-3 py-1.5 text-xs hover:bg-danger/10" onClick={onRetry}>Retry</button>}</div></div>
    </div>
  );
}

export function EmptyState({ title, detail }: { title: string; detail: string }) {
  return <div className="flex min-h-40 flex-col items-center justify-center rounded-xl border border-dashed border-line bg-ink/30 px-5 text-center"><Radio className="mb-3 text-slate-500" size={22} /><p className="font-medium text-slate-300">{title}</p><p className="mt-1 max-w-md text-sm text-slate-500">{detail}</p></div>;
}
