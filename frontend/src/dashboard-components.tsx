import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import { Card } from "./components";
import { DATASETS } from "./datasets.mjs";
import type { Dataset } from "./datasets.mjs";

export function DatasetSelector({ value, onChange, disabled = false }: { value: Dataset; onChange: (dataset: Dataset) => void; disabled?: boolean }) {
  return (
    <label className="flex items-center gap-2 text-xs text-slate-400">
      <span>Dataset</span>
      <select
        aria-label="Dataset"
        disabled={disabled}
        value={value}
        onChange={event => {
          const selected = Object.values(DATASETS).find(dataset => dataset.id === event.target.value);
          if (selected) onChange(selected.id);
        }}
        className="rounded-lg border border-line bg-ink px-3 py-2 text-xs text-slate-200 disabled:cursor-wait disabled:opacity-50"
      >
        <optgroup label="Existing datasets">
          <option value={DATASETS.cycle001.id}>{DATASETS.cycle001.label}</option>
          <option value={DATASETS.ai4i2020.id}>{DATASETS.ai4i2020.label}</option>
        </optgroup>
        <optgroup label="NASA C-MAPSS">
          <option value={DATASETS.nasa_fd001.id}>{DATASETS.nasa_fd001.label}</option>
          <option value={DATASETS.nasa_fd002.id}>{DATASETS.nasa_fd002.label}</option>
          <option value={DATASETS.nasa_fd003.id}>{DATASETS.nasa_fd003.label}</option>
          <option value={DATASETS.nasa_fd004.id}>{DATASETS.nasa_fd004.label}</option>
        </optgroup>
      </select>
    </label>
  );
}

export function PageHeader({ eyebrow, title, detail, action }: { eyebrow: string; title: string; detail: string; action?: ReactNode }) {
  return <div className="mb-8 flex flex-col justify-between gap-4 lg:flex-row lg:items-end"><div><p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-cyan">{eyebrow}</p><h1 className="text-3xl font-semibold tracking-tight text-slate-100 md:text-4xl">{title}</h1><p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">{detail}</p></div>{action}</div>;
}

export function Kpi({ icon: Icon, label, value, detail, status }: { icon: LucideIcon; label: string; value: string; detail: string; status: "ok" | "error" | "warn" | "neutral" }) {
  return <Card className="p-5"><div className="flex items-center justify-between"><span className="text-xs uppercase tracking-[0.15em] text-slate-500">{label}</span><Icon size={17} className={status === "ok" ? "text-lime" : status === "warn" ? "text-amber" : status === "error" ? "text-danger" : "text-cyan"} /></div><p className="kpi-value mt-4 text-2xl font-semibold text-slate-100">{value}</p><p className="mt-1 text-xs text-slate-500">{detail}</p></Card>;
}
