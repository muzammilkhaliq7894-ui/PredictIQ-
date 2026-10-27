import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import type { LucideIcon } from "lucide-react";
import {
  Activity, AlertOctagon, BarChart3, ChevronRight, CircleGauge, Database,
  Gauge, Home, Menu, RefreshCw, ServerCrash, ShieldCheck, X, ScanSearch
} from "lucide-react";
import { api, API_BASE } from "./api";
import { Card, EmptyState, ErrorState, SectionTitle, StatusPill } from "./components";
import { DatasetSelector, Kpi, PageHeader } from "./dashboard-components";
import { getDatasetSummary } from "./datasets.mjs";
import type { EvaluationResponse, HealthResponse, ModelInfo, Page } from "./types";
import type { Dataset } from "./datasets.mjs";

const AnomalyMonitor = lazy(() => import("./screens").then(module => ({ default: module.AnomalyMonitor })));
const FailureLab = lazy(() => import("./screens").then(module => ({ default: module.FailureLab })));
const Evaluation = lazy(() => import("./screens").then(module => ({ default: module.Evaluation })));
const DatasetExplorer = lazy(() => import("./screens").then(module => ({ default: module.DatasetExplorer })));
const NasaRulLab = lazy(() => import("./screens").then(module => ({ default: module.NasaRulLab })));

const navItems: { id: Page; label: string; icon: LucideIcon; description: string }[] = [
  { id: "overview", label: "Overview", icon: Home, description: "System posture" },
  { id: "explorer", label: "Dataset explorer", icon: ScanSearch, description: "Structure and profiling" },
  { id: "anomaly", label: "Anomaly monitor", icon: Activity, description: "Cycle 001 telemetry" },
  { id: "failure", label: "Prediction lab", icon: Gauge, description: "AI4I scenario" },
  { id: "evaluation", label: "Model evaluation", icon: BarChart3, description: "Offline evidence" },
  { id: "nasa-rul", label: "NASA RUL lab", icon: CircleGauge, description: "C-MAPSS benchmark" }
];

function App() {
  const [page, setPage] = useState<Page>("overview");
  const [dataset, setDataset] = useState<Dataset>("cycle001");
  const [mobileNav, setMobileNav] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);
  const [apiError, setApiError] = useState("");
  const [statusLoading, setStatusLoading] = useState(true);

  const loadStatus = useCallback(async () => {
    setStatusLoading(true);
    setApiError("");
    const [healthResult, modelsResult, evaluationResult] = await Promise.allSettled([
      api.health(),
      api.models(),
      api.evaluation()
    ]);
    const errors: string[] = [];
    if (healthResult.status === "fulfilled") {
      setHealth(healthResult.value);
    } else {
      setHealth(null);
      errors.push(`Health check failed: ${healthResult.reason instanceof Error ? healthResult.reason.message : "the API could not be reached."}`);
    }
    if (modelsResult.status === "fulfilled") {
      setModels(modelsResult.value.models);
    } else {
      setModels([]);
      errors.push(`Model registry unavailable: ${modelsResult.reason instanceof Error ? modelsResult.reason.message : "the API returned no model list."}`);
    }
    if (evaluationResult.status === "fulfilled") {
      setEvaluation(evaluationResult.value);
    } else {
      errors.push(`Saved evaluation unavailable: ${evaluationResult.reason instanceof Error ? evaluationResult.reason.message : "the API returned no saved report."} Showing previously loaded data if available.`);
    }
    setApiError(errors.join(" "));
    setStatusLoading(false);
  }, []);

  useEffect(() => { void loadStatus(); }, [loadStatus]);

  const navigate = (next: Page) => { setPage(next); setMobileNav(false); };
  return (
    <div className="min-h-screen bg-ink text-slate-200">
      <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-line bg-ink/90 px-4 backdrop-blur md:px-8">
        <div className="flex items-center gap-3"><button aria-label="Open navigation" className="rounded-lg p-2 text-slate-400 hover:bg-panel md:hidden" onClick={() => setMobileNav(true)}><Menu size={20} /></button><div className="flex h-9 w-9 items-center justify-center rounded-xl bg-cyan/15 text-cyan"><Activity size={18} /></div><div><p className="text-sm font-semibold tracking-wide text-slate-100">PredictIQ</p><p className="hidden text-[10px] uppercase tracking-[0.14em] text-slate-500 sm:block">AI-Powered Predictive Maintenance</p></div></div>
        <div className="flex items-center gap-3 text-xs text-slate-500"><span className="hidden md:inline">API · {API_BASE}</span>{health ? <StatusPill status="ok" label="Connected" /> : statusLoading ? <StatusPill status="loading" label="Checking" /> : <StatusPill status="error" label="Offline" />}</div>
      </header>
      <div className="mx-auto flex max-w-[1600px]">
        <aside className={`${mobileNav ? "fixed inset-y-0 left-0 z-40 w-72" : "hidden"} overflow-y-auto border-r border-line bg-[#091525] md:sticky md:top-16 md:block md:h-[calc(100vh-4rem)] md:w-64 md:shrink-0`}>
          <div className="flex min-h-full flex-col p-4">
            <div className="mb-6 flex items-center justify-between px-2 md:hidden"><span className="font-semibold">Navigation</span><button aria-label="Close navigation" onClick={() => setMobileNav(false)}><X size={18} /></button></div>
            <div className="mb-7 flex items-center gap-3 rounded-xl border border-cyan/15 bg-cyan/5 p-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-cyan/20 bg-ink/70 text-cyan"><Activity size={20} /></div>
              <div className="min-w-0"><p className="text-base font-semibold tracking-wide text-slate-100">PredictIQ</p><p className="mt-0.5 text-[10px] leading-4 text-slate-500">AI-Powered Predictive Maintenance</p></div>
            </div>
            <p className="mb-3 px-3 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-600">Workspace</p>
            <nav className="space-y-1">{navItems.map(item => { const Icon = item.icon; return <button key={item.id} onClick={() => navigate(item.id)} className={`sidebar-nav-item group flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left transition ${page === item.id ? "bg-cyan/10 text-cyan shadow-[inset_3px_0_0_#51d5e5]" : "text-slate-400 hover:bg-panel hover:text-slate-200"}`}><Icon size={17} /><span><span className="block text-sm font-medium">{item.label}</span><span className="block text-[11px] text-slate-600 group-hover:text-slate-500">{item.description}</span></span></button>; })}</nav>
            <div className="mt-auto pt-6">
              <div className="rounded-xl border border-line bg-ink/60 p-3"><div className="flex items-center gap-2 text-xs text-slate-400"><ShieldCheck size={14} className="text-lime" /> No live machine connection</div><p className="mt-2 text-[11px] leading-relaxed text-slate-600">This interface works with supplied datasets and explicit API responses only.</p></div>
              <div className="mt-3 border-t border-line/80 px-2 pt-3"><p className="text-xs font-medium text-slate-400">Made by Muzammil</p><p className="mt-1 text-[9px] leading-4 tracking-[0.12em] text-slate-600">ALONZA INTERNSHIP PROJECT</p></div>
            </div>
          </div>
        </aside>
        {mobileNav && <button className="fixed inset-0 z-30 bg-black/50 md:hidden" aria-label="Close navigation" onClick={() => setMobileNav(false)} />}
        <main className="min-w-0 flex-1 p-4 md:p-8">
          <div key={page} className="app-page-enter mx-auto max-w-7xl">
            {page === "overview" && <Overview health={health} models={models} evaluation={evaluation} error={apiError} loading={statusLoading} onRetry={() => void loadStatus()} onNavigate={navigate} dataset={dataset} onDatasetChange={setDataset} />}
            {page === "explorer" && <Suspense fallback={<div className="p-6 text-sm text-slate-400">Loading dataset explorer...</div>}><DatasetExplorer dataset={dataset} onDatasetChange={setDataset} /></Suspense>}
            {page === "anomaly" && <Suspense fallback={<div className="p-6 text-sm text-slate-400">Loading analysis module...</div>}><AnomalyMonitor models={models} evaluation={evaluation} dataset={dataset} onDatasetChange={setDataset} /></Suspense>}
            {page === "failure" && <Suspense fallback={<div className="p-6 text-sm text-slate-400">Loading analysis module...</div>}><FailureLab models={models} dataset={dataset} onDatasetChange={setDataset} /></Suspense>}
            {page === "evaluation" && <Suspense fallback={<div className="p-6 text-sm text-slate-400">Loading analysis module...</div>}><Evaluation dataset={dataset} onDatasetChange={setDataset} /></Suspense>}
            {page === "nasa-rul" && <Suspense fallback={<div className="p-6 text-sm text-slate-400">Loading NASA RUL module...</div>}><NasaRulLab dataset={dataset} onDatasetChange={setDataset} /></Suspense>}
          </div>
        </main>
      </div>
    </div>
  );
}

function Overview({ health, models, evaluation, error, loading, onRetry, onNavigate, dataset, onDatasetChange }: { health: HealthResponse | null; models: ModelInfo[]; evaluation: EvaluationResponse | null; error: string; loading: boolean; onRetry: () => void; onNavigate: (page: Page) => void; dataset: Dataset; onDatasetChange: (dataset: Dataset) => void }) {
  const cycle = evaluation && getDatasetSummary(evaluation, "cycle001");
  const ai4i = evaluation && getDatasetSummary(evaluation, "ai4i2020");
  const isNasa = dataset.startsWith("nasa_");
  const selectedDataset = dataset === "cycle001" ? cycle : dataset === "ai4i2020" ? ai4i : null;
  const datasetLabel = isNasa ? `NASA C-MAPSS ${dataset.slice(-5).toUpperCase()}` : dataset === "cycle001" ? "Cycle 001" : "AI4I 2020";
  return <><PageHeader eyebrow="Operations / overview" title="System posture" detail="A grounded view of the available models, datasets, and API readiness. NASA C-MAPSS has separate exploration and RUL benchmark workflows." action={<div className="flex flex-wrap items-center gap-2"><DatasetSelector value={dataset} onChange={onDatasetChange} /><button disabled={loading} onClick={onRetry} className="inline-flex items-center gap-2 self-start rounded-xl border border-line bg-panel px-3 py-2 text-xs text-slate-300 hover:border-cyan/40 disabled:cursor-wait disabled:opacity-50">{loading && <RefreshCw size={14} className="animate-spin" />}{loading ? "Refreshing..." : "Refresh Status"}</button></div>} />
    {error && <div className="mb-5"><ErrorState title="Unable to refresh system status" message={error} onRetry={onRetry} /></div>}
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <Kpi icon={ServerCrash} label="API health" value={health?.status ?? (loading ? "Checking…" : "Unavailable")} detail={health ? `Version ${health.model_version}` : loading ? "Contacting FastAPI" : "Start FastAPI to connect"} status={health ? "ok" : loading ? "neutral" : "error"} />
      <Kpi icon={Database} label={`${datasetLabel} rows`} value={selectedDataset ? selectedDataset.rows.toLocaleString() : "—"} detail={isNasa ? "See NASA dataset explorer for train/test dimensions" : evaluation ? "Saved dataset report" : loading ? "Loading saved evaluation..." : "Saved report unavailable"} status="neutral" />
      {dataset === "cycle001" ? <>
        <Kpi icon={AlertOctagon} label="Offline anomaly alerts" value={cycle ? cycle.alerts.toString() : "—"} detail={cycle ? `${(cycle.alertRate * 100).toFixed(2)}% flagged · offline baseline` : "Saved report unavailable"} status="warn" />
        <Kpi icon={CircleGauge} label="Scored features" value={cycle ? cycle.features.toString() : "—"} detail="Cycle 001 anomaly model schema" status="neutral" />
      </> : dataset === "ai4i2020" ? <>
        <Kpi icon={AlertOctagon} label="AI4I failure rows" value={ai4i ? ai4i.failures.toLocaleString() : "—"} detail="Dataset labels, not live outcomes" status="warn" />
        <Kpi icon={CircleGauge} label="AI4I positive rate" value={ai4i ? `${(ai4i.positiveRate * 100).toFixed(2)}%` : "—"} detail={ai4i ? `From ${ai4i.rows.toLocaleString()} labeled rows` : "Saved report unavailable"} status="neutral" />
      </> : <>
        <Kpi icon={ScanSearch} label="NASA C-MAPSS" value="4 subsets" detail="FD001–FD004; separate training and test files" status="neutral" />
        <Kpi icon={CircleGauge} label="RUL workflow" value="Available" detail="Engine-level validation and official test metrics" status="ok" />
      </>}
    </div>
    <div className="mt-6 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
      <Card className="p-6"><SectionTitle eyebrow="Available modules" title="Choose an analysis path" detail="Each module keeps offline evidence distinct from a live API prediction." /><div className="grid gap-3 sm:grid-cols-2"><ModuleCard icon={Activity} title="Machine anomaly monitor" detail="Score built-in Cycle 001 telemetry with its compatible anomaly model." onClick={() => onNavigate("anomaly")} /><ModuleCard icon={Gauge} title="Failure prediction lab" detail="Submit a valid AI4I-style feature set to the trained classifier." onClick={() => onNavigate("failure")} /><ModuleCard icon={ScanSearch} title="NASA dataset explorer" detail="Inspect C-MAPSS engines, cycle histories, sensor trends, and summaries." onClick={() => onNavigate("explorer")} /><ModuleCard icon={CircleGauge} title="NASA RUL benchmark" detail="Evaluate and query the separate C-MAPSS remaining-life regression baseline." onClick={() => onNavigate("nasa-rul")} /></div></Card>
      <Card className="p-6"><SectionTitle eyebrow="Model registry" title="Loaded artifacts" detail="Reported by the FastAPI /models endpoint." />{models.length ? <div className="space-y-3">{models.map(model => <div key={model.name} className="rounded-xl border border-line bg-ink/40 p-3"><div className="flex items-center justify-between gap-3"><span className="truncate text-sm font-medium text-slate-200">{model.name}</span><StatusPill status="ok" label={model.version} /></div><p className="mt-1 text-xs text-slate-500">{model.type} · {model.feature_names.length} features</p></div>)}</div> : <EmptyState title="No models loaded" detail="Train Phase 3 artifacts and start the API to populate the registry." />}</Card>
    </div>
  </>;
}


function ModuleCard({ icon: Icon, title, detail, onClick }: { icon: LucideIcon; title: string; detail: string; onClick: () => void }) {
  return <button onClick={onClick} className="group rounded-xl border border-line bg-ink/40 p-4 text-left transition hover:border-cyan/40 hover:bg-cyan/5"><div className="mb-5 flex h-9 w-9 items-center justify-center rounded-lg bg-cyan/10 text-cyan"><Icon size={18} /></div><div className="flex items-center justify-between gap-2"><p className="font-medium text-slate-200">{title}</p><ChevronRight size={16} className="text-slate-600 transition group-hover:translate-x-1 group-hover:text-cyan" /></div><p className="mt-2 text-xs leading-5 text-slate-500">{detail}</p></button>;
}

export default App;
