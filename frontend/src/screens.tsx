import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { AlertOctagon, CircleGauge, Database, Microscope, RefreshCw, ScanSearch } from "lucide-react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis
} from "recharts";
import { api } from "./api";
import { Card, EmptyState, ErrorState, SectionTitle, StatusPill } from "./components";
import { DatasetSelector, Kpi, PageHeader } from "./dashboard-components";
import { getDatasetSummary, supportsDataset } from "./datasets.mjs";
import type { Dataset } from "./datasets.mjs";
import type {
  AnomalyPoint,
  DatasetAnalysis,
  DatasetProfile,
  EvaluationResponse,
  ModelComparisonsResponse,
  ModelInfo,
  NasaDatasetOverview,
  NasaEngineHistory,
  NasaRulEvaluation,
  NasaRulPrediction,
  TelemetryRow
} from "./types";

export function DatasetExplorer(props: { dataset: Dataset; onDatasetChange: (dataset: Dataset) => void }) {
  return props.dataset.startsWith("nasa_")
    ? <NasaExplorer dataset={props.dataset} onDatasetChange={props.onDatasetChange} />
    : <CsvDatasetExplorer {...props} />;
}

function CsvDatasetExplorer({ dataset, onDatasetChange }: { dataset: Dataset; onDatasetChange: (dataset: Dataset) => void }) {
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [analysisResult, setAnalysisResult] = useState<{ dataset: Dataset; data: DatasetAnalysis } | null>(null);
  const [loading, setLoading] = useState(true);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [error, setError] = useState("");
  const [analysisError, setAnalysisError] = useState("");
  const [selectedNumeric, setSelectedNumeric] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("");

  useEffect(() => {
    let active = true;
    setProfile(null);
    setAnalysisResult(null);
    setError("");
    setAnalysisError("");
    setLoading(true);
    api.datasetProfile(dataset).then(result => {
      if (active) setProfile(result);
    }).catch(cause => {
      if (active) setError(cause instanceof Error ? cause.message : "The dataset profile could not be loaded.");
    }).finally(() => {
      if (active) setLoading(false);
    });
    return () => { active = false; };
  }, [dataset]);

  const runAnalysis = async () => {
    if (!currentProfile || analysisLoading) return;
    setAnalysisLoading(true);
    setAnalysisError("");
    try {
      setAnalysisResult({ dataset, data: await api.analyzeDataset(dataset) });
    } catch (cause) {
      setAnalysisError(cause instanceof Error ? cause.message : "Dataset profiling failed.");
    } finally {
      setAnalysisLoading(false);
    }
  };

  const currentProfile = profile?.id === dataset ? profile : null;
  const analysis = analysisResult?.dataset === dataset ? analysisResult.data : null;
  const numeric = analysis?.numeric_distributions ?? [];
  const categories = analysis?.categorical_counts.filter(item => item.values.length) ?? [];
  const numericColumn = numeric.find(item => item.column === selectedNumeric) ?? numeric[0];
  const categoryColumn = categories.find(item => item.column === selectedCategory) ?? categories[0];
  const tableValue = (value: string | number | boolean | null) => value === null ? "—" : String(value);

  return (
    <>
      <PageHeader
        eyebrow="Data / exploration"
        title="Dataset explorer"
        detail="Inspect the supplied Cycle 001, AI4I 2020, or separate NASA C-MAPSS benchmark files and descriptive profiles."
        action={<div className="flex flex-wrap items-center gap-2"><DatasetSelector value={dataset} onChange={onDatasetChange} disabled={loading || analysisLoading} /><button type="button" disabled={!currentProfile || loading || analysisLoading} onClick={() => void runAnalysis()} className="inline-flex items-center gap-2 rounded-xl bg-cyan px-4 py-2.5 text-sm font-semibold text-ink hover:bg-cyan/90 disabled:cursor-wait disabled:opacity-50">{(loading || analysisLoading) && <RefreshCw size={15} className="animate-spin" />}{analysisLoading ? "Profiling dataset..." : analysis ? "Refresh analysis" : "Analyze Dataset"}</button></div>}
      />
      <div className="mb-5 rounded-xl border border-cyan/20 bg-cyan/5 p-4 text-sm leading-6 text-slate-300">
        <ScanSearch size={16} className="mr-2 inline text-cyan" />
        Dataset exploration is descriptive only. It does not run anomaly detection, failure prediction, or the saved offline model evaluation.
      </div>
      {loading && <div className="mb-5" role="status"><StatusPill status="loading" label="Reading dataset structure..." /></div>}
      {error && <div className="mb-5" role="alert"><ErrorState title={error.includes("Unsupported dataset") ? "Invalid dataset" : "Dataset unavailable"} message={error} /></div>}
      {analysisError && <div className="mb-5" role="alert"><ErrorState title="Dataset analysis failed" message={analysisError} onRetry={() => void runAnalysis()} /></div>}
      {currentProfile && !loading && <div className="mb-5" role="status"><StatusPill status="ok" label={analysis ? `Profile complete · ${analysis.sample_size.toLocaleString()} chart sample rows` : "Dataset profile loaded"} /></div>}
      {!loading && !error && !currentProfile && <EmptyState title="No dataset profile available" detail="Choose a supported dataset and retry the request." />}
      {currentProfile && (
        <>
          <div className="mb-5 grid gap-4 sm:grid-cols-3">
            <Kpi icon={Database} label="Dataset" value={currentProfile.name} detail="Supplied local dataset" status="neutral" />
            <Kpi icon={Database} label="Rows" value={currentProfile.row_count.toLocaleString()} detail="Counted from the source file" status="neutral" />
            <Kpi icon={ScanSearch} label="Columns" value={currentProfile.column_count.toLocaleString()} detail="Empty export-only columns excluded" status="neutral" />
          </div>
          {currentProfile.row_count === 0 && <div className="mb-5"><EmptyState title="Dataset is empty" detail="The source contains no data rows to preview or analyze." /></div>}
          <Card className="mb-5 overflow-hidden p-5">
            <SectionTitle eyebrow="Schema" title="Columns and missing values" detail="Missing-value counts are scanned across the source dataset; dtypes are inferred by the CSV reader." />
            <div className="max-h-[28rem] overflow-auto rounded-lg border border-line">
              <table className="w-full min-w-[36rem] text-left text-xs">
                <thead className="sticky top-0 bg-ink text-slate-500"><tr><th className="p-3">Column</th><th className="p-3">Data type</th><th className="p-3">Missing</th><th className="p-3">Missing %</th></tr></thead>
                <tbody>{currentProfile.columns.map(column => <tr key={column.name} className="border-t border-line"><td className="p-3 text-slate-200">{column.name}</td><td className="p-3 text-cyan">{column.dtype}</td><td className="p-3 text-slate-300">{column.missing_count.toLocaleString()}</td><td className="p-3 text-slate-400">{column.missing_percent.toFixed(2)}%</td></tr>)}</tbody>
              </table>
            </div>
          </Card>
          <Card className="mb-5 overflow-hidden p-5">
            <SectionTitle eyebrow="Source preview" title={`First ${currentProfile.preview_rows} rows`} detail="Only a bounded preview is sent to the browser. The full dataset remains on the API server." />
            {currentProfile.preview.length ? (
              <div className="max-h-[28rem] overflow-auto rounded-lg border border-line">
                <table className="w-full min-w-max text-left text-xs">
                  <thead className="sticky top-0 bg-ink text-slate-500"><tr>{currentProfile.columns.map(column => <th key={column.name} className="whitespace-nowrap p-3">{column.name}</th>)}</tr></thead>
                  <tbody>{currentProfile.preview.map((row, index) => <tr key={index} className="border-t border-line">{currentProfile.columns.map(column => <td key={column.name} className="max-w-64 whitespace-nowrap p-3 text-slate-300">{tableValue(row[column.name] ?? null)}</td>)}</tr>)}</tbody>
                </table>
              </div>
            ) : <EmptyState title="No preview rows" detail="The dataset has no rows to preview." />}
          </Card>
          {analysis && (
            <>
              <Card className="mb-5 overflow-hidden p-5">
                <SectionTitle eyebrow="Dataset profile" title="Numeric summary statistics" detail={`Exact count, mean, standard deviation, minimum, and maximum are computed over ${analysis.row_count.toLocaleString()} rows.`} />
                {analysis.statistics.length ? (
                  <div className="max-h-[28rem] overflow-auto rounded-lg border border-line">
                    <table className="w-full min-w-[42rem] text-left text-xs">
                      <thead className="sticky top-0 bg-ink text-slate-500"><tr><th className="p-3">Column</th><th className="p-3">Count</th><th className="p-3">Mean</th><th className="p-3">Std. dev.</th><th className="p-3">Min</th><th className="p-3">Max</th></tr></thead>
                      <tbody>{analysis.statistics.map(item => <tr key={item.column} className="border-t border-line"><td className="p-3 text-slate-200">{item.column}</td><td className="p-3 text-slate-300">{item.count.toLocaleString()}</td><td className="p-3 text-slate-300">{item.mean.toPrecision(5)}</td><td className="p-3 text-slate-300">{item.std.toPrecision(5)}</td><td className="p-3 text-slate-300">{item.min.toPrecision(5)}</td><td className="p-3 text-slate-300">{item.max.toPrecision(5)}</td></tr>)}</tbody>
                    </table>
                  </div>
                ) : <EmptyState title="No numeric columns" detail="This dataset has no numeric columns to summarize." />}
              </Card>
              <div className="mb-2 grid gap-5 xl:grid-cols-2">
                <Card className="p-5">
                  <SectionTitle eyebrow="Sampled visualization" title="Numeric distributions" detail={`Histogram from ${analysis.sample_size.toLocaleString()} evenly spaced sample rows.`} />
                  {numericColumn ? <>
                    <label className="mb-4 block text-xs text-slate-400">Numeric column<select aria-label="Numeric distribution column" className="mt-2 block max-w-full rounded-lg border border-line bg-ink px-3 py-2 text-xs text-slate-200" value={numericColumn.column} onChange={event => setSelectedNumeric(event.target.value)}>{numeric.map(item => <option key={item.column} value={item.column}>{item.column}</option>)}</select></label>
                    <div className="h-72"><ResponsiveContainer width="100%" height="100%"><BarChart data={numericColumn.bins} className="chart-grid"><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="range" interval="preserveStartEnd" /><YAxis allowDecimals={false} /><Tooltip contentStyle={{ background: "#0b1929", border: "1px solid #1d3148", borderRadius: 10 }} /><Bar dataKey="count" fill="#51d5e5" name="Sample rows" /></BarChart></ResponsiveContainer></div>
                  </> : <EmptyState title="No numeric distribution available" detail="No numeric column contains values to chart." />}
                </Card>
                <Card className="p-5">
                  <SectionTitle eyebrow="Sampled visualization" title="Categorical values" detail="Top observed values in the same bounded sample; counts are sample counts, not full-dataset totals." />
                  {categoryColumn ? <>
                    <label className="mb-4 block text-xs text-slate-400">Category column<select aria-label="Category count column" className="mt-2 block max-w-full rounded-lg border border-line bg-ink px-3 py-2 text-xs text-slate-200" value={categoryColumn.column} onChange={event => setSelectedCategory(event.target.value)}>{categories.map(item => <option key={item.column} value={item.column}>{item.column}</option>)}</select></label>
                    <div className="h-72"><ResponsiveContainer width="100%" height="100%"><BarChart data={categoryColumn.values} className="chart-grid"><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="value" /><YAxis allowDecimals={false} /><Tooltip contentStyle={{ background: "#0b1929", border: "1px solid #1d3148", borderRadius: 10 }} /><Bar dataKey="count" fill="#a3e635" name="Sample rows" /></BarChart></ResponsiveContainer></div>
                  </> : <EmptyState title="No categorical chart available" detail="No categorical columns with observed values were found in the sample." />}
                </Card>
              </div>
            </>
          )}
        </>
      )}
    </>
  );
}

export function AnomalyMonitor({ models, evaluation, dataset, onDatasetChange }: { models: ModelInfo[]; evaluation: EvaluationResponse | null; dataset: Dataset; onDatasetChange: (dataset: Dataset) => void }) {
  const model = models.find(item => item.name === "cycle001_anomaly");
  const [rows, setRows] = useState<TelemetryRow[]>([]);
  const [selectedFeature, setSelectedFeature] = useState("");
  const [points, setPoints] = useState<AnomalyPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [scoring, setScoring] = useState(false);
  const [telemetryReload, setTelemetryReload] = useState(0);
  const [progress, setProgress] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const scoringLock = useRef(false);
  const featureNames = model?.feature_names ?? [];
  const canScoreDataset = supportsDataset(dataset, "anomaly");
  const visiblePoints = canScoreDataset ? points : [];

  useEffect(() => {
    if (!selectedFeature && featureNames[0]) setSelectedFeature(featureNames[0]);
  }, [featureNames, selectedFeature]);

  useEffect(() => {
    setRows([]);
    setPoints([]);
    setProgress("");
    setMessage("");
    setError("");
    if (!supportsDataset(dataset, "anomaly")) {
      setLoading(false);
      return;
    }
    let active = true;
    setLoading(true);
    api.cycleTelemetry().then(response => {
      if (!active) return;
      setRows(response.rows);
      setMessage(`Loaded ${response.rows.length} built-in Cycle 001 telemetry rows.`);
    }).catch(cause => {
      if (active) setError(cause instanceof Error ? cause.message : "Built-in Cycle 001 telemetry could not be loaded.");
    }).finally(() => {
      if (active) setLoading(false);
    });
    return () => { active = false; };
  }, [dataset, telemetryReload]);

  const scoreRows = async () => {
    if (!canScoreDataset || !rows.length || scoringLock.current) return;
    scoringLock.current = true;
    setScoring(true);
    setError("");
    setMessage("");
    setPoints([]);
    try {
      const scored: AnomalyPoint[] = [];
      for (const [index, row] of rows.entries()) {
        setProgress(`Scoring row ${index + 1} of ${rows.length}...`);
        const result = await api.scoreAnomaly(row.values);
        scored.push({ timestamp: row.timestamp, score: result.anomaly_score, alert: result.alert });
      }
      setPoints(scored);
      const flagged = scored.filter(point => point.alert).length;
      setMessage(`Scored ${scored.length} Cycle 001 row(s) with model ${model?.version ?? "reported by the API"}; ${flagged} flagged. Scores are not failure probabilities.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Telemetry scoring failed. No complete result set was returned.");
    } finally {
      scoringLock.current = false;
      setProgress("");
      setScoring(false);
    }
  };

  const trend = rows.map((row, index) => ({ index: index + 1, value: row.values[selectedFeature] }));
  const visibleError = canScoreDataset ? error : "";
  return (
    <>
      <PageHeader
        eyebrow="Analysis / Cycle 001"
        title="Machine anomaly monitor"
        detail="Score a bounded sample from the built-in Cycle 001 dataset using the existing unsupervised anomaly endpoint. No live machine connection is assumed."
        action={<DatasetSelector value={dataset} onChange={onDatasetChange} disabled={loading || scoring} />}
      />
      <Card className="mb-5 p-5">
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            disabled={!canScoreDataset || !rows.length || loading || scoring}
            onClick={() => void scoreRows()}
            className="inline-flex items-center gap-2 rounded-xl border border-cyan/40 px-4 py-2.5 text-sm font-semibold text-cyan disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading && <RefreshCw size={15} className="animate-spin" />}
            {loading ? "Loading telemetry..." : scoring ? "Scoring telemetry..." : "Analyze/Score Telemetry"}
          </button>
          {loading && <span className="text-xs text-cyan" role="status">Loading built-in dataset...</span>}
          {scoring && <span className="text-xs text-cyan" role="status">{progress}</span>}
          {!rows.length && !loading && canScoreDataset && <p className="text-xs text-slate-500">No built-in Cycle 001 telemetry rows are available.</p>}
        </div>
        {!canScoreDataset && <p className="mt-3 text-sm text-amber">Anomaly detection supports Cycle 001 only. AI4I 2020 supports its own failure model; NASA C-MAPSS supports its separate RUL workflow. Neither dataset is passed to the Cycle 001 model.</p>}
      </Card>
      <div className="mb-5 grid gap-4 md:grid-cols-3">
        <Kpi icon={Database} label="Dataset" value={dataset === "cycle001" ? "Cycle 001" : dataset === "ai4i2020" ? "AI4I 2020" : `NASA C-MAPSS ${dataset.slice(-5).toUpperCase()}`} detail={canScoreDataset ? "Built-in telemetry sample" : "Unsupported by this anomaly model"} status="neutral" />
        <Kpi icon={AlertOctagon} label="Offline reference alerts" value={evaluation ? evaluation.cycle_summary.alerts.toString() : "—"} detail="Saved Phase 2 result, not live" status="warn" />
        <Kpi icon={Database} label="Sample rows" value={(canScoreDataset ? rows.length : 0).toString()} detail="Maximum 20 built-in rows per analysis" status="neutral" />
      </div>
      {visibleError && <div className="mb-5" role="alert"><ErrorState title="Telemetry analysis unavailable" message={visibleError} onRetry={() => setTelemetryReload(value => value + 1)} /></div>}
      <div className="grid gap-6 xl:grid-cols-[1.5fr_1fr]">
        <Card className="p-6">
          <SectionTitle eyebrow="Built-in telemetry sample" title="Sensor trend" detail="Only supplied Cycle 001 readings are plotted. No values are inferred or filled in." />
          {canScoreDataset && rows.length ? (
            <>
              <label className="mb-4 flex flex-wrap items-center gap-3 text-xs text-slate-400">
                Sensor feature
                <select aria-label="Sensor feature" className="max-w-full rounded-lg border border-line bg-ink px-3 py-2 text-xs text-slate-300" value={selectedFeature} onChange={event => setSelectedFeature(event.target.value)}>
                  {featureNames.map(name => <option key={name} value={name}>{name}</option>)}
                </select>
              </label>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trend} className="chart-grid">
                    <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="index" /><YAxis /><Tooltip contentStyle={{ background: "#0d1a2b", border: "1px solid #1d3148", borderRadius: 8 }} />
                    <Line type="monotone" dataKey="value" stroke="#51d5e5" dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </>
          ) : <EmptyState title={canScoreDataset ? "No telemetry available" : "Unsupported dataset selected"} detail={canScoreDataset ? "Load the built-in Cycle 001 telemetry sample to display the sensor trend." : "The selected dataset is not compatible with the Cycle 001 anomaly model. Use Dataset Explorer, the AI4I Failure Prediction Lab, or the NASA RUL Lab."} />}
        </Card>
        <Card className="p-6">
          <SectionTitle eyebrow="Live API results" title="Anomaly scores" detail="Returned by POST /anomaly/score; these values are not failure probabilities." />
          {visiblePoints.length ? (
            <>
              <div className="mb-4 h-40">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={visiblePoints.map((point, index) => ({ ...point, index: index + 1 }))} className="chart-grid">
                    <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="index" /><YAxis /><Tooltip contentStyle={{ background: "#0d1a2b", border: "1px solid #1d3148", borderRadius: 8 }} />
                    <Area type="monotone" dataKey="score" stroke="#f7ba61" fill="#f7ba61" fillOpacity={0.12} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="max-h-44 overflow-auto rounded-lg border border-line">
                <table className="w-full text-left text-xs">
                  <thead className="sticky top-0 bg-ink text-slate-500"><tr><th className="p-2">Timestamp</th><th className="p-2">Score</th><th className="p-2">Alert</th></tr></thead>
                  <tbody>{visiblePoints.map((point, index) => <tr key={`${point.timestamp}-${index}`} className="border-t border-line"><td className="p-2 text-slate-400">{point.timestamp}</td><td className="p-2 text-slate-200">{point.score.toFixed(5)}</td><td className={`p-2 ${point.alert ? "text-amber" : "text-slate-500"}`}>{point.alert ? "Flagged" : "Normal"}</td></tr>)}</tbody>
                </table>
              </div>
            </>
          ) : <EmptyState title="Waiting for a score" detail="Select Cycle 001 and run Analyze/Score Telemetry on its built-in sample to see API results." />}
          {canScoreDataset && message && <p className="mt-4 text-xs text-lime" role="status">{message}</p>}
          <div className="mt-5 rounded-xl border border-amber/20 bg-amber/5 p-3 text-xs leading-5 text-amber/80"><AlertOctagon size={15} className="mb-1" />An anomaly alert is an unsupervised signal. It is not a confirmed failure, diagnosis, or advance warning.</div>
        </Card>
      </div>
      <Card className="mt-6 p-6">
        <SectionTitle eyebrow="Saved Phase 2 report · offline" title="Flagged Cycle 001 timestamps" detail="Saved baseline alerts are separate from current API scores. Alerts are unsupervised and are not confirmed failures." />
        {evaluation?.cycle_flagged.length ? (
          <div className="overflow-auto rounded-lg border border-line">
            <table className="w-full text-left text-xs">
              <thead className="bg-ink text-slate-500"><tr><th className="p-3">Timestamp</th><th className="p-3">Machine ID</th><th className="p-3">Anomaly score</th></tr></thead>
              <tbody>{evaluation.cycle_flagged.slice(0, 8).map(row => <tr key={`${row.timestamp}-${row.machine_id}`} className="border-t border-line"><td className="p-3 text-slate-300">{row.timestamp}</td><td className="p-3 text-slate-400">{row.machine_id}</td><td className="p-3 text-amber">{row.anomaly_score.toFixed(5)}</td></tr>)}</tbody>
            </table>
          </div>
        ) : <EmptyState title="No saved alert report available" detail="Run the Phase 2 baseline command to produce Cycle 001 flagged timestamps." />}
      </Card>
    </>
  );
}

const failureFields = [
  ["air_temperature", "Air temperature [K]"],
  ["process_temperature", "Process temperature [K]"],
  ["rotational_speed", "Rotational speed [rpm]"],
  ["torque", "Torque [Nm]"],
  ["tool_wear", "Tool wear [min]"]
] as const;

export function FailureLab({ models, dataset, onDatasetChange }: { models: ModelInfo[]; dataset: Dataset; onDatasetChange: (dataset: Dataset) => void }) {
  const [values, setValues] = useState({ Type: "L", air_temperature: "", process_temperature: "", rotational_speed: "", torque: "", tool_wear: "" });
  const [result, setResult] = useState<Awaited<ReturnType<typeof api.predictFailure>> | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const requestLock = useRef(false);
  const model = models.find(item => item.name === "ai4i_failure");
  const canPredictDataset = supportsDataset(dataset, "failure");

  const update = (key: keyof typeof values, value: string) => {
    setValues(current => ({ ...current, [key]: value }));
    setResult(null);
    setError("");
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!canPredictDataset || requestLock.current) return;
    setError("");
    const rawNumbers = failureFields.map(([key]) => values[key].trim());
    if (rawNumbers.some(value => value.length === 0)) {
      setError("Enter a value for every AI4I predictor before submitting.");
      return;
    }
    const numbers = {
      air_temperature: Number(values.air_temperature),
      process_temperature: Number(values.process_temperature),
      rotational_speed: Number(values.rotational_speed),
      torque: Number(values.torque),
      tool_wear: Number(values.tool_wear)
    };
    if (Object.values(numbers).some(value => !Number.isFinite(value))) {
      setError("Each AI4I predictor must be a finite number.");
      return;
    }

    requestLock.current = true;
    setLoading(true);
    setResult(null);
    try {
      setResult(await api.predictFailure({ Type: values.Type, ...numbers }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Prediction failed. The API returned no result.");
    } finally {
      requestLock.current = false;
      setLoading(false);
    }
  };

  return (
    <>
      <PageHeader
        eyebrow="Analysis / AI4I 2020"
        title="Failure prediction lab"
        detail="Submit five explicit AI4I-style predictors to the existing classifier. This static benchmark is not a live machine feed."
        action={<DatasetSelector value={dataset} disabled={loading} onChange={next => { onDatasetChange(next); setResult(null); setError(""); }} />}
      />
      {!canPredictDataset && <div className="mb-5 rounded-xl border border-amber/20 bg-amber/5 p-4 text-sm leading-6 text-amber" role="status">The failure classifier is trained for AI4I 2020 inputs only. Switch the dataset selector to AI4I 2020; neither Cycle 001 telemetry nor NASA C-MAPSS data is compatible with this model.</div>}
      <div className="grid gap-6 xl:grid-cols-[1fr_1.15fr]">
        <Card className="p-6">
          <SectionTitle eyebrow="Dataset · AI4I 2020" title="Predictor inputs" detail="All five numeric predictors and a product type are required. Values are sent unchanged to POST /failure/predict." />
          <form onSubmit={submit} className="space-y-4">
            <label className="block text-sm text-slate-300">
              Product type
              <select aria-label="Product type" disabled={!canPredictDataset || loading} value={values.Type} onChange={event => update("Type", event.target.value)} className="mt-2 w-full rounded-lg border border-line bg-ink px-3 py-2.5 text-sm text-slate-200 disabled:opacity-50">
                <option value="L">L · Low</option><option value="M">M · Medium</option><option value="H">H · High</option>
              </select>
            </label>
            <div className="grid gap-4 sm:grid-cols-2">
              {failureFields.map(([key, label]) => (
                <label key={key} className="block text-sm text-slate-300">
                  {label}
                  <input aria-label={label} disabled={!canPredictDataset || loading} type="number" step="any" value={values[key]} onChange={event => update(key, event.target.value)} className="mt-2 w-full rounded-lg border border-line bg-ink px-3 py-2.5 text-sm text-slate-200 outline-none focus:border-cyan disabled:opacity-50" placeholder="Required" />
                </label>
              ))}
            </div>
            {error && <p className="rounded-lg border border-danger/30 bg-danger/10 p-3 text-sm text-danger" role="alert">{error}</p>}
            <button disabled={!canPredictDataset || loading} className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-cyan px-4 py-3 text-sm font-semibold text-ink hover:bg-cyan/90 disabled:cursor-not-allowed disabled:opacity-50">
              {loading && <RefreshCw size={15} className="animate-spin" />}
              {loading ? "Predicting..." : "Predict Failure"}
            </button>
          </form>
        </Card>
        <Card className="p-6">
          <SectionTitle eyebrow="Actual API response" title="Failure prediction" detail={model ? `Model ${model.version} · ${model.type}` : "Model metadata is unavailable"} />
          {!canPredictDataset ? <EmptyState title="AI4I 2020 required" detail="Select AI4I 2020 to enable the classifier and submit predictors." /> : result ? (
            <div aria-live="polite">
              <div className="rounded-2xl border border-cyan/20 bg-cyan/5 p-6 text-center">
                <p className="text-sm text-slate-400">API-returned failure probability</p>
                <p className="mt-2 text-6xl font-semibold text-cyan">{(result.failure_probability * 100).toFixed(1)}<span className="text-2xl">%</span></p>
                <p className={`mt-3 text-sm font-medium ${result.predicted_failure ? "text-danger" : "text-lime"}`}>{result.predicted_failure ? "Predicted failure class" : "Predicted non-failure class"}</p>
              </div>
              <div className="mt-5 rounded-xl border border-line bg-ink/40 p-4 text-sm leading-6 text-slate-400"><p className="mb-2 font-medium text-slate-200">Model limitations</p>{result.limitations}</div>
              <p className="mt-4 text-xs text-slate-500">Model version: {result.model_version}. This result is separate from Cycle 001 anomaly scores.</p>
            </div>
          ) : <EmptyState title="No prediction yet" detail="Submit a complete AI4I 2020 input row. No readings or probability will be generated without an API response." />}
          {loading && <div className="mt-4" role="status"><StatusPill status="loading" label="Waiting for the AI4I prediction API..." /></div>}
        </Card>
      </div>
    </>
  );
}

export function Evaluation({ dataset, onDatasetChange }: { dataset: Dataset; onDatasetChange: (dataset: Dataset) => void }) {
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);
  const [comparison, setComparison] = useState<ModelComparisonsResponse | null>(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonSeconds, setComparisonSeconds] = useState(0);
  const [comparisonError, setComparisonError] = useState("");
  const [selectedModel, setSelectedModel] = useState("random_forest_balanced");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const requestLock = useRef(false);

  const loadEvaluation = async () => {
    if (requestLock.current) return;
    requestLock.current = true;
    setLoading(true);
    setError("");
    try {
      const response = await api.evaluation();
      setEvaluation(response);
      if (response.metrics.length && !response.metrics.some(row => row.model === selectedModel)) {
        setSelectedModel(response.metrics[0].model);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load saved evaluation reports.");
    } finally {
      requestLock.current = false;
      setLoading(false);
    }
  };

  useEffect(() => { void loadEvaluation(); }, []);

  useEffect(() => {
    let active = true;
    setComparison(null);
    setComparisonError("");
    setComparisonSeconds(0);
    setComparisonLoading(true);
    const timer = window.setInterval(() => {
      if (active) setComparisonSeconds(seconds => seconds + 1);
    }, 1000);
    api.modelComparisons(dataset).then(result => {
      if (active) setComparison(result);
    }).catch(cause => {
      if (active) setComparisonError(cause instanceof Error ? cause.message : "Model comparisons could not be loaded.");
    }).finally(() => {
      if (active) setComparisonLoading(false);
      window.clearInterval(timer);
    });
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [dataset]);

  const ai4i = evaluation && getDatasetSummary(evaluation, "ai4i2020");
  const cycle = evaluation && getDatasetSummary(evaluation, "cycle001");
  const selectedRows = evaluation?.confusion_matrices.filter(row => row.model === selectedModel) ?? [];
  const chartData = evaluation?.metrics.map(row => ({
    name: row.model.replaceAll("_", " "),
    precision: row.precision,
    recall: row.recall,
    average_precision: row.average_precision
  })) ?? [];
  const datasetReady = dataset === "ai4i2020" ? Boolean(ai4i?.metrics.length) : dataset === "cycle001" && Boolean(cycle);

  return (
    <>
      <PageHeader
        eyebrow="Evidence / saved evaluation"
        title="Model evaluation"
        detail="Refresh retrieves saved Phase 2 reports from the existing /evaluation endpoint. It does not retrain models or create new evaluation results."
        action={<div className="flex flex-wrap items-center gap-2"><DatasetSelector value={dataset} onChange={onDatasetChange} /><button type="button" disabled={loading} onClick={() => void loadEvaluation()} className="inline-flex items-center gap-2 rounded-xl border border-line bg-panel px-3 py-2 text-xs text-slate-300 hover:border-cyan/40 disabled:cursor-wait disabled:opacity-50">{loading && <RefreshCw size={14} className="animate-spin" />}{loading ? "Loading..." : "Run/Refresh Evaluation"}</button></div>}
      />
      {error && <div className="mb-5" role="alert"><ErrorState title="Unable to load saved evaluation" message={evaluation ? `${error} Showing the last successfully loaded saved report.` : error} onRetry={() => void loadEvaluation()} /></div>}
      {loading && <div className="mb-5" role="status"><StatusPill status="loading" label={evaluation ? "Refreshing saved reports..." : "Loading saved reports..."} /></div>}
      {!evaluation && !loading && error && <EmptyState title="Evaluation report unavailable" detail="Check the API connection and required saved Phase 2 reports, then refresh to try again." />}
      {comparisonLoading && <div className="mb-5 rounded-xl border border-cyan/20 bg-cyan/5 p-4" role="status" aria-live="polite">
        <StatusPill status="loading" label={`Preparing ${dataset.startsWith("nasa_") ? "NASA engine-grouped model comparisons" : "model comparisons"} · ${comparisonSeconds}s elapsed`} />
        {dataset.startsWith("nasa_") && <p className="mt-2 text-xs leading-5 text-slate-400">The first NASA request evaluates several regressors on held-out engines and may take a few minutes. This is an indeterminate running state, not a percentage estimate; results appear when the API finishes.</p>}
        {dataset.startsWith("nasa_") && <EvaluationSkeleton />}
      </div>}
      {comparisonError && <div className="mb-5" role="alert"><ErrorState title="Model comparison unavailable" message={comparisonError} /></div>}
      {comparison && <ModelComparisonPanel comparison={comparison} />}
      {evaluation && dataset === "cycle001" && cycle && (
        <>
          <div className="mb-5 rounded-xl border border-amber/20 bg-amber/5 p-4 text-sm leading-6 text-amber/80">
            <Microscope size={16} className="mr-2 inline" />Cycle 001 offline anomaly report: {cycle.rows.toLocaleString()} rows, {cycle.features} scored features, and {cycle.alerts} alerts ({(cycle.alertRate * 100).toFixed(2)}%). Alerts are unsupervised; Cycle 001 has no verified failure labels.
          </div>
          <Card className="p-6">
            <SectionTitle eyebrow="Cycle 001 · saved offline report" title="Flagged timestamps" detail="These stored Phase 2 baseline rows are not live API scoring results." />
            {cycle.flagged.length ? (
              <div className="max-h-[32rem] overflow-auto rounded-lg border border-line">
                <table className="w-full text-left text-xs">
                  <thead className="sticky top-0 bg-ink text-slate-500"><tr><th className="p-3">Timestamp</th><th className="p-3">Machine ID</th><th className="p-3">Anomaly score</th><th className="p-3">Flag</th></tr></thead>
                  <tbody>{cycle.flagged.map((row, index) => <tr key={`${row.timestamp}-${row.machine_id}-${index}`} className="border-t border-line"><td className="p-3 text-slate-300">{row.timestamp}</td><td className="p-3 text-slate-400">{row.machine_id}</td><td className="p-3 text-amber">{row.anomaly_score.toFixed(5)}</td><td className="p-3 text-amber">{row.alert ? "Flagged" : "Normal"}</td></tr>)}</tbody>
                </table>
              </div>
            ) : <EmptyState title="No flagged timestamps in the report" detail="The saved report contains no Cycle 001 rows to display." />}
          </Card>
        </>
      )}
      {evaluation && dataset === "ai4i2020" && ai4i && (
        <>
          <div className="mb-5 grid gap-4 sm:grid-cols-3">
            <Kpi icon={Database} label="AI4I 2020 rows" value={ai4i.rows.toLocaleString()} detail="Saved dataset label counts" status="neutral" />
            <Kpi icon={AlertOctagon} label="Labeled failures" value={ai4i.failures.toLocaleString()} detail="Offline benchmark labels" status="warn" />
            <Kpi icon={Microscope} label="Positive label rate" value={`${(ai4i.positiveRate * 100).toFixed(2)}%`} detail="Not a live failure rate" status="neutral" />
          </div>
          {ai4i.metrics.length ? (
            <>
              <div className="mb-5 rounded-xl border border-amber/20 bg-amber/5 p-4 text-sm leading-6 text-amber/80">AI4I 2020 held-out test set: {evaluation.metrics[0]?.test_rows.toLocaleString() ?? "—"} rows. Saved precision, recall, and average precision are offline results, not production claims.</div>
              <div className="grid gap-6 xl:grid-cols-[1.35fr_1fr]">
                <Card className="p-6">
                  <SectionTitle eyebrow="AI4I 2020 · offline test set" title="Precision, recall, and average precision" detail="Loaded from the saved AI4I evaluation report." />
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 15 }} className="chart-grid">
                        <CartesianGrid strokeDasharray="3 3" /><XAxis type="number" domain={[0, 1]} /><YAxis dataKey="name" type="category" width={165} tick={{ fill: "#91a4b8", fontSize: 11 }} />
                        <Tooltip formatter={(value: number) => value.toFixed(3)} contentStyle={{ background: "#0d1a2b", border: "1px solid #1d3148", borderRadius: 8 }} />
                        <Bar dataKey="precision" fill="#51d5e5" radius={[0, 3, 3, 0]} /><Bar dataKey="recall" fill="#b8e986" radius={[0, 3, 3, 0]} /><Bar dataKey="average_precision" fill="#f7ba61" radius={[0, 3, 3, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-4 grid grid-cols-3 gap-2 text-center text-[11px] text-slate-500"><span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-cyan" />Precision</span><span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-lime" />Recall</span><span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-amber" />Avg precision</span></div>
                </Card>
                <Card className="p-6">
                  <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
                    <SectionTitle eyebrow="AI4I 2020 · offline test set" title="Confusion matrix" detail="Counts from the selected saved model." />
                    <label className="text-xs text-slate-400">Evaluation model
                      <select aria-label="Evaluation model" value={selectedModel} onChange={event => setSelectedModel(event.target.value)} className="mt-1 rounded-lg border border-line bg-ink px-3 py-2 text-xs text-slate-300">
                        {evaluation.metrics.map(row => <option key={row.model} value={row.model}>{row.model}</option>)}
                      </select>
                    </label>
                  </div>
                  {selectedRows.length ? (
                    <div className="grid grid-cols-3 gap-2 text-center text-xs">
                      <div /><div className="rounded-lg bg-lime/10 p-2 text-lime">Predicted 0</div><div className="rounded-lg bg-danger/10 p-2 text-danger">Predicted 1</div>
                      {selectedRows.map(row => <div key={`${row.model}-${row.actual}`} className="contents"><div className="flex items-center justify-center rounded-lg bg-panel p-3 text-slate-400">{row.actual.replace("actual_", "Actual ")}</div><div className="rounded-lg border border-line p-4 text-xl font-semibold text-slate-200">{row.predicted_0}</div><div className="rounded-lg border border-line p-4 text-xl font-semibold text-slate-200">{row.predicted_1}</div></div>)}
                    </div>
                  ) : <EmptyState title="No confusion rows for this model" detail="The saved report does not include confusion-matrix counts for this selection." />}
                  <div className="mt-6 space-y-2">{evaluation.metrics.map(row => <div key={row.model} className="rounded-lg border border-line bg-ink/30 p-3 text-xs"><div className="flex justify-between"><span className="text-slate-400">{row.model}</span><span className="text-cyan">F1 {(row.f1 * 100).toFixed(1)}%</span></div><div className="mt-2 h-1.5 overflow-hidden rounded-full bg-line"><div className="h-full rounded-full bg-cyan" style={{ width: `${row.average_precision * 100}%` }} /></div></div>)}</div>
                </Card>
              </div>
              <Card className="mt-6 p-6"><SectionTitle eyebrow="Traceability" title="Offline metrics, explicit boundaries" detail="AI4I subtype discrepancies are reported without changing labels. Neither dataset supports advance-warning claims." /></Card>
            </>
          ) : <EmptyState title="No AI4I evaluation metrics found" detail="The saved report endpoint did not return AI4I 2020 metrics. Refresh to retry." />}
        </>
      )}
      {dataset.startsWith("nasa_") && <EmptyState title="NASA evaluation is separate" detail="NASA C-MAPSS RUL metrics are produced in the NASA RUL Lab. They are intentionally not combined with Cycle 001 or AI4I saved evaluations." />}
      {evaluation && !datasetReady && !loading && !error && !dataset.startsWith("nasa_") && <EmptyState title="No evaluation data for this dataset" detail="The saved evaluation response did not include the selected dataset's expected report data." />}
    </>
  );
}

function ModelComparisonPanel({ comparison }: { comparison: ModelComparisonsResponse }) {
  const metadata = comparison.metadata;
  const features = metadata.feature_names;
  const seeds = metadata.random_seeds ?? metadata.validation_seeds;
  const columns = comparison.model_type === "unsupervised_anomaly_detection"
    ? ["Alerts", "Alert rate", "Overlap", "Jaccard"]
    : comparison.model_type === "remaining_useful_life_regression"
      ? ["MAE", "RMSE", "R²", "NASA score"]
      : ["Precision", "Recall", "F1", "Avg precision"];
  const metricValue = (row: typeof comparison.model_comparisons[number], column: string) => {
    const values: Record<string, number | undefined> = {
      Alerts: row.alerts,
      "Alert rate": row.alert_rate,
      Overlap: row.overlap_with_current,
      Jaccard: row.jaccard_with_current,
      MAE: row.mae,
      RMSE: row.rmse,
      "R²": row.r2,
      "NASA score": row.nasa_score,
      Precision: row.precision,
      Recall: row.recall,
      F1: row.f1,
      "Avg precision": row.average_precision
    };
    const deviations: Record<string, number | undefined> = {
      Precision: row.precision_std,
      Recall: row.recall_std,
      F1: row.f1_std,
      "Avg precision": row.average_precision_std,
      MAE: row.mae_std,
      RMSE: row.rmse_std,
      "R²": row.r2_std,
      "NASA score": row.nasa_score_std
    };
    const value = values[column];
    if (value === undefined) return "—";
    if (column === "Alert rate") return `${(value * 100).toFixed(2)}%`;
    const formatted = value.toFixed(column === "Alerts" || column === "Overlap" ? 0 : 3);
    const deviation = deviations[column];
    return deviation === undefined ? formatted : `${formatted} ± ${deviation.toFixed(3)}`;
  };
  return (
    <Card className="mb-5 overflow-hidden p-5">
      <SectionTitle
        eyebrow={`${comparison.dataset} · model audit`}
        title="Baseline and candidate comparison"
        detail={`Model family: ${comparison.model_type}. ${String(metadata.split_strategy ?? metadata.validation_split ?? "Comparisons use only documented offline evidence.")}${seeds ? ` · seeds ${Array.isArray(seeds) ? seeds.join(", ") : String(seeds)}` : ""}`}
      />
      <p className="mb-4 rounded-lg border border-cyan/20 bg-cyan/5 p-3 text-sm text-cyan">{comparison.recommendation}</p>
      <div className="overflow-auto rounded-lg border border-line">
        <table className="w-full min-w-[42rem] text-left text-xs">
          <thead className="bg-ink text-slate-500"><tr><th className="p-3">Model</th><th className="p-3">Role</th>{columns.map(column => <th key={column} className="p-3">{column}</th>)}</tr></thead>
          <tbody>{comparison.model_comparisons.map(row => <tr key={row.model} className="border-t border-line"><td className="p-3 font-medium text-slate-200">{row.model}</td><td className="p-3 text-cyan">{row.role}</td>{columns.map(column => <td key={column} className="p-3 text-slate-300">{metricValue(row, column)}</td>)}</tr>)}</tbody>
        </table>
      </div>
      <details className="mt-4 text-xs text-slate-400">
        <summary className="cursor-pointer">Reproducibility and feature details</summary>
        <pre className="mt-3 max-h-56 overflow-auto whitespace-pre-wrap rounded-lg bg-ink/60 p-3">{JSON.stringify({ ...metadata, feature_count: Array.isArray(features) ? features.length : undefined }, null, 2)}</pre>
        {Array.isArray(features) && <p className="mt-2 break-words leading-5">Features: {features.join(", ")}</p>}
      </details>
      {comparison.official_test_current && <p className="mt-3 text-xs leading-5 text-amber">{comparison.official_test_note} Current-model official test MAE {comparison.official_test_current.mae.toFixed(2)}, RMSE {comparison.official_test_current.rmse.toFixed(2)}, R² {comparison.official_test_current.r2.toFixed(3)}, NASA score {comparison.official_test_current.nasa_score.toFixed(2)}.</p>}
      {comparison.limitations && <p className="mt-3 text-xs leading-5 text-slate-500">{comparison.limitations}</p>}
    </Card>
  );
}

function nasaDatasetId(dataset: Dataset): string {
  return dataset.slice(-5).toUpperCase();
}

export function NasaExplorer({ dataset, onDatasetChange }: { dataset: Dataset; onDatasetChange: (dataset: Dataset) => void }) {
  const [overview, setOverview] = useState<NasaDatasetOverview | null>(null);
  const [history, setHistory] = useState<NasaEngineHistory | null>(null);
  const [split, setSplit] = useState<"train" | "test">("test");
  const [engineId, setEngineId] = useState(1);
  const [sensor, setSensor] = useState("");
  const [loading, setLoading] = useState(true);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [error, setError] = useState("");
  const [historyError, setHistoryError] = useState("");
  const datasetId = nasaDatasetId(dataset);

  useEffect(() => {
    let active = true;
    setOverview(null);
    setHistory(null);
    setError("");
    setLoading(true);
    api.nasaOverview(datasetId).then(result => {
      if (!active) return;
      setOverview(result);
      setEngineId(result.test.engine_ids[0] ?? result.training.engine_ids[0] ?? 1);
      setSensor(result.sensors[0] ?? "");
    }).catch(cause => {
      if (active) setError(cause instanceof Error ? cause.message : "NASA C-MAPSS overview could not be loaded.");
    }).finally(() => {
      if (active) setLoading(false);
    });
    return () => { active = false; };
  }, [datasetId]);

  useEffect(() => {
    const ids = split === "train" ? overview?.training.engine_ids : overview?.test.engine_ids;
    if (!ids?.length) return;
    if (!ids.includes(engineId)) setEngineId(ids[0]);
  }, [engineId, overview, split]);

  useEffect(() => {
    if (!overview) return;
    let active = true;
    setHistory(null);
    setHistoryError("");
    setHistoryLoading(true);
    api.nasaEngine(datasetId, engineId, split).then(result => {
      if (active) setHistory(result);
    }).catch(cause => {
      if (active) setHistoryError(cause instanceof Error ? cause.message : "NASA engine history could not be loaded.");
    }).finally(() => {
      if (active) setHistoryLoading(false);
    });
    return () => { active = false; };
  }, [datasetId, engineId, overview, split]);

  const splitSummary = split === "train" ? overview?.training : overview?.test;
  const trend = history?.rows.map(row => ({ cycle: row.cycle, value: row[sensor] })) ?? [];
  return (
    <>
      <PageHeader
        eyebrow="Data / NASA C-MAPSS"
        title="NASA engine explorer"
        detail="Explore the independent C-MAPSS train and test splits, engine cycle histories, sensor trends, dimensions, and numeric summaries."
        action={<DatasetSelector value={dataset} onChange={onDatasetChange} disabled={loading || historyLoading} />}
      />
      <div className="mb-5 rounded-xl border border-cyan/20 bg-cyan/5 p-4 text-sm leading-6 text-slate-300">
        NASA benchmark data is kept separate from Cycle 001 anomaly detection and AI4I failure prediction. This explorer is descriptive; the RUL Lab is a separate regression workflow.
      </div>
      {loading && <div className="mb-5" role="status"><StatusPill status="loading" label={`Loading ${datasetId} train, test, and RUL files...`} /></div>}
      {error && <div className="mb-5" role="alert"><ErrorState title="NASA dataset unavailable" message={error} /></div>}
      {overview && (
        <>
          <div className="mb-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <Kpi icon={Database} label="Training rows" value={overview.training.rows.toLocaleString()} detail={`${overview.training.engine_count} run-to-failure engines`} status="neutral" />
            <Kpi icon={Database} label="Test rows" value={overview.test.rows.toLocaleString()} detail={`${overview.test.engine_count} truncated engines`} status="neutral" />
            <Kpi icon={ScanSearch} label="Columns" value={overview.training.columns.toString()} detail="Engine, cycle, 3 settings, 21 sensors" status="neutral" />
            <Kpi icon={Database} label="RUL labels" value={overview.test.engine_count.toString()} detail="Separate one-value-per-test-engine file" status="neutral" />
          </div>
          <Card className="mb-5 p-5">
            <SectionTitle eyebrow={`${datasetId} · trajectories`} title="Engine cycle history" detail="Select a split and engine to view all recorded cycles. The test split is truncated before failure." />
            <div className="mb-4 flex flex-wrap gap-3">
              <label className="text-xs text-slate-400">Split
                <select aria-label="NASA data split" value={split} onChange={event => setSplit(event.target.value as "train" | "test")} className="ml-2 rounded-lg border border-line bg-ink px-3 py-2 text-xs text-slate-200">
                  <option value="test">Test</option><option value="train">Train</option>
                </select>
              </label>
              <label className="text-xs text-slate-400">Engine
                <select aria-label="NASA engine" value={engineId} onChange={event => setEngineId(Number(event.target.value))} className="ml-2 rounded-lg border border-line bg-ink px-3 py-2 text-xs text-slate-200">
                  {(split === "train" ? overview.training.engine_ids : overview.test.engine_ids).map(id => <option key={id} value={id}>{id}</option>)}
                </select>
              </label>
              <label className="text-xs text-slate-400">Sensor trend
                <select aria-label="NASA sensor" value={sensor} onChange={event => setSensor(event.target.value)} className="ml-2 rounded-lg border border-line bg-ink px-3 py-2 text-xs text-slate-200">
                  {overview.sensors.map(name => <option key={name} value={name}>{name}</option>)}
                </select>
              </label>
            </div>
            {historyLoading && <div role="status"><StatusPill status="loading" label="Loading engine cycle history..." /></div>}
            {historyError && <div role="alert"><ErrorState title="Engine history unavailable" message={historyError} /></div>}
            {history && (
              <>
                <p className="mb-3 text-xs text-slate-500">{history.rows.length} observed cycles · engine {history.engine_id} · {history.split} split</p>
                <div className="mb-5 h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trend} className="chart-grid">
                      <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="cycle" /><YAxis /><Tooltip contentStyle={{ background: "#0d1a2b", border: "1px solid #1d3148", borderRadius: 8 }} />
                      <Line type="monotone" dataKey="value" name={sensor} stroke="#51d5e5" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="max-h-80 overflow-auto rounded-lg border border-line">
                  <table className="w-full text-left text-xs">
                    <thead className="sticky top-0 bg-ink text-slate-500"><tr><th className="p-2">Cycle</th><th className="p-2">{sensor}</th>{overview.settings.map(name => <th key={name} className="p-2">{name}</th>)}</tr></thead>
                    <tbody>{history.rows.map(row => <tr key={row.cycle} className="border-t border-line"><td className="p-2 text-slate-300">{row.cycle}</td><td className="p-2 text-cyan">{row[sensor]?.toFixed(4)}</td>{overview.settings.map(name => <td key={name} className="p-2 text-slate-400">{row[name]?.toFixed(4)}</td>)}</tr>)}</tbody>
                  </table>
                </div>
              </>
            )}
          </Card>
          <Card className="p-5">
            <SectionTitle eyebrow={`${split} split · numeric profile`} title="Numeric summaries" detail="Mean, standard deviation, minimum, and maximum are computed from every row in the selected split. No rows are sampled or dropped." />
            <div className="max-h-[32rem] overflow-auto rounded-lg border border-line">
              <table className="w-full min-w-[36rem] text-left text-xs">
                <thead className="sticky top-0 bg-ink text-slate-500"><tr><th className="p-2">Column</th><th className="p-2">Count</th><th className="p-2">Mean</th><th className="p-2">Std dev</th><th className="p-2">Min</th><th className="p-2">Max</th></tr></thead>
                <tbody>{splitSummary?.numeric_summary.map(item => <tr key={item.column} className="border-t border-line"><td className="p-2 text-slate-200">{item.column}</td><td className="p-2 text-slate-400">{item.count.toLocaleString()}</td><td className="p-2 text-cyan">{item.mean.toFixed(4)}</td><td className="p-2 text-slate-400">{item.std.toFixed(4)}</td><td className="p-2 text-slate-400">{item.min.toFixed(4)}</td><td className="p-2 text-slate-400">{item.max.toFixed(4)}</td></tr>)}</tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </>
  );
}

export function NasaRulLab({ dataset, onDatasetChange }: { dataset: Dataset; onDatasetChange: (dataset: Dataset) => void }) {
  const [overview, setOverview] = useState<NasaDatasetOverview | null>(null);
  const [evaluation, setEvaluation] = useState<NasaRulEvaluation | null>(null);
  const [engineId, setEngineId] = useState(1);
  const [prediction, setPrediction] = useState<NasaRulPrediction | null>(null);
  const [overviewLoading, setOverviewLoading] = useState(false);
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [evaluationSeconds, setEvaluationSeconds] = useState(0);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState("");
  const [retryKey, setRetryKey] = useState(0);
  const datasetId = dataset.startsWith("nasa_") ? nasaDatasetId(dataset) : "";

  useEffect(() => {
    if (!datasetId) {
      setOverview(null);
      setEvaluation(null);
      setPrediction(null);
      setOverviewLoading(false);
      setEvaluationLoading(false);
      return;
    }
    let active = true;
    setOverview(null);
    setEvaluation(null);
    setOverviewLoading(true);
    setEvaluationLoading(true);
    setEvaluationSeconds(0);
    setError("");
    setPrediction(null);
    const timer = window.setInterval(() => {
      if (active) setEvaluationSeconds(seconds => seconds + 1);
    }, 1000);
    api.nasaOverview(datasetId).then(datasetOverview => {
      if (!active) return;
      setOverview(datasetOverview);
      setEngineId(datasetOverview.test.engine_ids[0] ?? 1);
    }).catch(cause => {
      if (active) setError(cause instanceof Error ? cause.message : "NASA dataset overview could not be loaded.");
    }).finally(() => {
      if (active) setOverviewLoading(false);
    });
    api.nasaRulEvaluation(datasetId).then(rulEvaluation => {
      if (active) setEvaluation(rulEvaluation);
    }).catch(cause => {
      if (active) setError(cause instanceof Error ? cause.message : "NASA RUL evaluation could not be loaded.");
    }).finally(() => {
      if (active) setEvaluationLoading(false);
      window.clearInterval(timer);
    });
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [datasetId, retryKey]);
  const loading = overviewLoading || evaluationLoading;

  const predict = async () => {
    if (!datasetId || predicting) return;
    setPredicting(true);
    setError("");
    setPrediction(null);
    try {
      setPrediction(await api.nasaRulPredict(datasetId, engineId));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "NASA RUL prediction failed.");
    } finally {
      setPredicting(false);
    }
  };

  return (
    <>
      <PageHeader
        eyebrow="Prediction / NASA C-MAPSS"
        title="Remaining useful life lab"
        detail="Train and evaluate a separate regression baseline on NASA C-MAPSS trajectories. It does not use or change the Cycle 001 or AI4I models."
        action={<DatasetSelector value={dataset} onChange={onDatasetChange} disabled={loading || predicting} />}
      />
      {!datasetId && <div className="mb-5 rounded-xl border border-amber/20 bg-amber/5 p-4 text-sm leading-6 text-amber" role="status">Select one of the NASA C-MAPSS FD001–FD004 datasets to load its RUL benchmark. Existing Cycle 001 and AI4I datasets are not compatible with this regressor.</div>}
      {overviewLoading && <div className="mb-5" role="status"><StatusPill status="loading" label={`Loading ${datasetId} dataset details...`} /></div>}
      {evaluationLoading && <div className="mb-5 rounded-xl border border-cyan/20 bg-cyan/5 p-4" role="status" aria-live="polite">
        <StatusPill status="loading" label={`Running ${datasetId} engine-held-out RUL comparison · ${evaluationSeconds}s elapsed`} />
        <p className="mt-2 text-xs leading-5 text-slate-400">The first run fits the current model and candidate regressors on engine-grouped simulated truncations. No percentage is shown because the backend evaluation runs as one task; the page stays active and will display results when complete.</p>
        <EvaluationSkeleton />
      </div>}
      {error && <div className="mb-5" role="alert"><ErrorState title="NASA RUL workflow unavailable" message={error} onRetry={() => setRetryKey(value => value + 1)} /></div>}
      {evaluation && overview && (
        <>
          <div className="mb-5 rounded-xl border border-amber/20 bg-amber/5 p-4 text-sm leading-6 text-amber/90">
            Target: {evaluation.target_definition} Validation holds out whole engines using {evaluation.validation.split}. The separate official test RUL labels are used only for the test metric and displayed benchmark comparison, never for model fitting.
          </div>
          <div className="mb-5 grid gap-4 md:grid-cols-2">
            <RulMetricCard title="Engine-held-out validation" detail={`${evaluation.validation.engine_count} engines · ${evaluation.validation.row_count.toLocaleString()} cycle rows`} metrics={evaluation.validation} />
            <RulMetricCard title="Official test benchmark" detail={`${evaluation.official_test.engine_count} engines · final observed cycle per engine`} metrics={evaluation.official_test} />
          </div>
          <Card className="mb-5 p-5">
            <SectionTitle eyebrow={`${datasetId} · test engines`} title="Estimate remaining useful life" detail="The model uses the selected test engine's last observed cycle, operational settings, and sensor values. Its official benchmark label is shown only for offline comparison." />
            <div className="flex flex-wrap items-end gap-3">
              <label className="text-xs text-slate-400">Test engine
                <select aria-label="RUL test engine" value={engineId} onChange={event => { setEngineId(Number(event.target.value)); setPrediction(null); }} className="mt-1 block rounded-lg border border-line bg-ink px-3 py-2 text-xs text-slate-200">
                  {overview.test.engine_ids.map(id => <option key={id} value={id}>Engine {id}</option>)}
                </select>
              </label>
              <button type="button" disabled={predicting || loading} onClick={() => void predict()} className="inline-flex items-center gap-2 rounded-xl bg-cyan px-4 py-2.5 text-sm font-semibold text-ink hover:bg-cyan/90 disabled:cursor-wait disabled:opacity-50">
                {predicting && <RefreshCw size={15} className="animate-spin" />}{predicting ? "Predicting..." : "Predict RUL"}
              </button>
            </div>
            {prediction && <div className="mt-5 grid gap-4 sm:grid-cols-3" aria-live="polite">
              <Kpi icon={CircleGauge} label="Predicted RUL" value={`${prediction.predicted_rul.toFixed(1)} cycles`} detail={`At engine cycle ${prediction.last_observed_cycle}`} status="ok" />
              <Kpi icon={Database} label="Official benchmark RUL" value={`${prediction.benchmark_rul} cycles`} detail="Ground-truth label for offline comparison" status="neutral" />
              <Kpi icon={Microscope} label="Absolute error" value={`${Math.abs(prediction.predicted_rul - prediction.benchmark_rul).toFixed(1)} cycles`} detail="One benchmark engine; not a live forecast guarantee" status="warn" />
            </div>}
          </Card>
          <div className="rounded-xl border border-line bg-panel/60 p-4 text-sm leading-6 text-slate-400">
            <p className="font-medium text-slate-200">Evaluation and limitations</p>
            <p className="mt-2">MAE and RMSE are cycle errors; R² is the coefficient of determination. NASA score is the standard asymmetric sum that penalizes late (overestimated RUL) predictions more strongly than early predictions. Validation is grouped by engine to prevent cycles from the same engine crossing the train/validation boundary.</p>
            <p className="mt-2">{evaluation.limitations}</p>
          </div>
        </>
      )}
    </>
  );
}

function RulMetricCard({ title, detail, metrics }: { title: string; detail: string; metrics: NasaRulEvaluation["validation"] }) {
  return (
    <Card className="p-5">
      <SectionTitle eyebrow="Offline regression metrics" title={title} detail={detail} />
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-lg border border-line p-3"><p className="text-xs text-slate-500">MAE · cycles</p><p className="mt-1 font-semibold text-cyan">{metrics.mae.toFixed(2)}</p></div>
        <div className="rounded-lg border border-line p-3"><p className="text-xs text-slate-500">RMSE · cycles</p><p className="mt-1 font-semibold text-cyan">{metrics.rmse.toFixed(2)}</p></div>
        <div className="rounded-lg border border-line p-3"><p className="text-xs text-slate-500">R²</p><p className="mt-1 font-semibold text-slate-200">{metrics.r2.toFixed(3)}</p></div>
        <div className="rounded-lg border border-line p-3"><p className="text-xs text-slate-500">NASA score</p><p className="mt-1 font-semibold text-amber">{metrics.nasa_score.toFixed(2)}</p></div>
      </div>
    </Card>
  );
}

function EvaluationSkeleton() {
  return (
    <div aria-hidden="true" className="mt-4 grid gap-3 sm:grid-cols-3">
      {[0, 1, 2].map(item => (
        <div key={item} className="rounded-xl border border-line/80 bg-ink/30 p-3">
          <div className="skeleton-shimmer h-2 w-2/5 rounded" />
          <div className="mt-4 grid grid-cols-2 gap-2">
            <div className="skeleton-shimmer h-8 rounded-md" />
            <div className="skeleton-shimmer h-8 rounded-md" />
          </div>
          <div className="skeleton-shimmer mt-3 h-2 w-3/4 rounded" />
        </div>
      ))}
    </div>
  );
}
