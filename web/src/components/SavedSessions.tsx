import { useLayoutEffect, useRef, useState } from "preact/hooks";
import { api } from "../api/client";
import type { SavedRunDetail, SavedRunList } from "../types";

interface SavedSessionsProps {
  disabled: boolean;
  onImport: (checkpoint: SavedRunDetail) => void;
}

export function SavedSessions({ disabled, onImport }: SavedSessionsProps) {
  const [open, setOpen] = useState(false);
  const [listing, setListing] = useState<SavedRunList | null>(null);
  const [selected, setSelected] = useState<SavedRunDetail | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const requestId = useRef(0);

  useLayoutEffect(() => {
    if (disabled) {
      requestId.current += 1;
      setBusy(false);
    }
    return () => { requestId.current += 1; };
  }, [disabled]);

  const load = async (offset = 0) => {
    const current = ++requestId.current;
    setOpen(true);
    setBusy(true);
    setError("");
    setSelected(null);
    try {
      const result = await api.listSessions(offset);
      if (current === requestId.current) setListing(result);
    } catch (caught) {
      if (current === requestId.current) {
        setListing(null);
        setError(caught instanceof Error ? caught.message : "Could not load saved checkpoints.");
      }
    } finally {
      if (current === requestId.current) setBusy(false);
    }
  };

  const inspect = async (runId: string) => {
    const current = ++requestId.current;
    setBusy(true);
    setSelected(null);
    setError("");
    try {
      const result = await api.showSession(runId);
      if (current === requestId.current) setSelected(result);
    } catch (caught) {
      if (current === requestId.current) setError(caught instanceof Error ? caught.message : "Could not inspect checkpoint.");
    } finally {
      if (current === requestId.current) setBusy(false);
    }
  };

  return (
    <section class="saved-sessions" aria-label="Saved server checkpoints">
      <button disabled={disabled || busy} onClick={() => {
        if (open) { setOpen(false); setSelected(null); }
        else void load();
      }}>
        {open ? "Hide checkpoints" : "Saved checkpoints"}
      </button>
      {open && <div class="saved-sessions__panel" aria-busy={busy}>
        <p>Server logs contain sensitive history. Import creates a new browser-local conversation; no prompt is sent.</p>
        <button disabled={disabled || busy} onClick={() => void load()}>Refresh checkpoints</button>
        {error && <p role="alert">{error}</p>}
        {busy && <p role="status">Loading checkpoints…</p>}
        {listing && <>
          {!listing.runs.length && <p>No saved checkpoints on this page.</p>}
          {listing.unavailable_count > 0 && <p>{listing.unavailable_count} legacy or invalid files are available only for local inspection.</p>}
          <div class="saved-sessions__list">
            {listing.runs.map((run) => <button
              key={run.run_id} disabled={disabled || busy || !run.resumable}
              title={run.reason} onClick={() => void inspect(run.run_id)}
            >
              <strong>{run.prompt_preview || run.run_id.slice(0, 8)}</strong>
              <small>{run.agent ? `${run.agent} r${run.agent_revision}` : run.profile} · {run.model}</small>
              <small>{run.timestamp} · {run.outcome}</small>
              <small>Conversation: {(run.conversation_id ?? run.run_id).slice(0, 8)} · parent: {run.parent_run_id?.slice(0, 8) ?? "none"}</small>
              <small>{run.workspace}</small>
              {!run.resumable && <small>{run.reason}</small>}
              {run.warnings.map((warning) => <small key={warning}>{warning}</small>)}
            </button>)}
          </div>
          <div class="saved-sessions__pagination">
            <button disabled={disabled || busy || listing.offset === 0}
              onClick={() => void load(Math.max(0, listing.offset - listing.limit))}>Previous</button>
            <span>{listing.total} runs</span>
            <button disabled={disabled || busy || listing.offset + listing.limit >= listing.total}
              onClick={() => void load(listing.offset + listing.limit)}>Next</button>
          </div>
        </>}
        {selected && <div class="saved-sessions__selection">
          <strong>Import checkpoint</strong>
          <code>{selected.run_id}</code>
          <p>{selected.messages.length} checkpoint messages. Current resources and credentials will be used when continuing.</p>
          {selected.warnings.map((warning) => <p key={warning}>{warning}</p>)}
          <button disabled={disabled || busy || !selected.resumable} onClick={() => {
            onImport(selected);
            setOpen(false);
            setSelected(null);
          }}>Import as new conversation</button>
        </div>}
      </div>}
    </section>
  );
}
