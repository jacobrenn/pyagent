import type { TranscriptEntry } from "../types";

export function ToolActivity({ entry }: { entry: TranscriptEntry }) {
  const status = entry.toolStatus ?? "running";
  const summary =
    status === "running" ? "Running" : status === "error" ? "Failed" : "Complete";
  return (
    <details class={`tool-card tool-card--${status}`} open={status === "running"}>
      <summary>
        <span class="tool-card__icon" aria-hidden="true">
          {status === "running" ? "↻" : status === "error" ? "!" : "✓"}
        </span>
        <span class="tool-card__name">{entry.toolName || "Tool"}</span>
        <span class="tool-card__status">{summary}</span>
      </summary>
      <div class="tool-card__body">
        {entry.arguments && Object.keys(entry.arguments).length > 0 && (
          <section>
            <h4>Arguments</h4>
            <pre>{JSON.stringify(entry.arguments, null, 2)}</pre>
          </section>
        )}
        {entry.result !== undefined && (
          <section>
            <h4>Result</h4>
            <pre>{entry.result}</pre>
          </section>
        )}
      </div>
    </details>
  );
}
