import { useEffect, useRef } from "preact/hooks";
import type { TranscriptEntry } from "../types";
import { Markdown } from "./Markdown";
import { ToolActivity } from "./ToolActivity";

interface TranscriptProps {
  entries: TranscriptEntry[];
  activeAssistantId: string | null;
  streaming: boolean;
}

export function Transcript({
  entries,
  activeAssistantId,
  streaming,
}: TranscriptProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const followOutputRef = useRef(true);

  useEffect(() => {
    const shell = bottomRef.current?.parentElement?.parentElement;
    if (!shell) return;
    const handleScroll = () => {
      followOutputRef.current =
        shell.scrollHeight - shell.scrollTop - shell.clientHeight < 100;
    };
    shell.addEventListener("scroll", handleScroll, { passive: true });
    return () => shell.removeEventListener("scroll", handleScroll);
  }, [entries.length]);

  useEffect(() => {
    if (followOutputRef.current) {
      bottomRef.current?.scrollIntoView({ block: "end" });
    }
  }, [entries, activeAssistantId, streaming]);

  if (!entries.length) {
    return (
      <div class="empty-state">
        <div class="empty-state__mark">P</div>
        <p class="eyebrow">Workspace copilot</p>
        <h2>What are we building?</h2>
        <p>
          Ask PyAgent to inspect the project, explain code, make changes, or run
          its configured tools.
        </p>
        <div class="prompt-ideas">
          <span>Summarize this repository</span>
          <span>Review the current changes</span>
          <span>Find the next useful task</span>
        </div>
      </div>
    );
  }

  return (
    <div class="transcript" aria-live="polite">
      {entries.map((entry) => {
        if (entry.kind === "tool") {
          return <ToolActivity key={entry.id} entry={entry} />;
        }
        const isActive = streaming && entry.id === activeAssistantId;
        return (
          <article key={entry.id} class={`message message--${entry.kind}`}>
            <header>
              <span>
                {entry.kind === "user"
                  ? "You"
                  : entry.kind === "assistant"
                    ? "PyAgent"
                    : entry.kind === "error"
                      ? "Run error"
                      : "System"}
              </span>
              <time dateTime={entry.createdAt}>
                {new Date(entry.createdAt).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </time>
            </header>
            {entry.kind === "assistant" ? (
              entry.content ? (
                <Markdown content={entry.content} streaming={isActive} />
              ) : isActive ? (
                <div class="thinking-dots" aria-label="PyAgent is thinking">
                  <i />
                  <i />
                  <i />
                </div>
              ) : (
                <div class="message__empty">No response content.</div>
              )
            ) : (
              <div class="message__plain">{entry.content}</div>
            )}
          </article>
        );
      })}
      <div ref={bottomRef} />
    </div>
  );
}
