import type { StreamEvent } from "../types";

export class SseParser {
  private buffer = "";
  private eventName = "";
  private eventId = "";
  private dataLines: string[] = [];

  constructor(private readonly onEvent: (event: StreamEvent) => void) {}

  push(chunk: string): void {
    this.buffer += chunk;
    let newline = this.buffer.indexOf("\n");
    while (newline >= 0) {
      const rawLine = this.buffer.slice(0, newline);
      this.buffer = this.buffer.slice(newline + 1);
      this.processLine(rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine);
      newline = this.buffer.indexOf("\n");
    }
  }

  finish(): void {
    if (this.buffer) {
      this.processLine(this.buffer.endsWith("\r") ? this.buffer.slice(0, -1) : this.buffer);
      this.buffer = "";
    }
    this.dispatch();
  }

  private processLine(line: string): void {
    if (!line) {
      this.dispatch();
      return;
    }
    if (line.startsWith(":")) {
      return;
    }

    const separator = line.indexOf(":");
    const field = separator >= 0 ? line.slice(0, separator) : line;
    let value = separator >= 0 ? line.slice(separator + 1) : "";
    if (value.startsWith(" ")) {
      value = value.slice(1);
    }

    if (field === "event") {
      this.eventName = value;
    } else if (field === "id" && !value.includes("\0")) {
      this.eventId = value;
    } else if (field === "data") {
      this.dataLines.push(value);
    }
  }

  private dispatch(): void {
    if (!this.dataLines.length) {
      this.resetEvent();
      return;
    }

    const decoded: unknown = JSON.parse(this.dataLines.join("\n"));
    if (!decoded || typeof decoded !== "object" || Array.isArray(decoded)) {
      throw new Error("PyAgent returned a non-object SSE event.");
    }
    const event = decoded as StreamEvent;
    if (!event.type && this.eventName) {
      event.type = this.eventName;
    }
    if (event.sequence === undefined && this.eventId) {
      const numericId = Number(this.eventId);
      event.sequence = Number.isFinite(numericId) ? numericId : this.eventId;
    }
    if (!event.type) {
      throw new Error("PyAgent returned an SSE event without a type.");
    }
    this.onEvent(event);
    this.resetEvent();
  }

  private resetEvent(): void {
    this.eventName = "";
    this.eventId = "";
    this.dataLines = [];
  }
}

export async function consumeSse(
  body: ReadableStream<Uint8Array>,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let terminalSeen = false;
  const parser = new SseParser((event) => {
    if (terminalSeen) {
      throw new Error("The PyAgent stream emitted data after its terminal event.");
    }
    if (event.type === "done" || event.type === "error") {
      terminalSeen = true;
    }
    onEvent(event);
  });
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      parser.push(decoder.decode(value, { stream: true }));
    }
    parser.push(decoder.decode());
    parser.finish();
    if (!terminalSeen) {
      throw new Error("The PyAgent stream ended without a terminal event.");
    }
  } finally {
    reader.releaseLock();
  }
}
