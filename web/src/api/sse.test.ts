import { describe, expect, it } from "vitest";
import { consumeSse, SseParser } from "./sse";

function streamFor(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });
}

describe("SseParser", () => {
  it("parses split chunks, multiline data, comments, and event IDs", () => {
    const events: Array<Record<string, unknown>> = [];
    const parser = new SseParser((event) => events.push(event));

    parser.push(": heart");
    parser.push("beat\n\nid: 2\nevent: content_delta\n");
    parser.push('data: {"type":"content_delta",\n');
    parser.push('data: "delta":"héllo\\nworld"}\n\n');
    parser.finish();

    expect(events).toEqual([
      {
        type: "content_delta",
        sequence: 2,
        delta: "héllo\nworld",
      },
    ]);
  });

  it("uses the SSE event name when type is absent", () => {
    const events: Array<Record<string, unknown>> = [];
    const parser = new SseParser((event) => events.push(event));

    parser.push('event: done\ndata: {"response":"ok"}');
    parser.finish();

    expect(events).toEqual([{ type: "done", response: "ok" }]);
  });

  it("preserves UTF-8 code points split across response chunks", async () => {
    const encoded = new TextEncoder().encode(
      'event: content_delta\ndata: {"type":"content_delta","delta":"héllo"}\n\n' +
        'event: done\ndata: {"type":"done","response":"héllo"}\n\n',
    );
    const splitAt = encoded.indexOf(0xc3) + 1;
    const events: Array<Record<string, unknown>> = [];

    await consumeSse(
      streamFor([encoded.slice(0, splitAt), encoded.slice(splitAt)]),
      (event) => events.push(event),
    );

    expect(events[0].delta).toBe("héllo");
    expect(events.at(-1)?.type).toBe("done");
  });

  it("requires a terminal done or error event", async () => {
    const stream = streamFor([
      new TextEncoder().encode('event: start\ndata: {"type":"start"}\n\n'),
    ]);

    await expect(consumeSse(stream, () => undefined)).rejects.toThrow(
      "without a terminal event",
    );
  });

  it("rejects malformed JSON events", () => {
    const parser = new SseParser(() => undefined);
    expect(() => parser.push("event: done\ndata: not-json\n\n")).toThrow();
  });
});
