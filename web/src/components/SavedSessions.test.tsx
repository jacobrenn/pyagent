// @vitest-environment jsdom
import { render } from "preact";
import { act } from "preact/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { SavedRunList } from "../types";
import { SavedSessions } from "./SavedSessions";

vi.mock("../api/client", () => ({ api: { listSessions: vi.fn(), showSession: vi.fn() } }));
let container: HTMLDivElement;
const empty: SavedRunList = { runs: [], total: 41, offset: 0, limit: 20, unavailable_count: 2 };
const imported = vi.fn();

beforeEach(() => {
  vi.resetAllMocks();
  container = document.createElement("div");
  document.body.append(container);
  vi.mocked(api.listSessions).mockImplementation(async (offset = 0) => ({ ...empty, offset }));
});
afterEach(() => { act(() => render(null, container)); container.remove(); });

async function click(text: string) {
  await act(async () => {
    const button = [...container.querySelectorAll("button")].find((item) => item.textContent?.trim() === text);
    expect(button).toBeDefined();
    button!.click();
  });
}

describe("saved checkpoint browser", () => {
  it("loads only on request and paginates without importing or executing", async () => {
    act(() => render(<SavedSessions disabled={false} onImport={imported} />, container));
    expect(api.listSessions).not.toHaveBeenCalled();
    await click("Saved checkpoints");
    expect(api.listSessions).toHaveBeenLastCalledWith(0);
    expect(container.textContent).toContain("2 legacy or invalid");
    await click("Next");
    expect(api.listSessions).toHaveBeenLastCalledWith(20);
    await click("Previous");
    expect(api.listSessions).toHaveBeenLastCalledWith(0);
    expect(imported).not.toHaveBeenCalled();
  });

  it("ignores a late listing when streaming disables checkpoint operations", async () => {
    let release!: (value: SavedRunList) => void;
    vi.mocked(api.listSessions).mockReturnValue(new Promise((resolve) => { release = resolve; }));
    act(() => render(<SavedSessions disabled={false} onImport={imported} />, container));
    await click("Saved checkpoints");
    act(() => render(<SavedSessions disabled={true} onImport={imported} />, container));
    await act(async () => { release(empty); });
    expect(container.textContent).not.toContain("41 runs");
    expect(imported).not.toHaveBeenCalled();
    expect(container.querySelector("button")!.disabled).toBe(true);
  });

  it("ignores late results after unmount", async () => {
    let release!: (value: SavedRunList) => void;
    vi.mocked(api.listSessions).mockReturnValue(new Promise((resolve) => { release = resolve; }));
    act(() => render(<SavedSessions disabled={false} onImport={imported} />, container));
    await click("Saved checkpoints");
    act(() => render(null, container));
    await act(async () => { release(empty); });
    expect(container.textContent).toBe("");
    expect(imported).not.toHaveBeenCalled();
  });
});
