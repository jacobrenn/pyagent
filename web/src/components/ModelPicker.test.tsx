// @vitest-environment jsdom
import { render } from "preact";
import { act } from "preact/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { ModelPicker } from "./ModelPicker";

type PickerProps = Parameters<typeof ModelPicker>[0];
type ModelResponse = Awaited<ReturnType<typeof api.listProfileModels>>;

let container: HTMLDivElement;
let onChange: ReturnType<typeof vi.fn>;

beforeEach(() => {
  container = document.createElement("div");
  document.body.append(container);
  onChange = vi.fn();
});

afterEach(() => {
  act(() => render(null, container));
  container.remove();
  vi.restoreAllMocks();
});

function mount(props: Partial<PickerProps> = {}) {
  act(() => render(
    <ModelPicker
      id="model"
      label="Model"
      value="current-model"
      profileName="local"
      required
      onChange={onChange}
      {...props}
    />,
    container,
  ));
}

function button(text: string): HTMLButtonElement {
  const found = [...container.querySelectorAll("button")].find((item) => item.textContent === text);
  if (!found) throw new Error(`Missing button: ${text}`);
  return found;
}

async function click(text: string) {
  await act(async () => button(text).click());
}

function select(): HTMLSelectElement {
  const found = container.querySelector("select");
  if (!found) throw new Error("Missing model dropdown");
  return found;
}

describe("ModelPicker", () => {
  it("shows discovered models in a native dropdown without changing an unlisted current model", async () => {
    const discover = vi.spyOn(api, "listProfileModels").mockResolvedValue({
      profile: "local",
      models: ["model-a", "model-b", "model-a"],
    });
    mount();
    await click("Discover models");

    expect(discover).toHaveBeenCalledWith("local");
    expect([...select().options].map((option) => option.value)).toEqual([
      "", "current-model", "model-a", "model-b",
    ]);
    expect(select().value).toBe("current-model");
    expect(select().required).toBe(true);
    expect(select().options[0].disabled).toBe(true);
    expect(onChange).not.toHaveBeenCalled();
    act(() => {
      select().value = "model-b";
      select().dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(onChange).toHaveBeenCalledWith("model-b");
  });

  it("allows manual entry and returning to the dropdown without losing the custom model", async () => {
    vi.spyOn(api, "listProfileModels").mockResolvedValue({ profile: "local", models: ["model-a"] });
    mount();
    await click("Discover models");
    await click("Enter a model manually");
    const input = container.querySelector("input")!;
    expect(input.value).toBe("current-model");
    act(() => {
      input.value = "custom-model";
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    expect(onChange).toHaveBeenCalledWith("custom-model");
    mount({ value: "custom-model" });
    await click("Choose from discovered models");
    expect(select().value).toBe("custom-model");
    expect([...select().options].map((option) => option.value)).toContain("model-a");
  });

  it("keeps agent inheritance selectable and does not choose an override automatically", async () => {
    vi.spyOn(api, "listProfileModels").mockResolvedValue({ profile: "local", models: ["model-a"] });
    mount({ required: false, value: "", profileModel: "profile-default" });
    await click("Discover models");
    expect(select().value).toBe("");
    expect(select().options[0].textContent).toBe("Use profile model (profile-default)");
    expect(select().options[0].disabled).toBe(false);
    expect(onChange).not.toHaveBeenCalled();

    mount({ required: false, value: "model-a", profileModel: "profile-default" });
    act(() => {
      select().value = "";
      select().dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(onChange).toHaveBeenCalledWith("");
  });

  it("keeps manual entry available for a new profile that cannot be discovered yet", () => {
    mount({ profileName: undefined, value: "", discoveryHint: "Save this profile before discovering models." });
    expect(button("Discover models").disabled).toBe(true);
    expect(container.querySelector("input")?.required).toBe(true);
    expect(container.textContent).toContain("Save this profile before discovering models.");
  });

  it("handles empty listings and failed discovery without losing manual input, and allows retry", async () => {
    vi.spyOn(api, "listProfileModels")
      .mockResolvedValueOnce({ profile: "local", models: [] })
      .mockRejectedValueOnce(new Error("Endpoint unavailable"))
      .mockResolvedValueOnce({ profile: "local", models: ["model-a"] });
    mount();
    await click("Discover models");
    expect(container.querySelector("select")).toBeNull();
    expect(container.querySelector("input")?.value).toBe("current-model");
    expect(container.querySelector('[role="status"]')?.textContent).toContain("No models reported");

    await click("Discover models");
    expect(container.querySelector('[role="alert"]')?.textContent).toBe("Endpoint unavailable");
    expect(container.querySelector("input")?.value).toBe("current-model");
    expect(button("Discover models").disabled).toBe(false);
    await click("Discover models");
    expect(select().value).toBe("current-model");
    expect(container.querySelector('[role="alert"]')).toBeNull();
  });

  it.each(["resolve", "reject"])("ignores a late %s after switching profiles", async (outcome) => {
    let resolve!: (response: ModelResponse) => void;
    let reject!: (error: Error) => void;
    const pending = new Promise<ModelResponse>((yes, no) => { resolve = yes; reject = no; });
    vi.spyOn(api, "listProfileModels")
      .mockReturnValueOnce(pending)
      .mockResolvedValueOnce({ profile: "remote", models: ["remote-model"] });
    mount();
    await click("Discover models");
    expect(button("Discovering…").disabled).toBe(true);
    mount({ profileName: "remote" });
    await click("Discover models");
    await act(async () => {
      if (outcome === "resolve") resolve({ profile: "local", models: ["stale-model"] });
      else reject(new Error("Stale failure"));
    });
    expect([...select().options].map((option) => option.value)).toEqual([
      "", "current-model", "remote-model",
    ]);
    expect(container.querySelector('[role="alert"]')).toBeNull();
    expect(button("Discover models").disabled).toBe(false);
  });

  it("clears the previously discovered list when the profile changes", async () => {
    vi.spyOn(api, "listProfileModels").mockResolvedValue({ profile: "local", models: ["model-a"] });
    mount();
    await click("Discover models");
    mount({ profileName: "remote" });
    expect(container.querySelector("select")).toBeNull();
    expect(container.querySelector("input")?.value).toBe("current-model");
    expect(container.querySelector('[role="status"]')).toBeNull();
  });
});
