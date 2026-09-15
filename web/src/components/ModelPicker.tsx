import { useLayoutEffect, useRef, useState } from "preact/hooks";
import { api } from "../api/client";

interface ModelPickerProps {
  id: string;
  label: string;
  value: string;
  onChange: (model: string) => void;
  profileName?: string;
  profileModel?: string;
  required?: boolean;
  discoveryHint?: string;
}

export function ModelPicker({
  id,
  label,
  value,
  onChange,
  profileName,
  profileModel,
  required = false,
  discoveryHint,
}: ModelPickerProps) {
  const [models, setModels] = useState<string[]>([]);
  const [manual, setManual] = useState(false);
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const requestId = useRef(0);

  useLayoutEffect(() => {
    setModels([]);
    setManual(false);
    setLoading(false);
    setNotice("");
    setError("");
    return () => {
      // A late response must not populate another profile's model selector.
      requestId.current += 1;
    };
  }, [profileName]);

  const discover = async () => {
    if (!profileName) return;
    const currentRequest = ++requestId.current;
    setLoading(true);
    setNotice("");
    setError("");
    try {
      const response = await api.listProfileModels(profileName);
      if (currentRequest !== requestId.current) return;
      const available = [...new Set(response.models)];
      setModels(available);
      setManual(false);
      setNotice(
        available.length
          ? `Found ${available.length} models.`
          : "No models reported. You can still enter a model name manually.",
      );
    } catch (caught) {
      if (currentRequest !== requestId.current) return;
      setError(caught instanceof Error ? caught.message : "Could not discover models.");
    } finally {
      if (currentRequest === requestId.current) setLoading(false);
    }
  };

  const options = value && !models.includes(value) ? [value, ...models] : models;
  const useDropdown = models.length > 0 && !manual;
  const inheritedLabel = profileModel
    ? `Use profile model (${profileModel})`
    : "Use the profile model";
  const helpId = `${id}-help`;

  return (
    <div class="model-picker" aria-busy={loading}>
      <label htmlFor={id}>
        <span>{label}{!required && <small>optional</small>}</span>
      </label>
      <div class="field-action">
        {useDropdown ? (
          <select
            id={id}
            value={value}
            required={required}
            aria-describedby={helpId}
            onChange={(event) => onChange(event.currentTarget.value)}
          >
            <option value="" disabled={required}>
              {required ? "Select a model" : inheritedLabel}
            </option>
            {options.map((model) => <option key={model} value={model}>{model}</option>)}
          </select>
        ) : (
          <input
            id={id}
            value={value}
            required={required}
            aria-describedby={helpId}
            onInput={(event) => onChange(event.currentTarget.value)}
            placeholder={required ? "Enter a model name" : inheritedLabel}
          />
        )}
        <button
          type="button"
          onClick={() => void discover()}
          disabled={loading || !profileName}
        >
          {loading ? "Discovering…" : "Discover models"}
        </button>
      </div>
      {models.length > 0 && (
        <button
          class="text-button model-picker__toggle"
          type="button"
          onClick={() => setManual((current) => !current)}
        >
          {manual ? "Choose from discovered models" : "Enter a model manually"}
        </button>
      )}
      <div id={helpId}>
        {(discoveryHint || !profileName) && (
          <p class="model-picker__hint">
            {discoveryHint || "Select a configured profile to discover models."}
          </p>
        )}
        {notice && <p class="model-picker__hint" role="status">{notice}</p>}
        {error && <p class="model-picker__hint model-picker__hint--error" role="alert">{error}</p>}
      </div>
    </div>
  );
}
