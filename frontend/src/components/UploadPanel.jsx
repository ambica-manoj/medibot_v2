import { useState } from "react";
import { uploadDocument } from "../api";

export default function UploadPanel({ onUploaded }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  async function handleChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const doc = await uploadDocument(file);
      onUploaded(doc);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
      e.target.value = "";
    }
  }

  return (
    <div className="upload-panel">
      <label className="upload-button">
        {busy ? "Uploading…" : "Upload PDF"}
        <input type="file" accept=".pdf" onChange={handleChange} disabled={busy} hidden />
      </label>
      {error && <p className="upload-error">{error}</p>}
    </div>
  );
}
