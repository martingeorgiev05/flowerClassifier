import { useState } from "react";
import { classifyImage } from "./api/client";
import { DropZone } from "./components/DropZone";
import { ResultCard } from "./components/ResultCard";

type AppState =
  | { status: "idle" }
  | { status: "preview"; file: File; previewUrl: string }
  | { status: "loading"; previewUrl: string }
  | { status: "result"; previewUrl: string; flower: string; confidence: number }
  | { status: "error"; previewUrl: string | null; message: string };

export default function App() {
  const [state, setState] = useState<AppState>({ status: "idle" });

  function handleFile(file: File) {
    const previewUrl = URL.createObjectURL(file);
    setState({ status: "preview", file, previewUrl });
  }

  function handleError(message: string) {
    setState({ status: "error", previewUrl: null, message });
  }

  async function handleSubmit() {
    if (state.status !== "preview") return;
    const { file, previewUrl } = state;
    setState({ status: "loading", previewUrl });
    try {
      const result = await classifyImage(file);
      setState({ status: "result", previewUrl, ...result });
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : "Something went wrong";
      setState({ status: "error", previewUrl, message });
    }
  }

  function reset() {
    if ("previewUrl" in state && state.previewUrl)
      URL.revokeObjectURL(state.previewUrl);
    setState({ status: "idle" });
  }

  const previewUrl = "previewUrl" in state ? state.previewUrl : null;

  return (
    <div
      style={{
        maxWidth: 480,
        margin: "60px auto",
        padding: "0 16px",
        fontFamily: "system-ui, sans-serif",
      }}>
      <h1 style={{ textAlign: "center", marginBottom: 32 }}>
        🌸 Flower Classifier
      </h1>

      <DropZone
        onFile={handleFile}
        onError={handleError}
        disabled={state.status === "loading"}
      />

      {previewUrl && (
        <img
          src={previewUrl}
          alt="preview"
          style={{
            display: "block",
            margin: "24px auto 0",
            maxWidth: "100%",
            maxHeight: 300,
            borderRadius: 8,
            objectFit: "cover",
          }}
        />
      )}

      {state.status === "preview" && (
        <button onClick={handleSubmit} style={btnStyle}>
          Classify
        </button>
      )}

      {state.status === "loading" && (
        <p style={{ textAlign: "center", marginTop: 24, color: "#555" }}>
          Classifying...
        </p>
      )}

      {state.status === "result" && (
        <>
          <ResultCard flower={state.flower} confidence={state.confidence} />
          <button onClick={reset} style={btnStyle}>
            Try another
          </button>
        </>
      )}

      {state.status === "error" && (
        <>
          <p style={{ textAlign: "center", color: "#dc2626", marginTop: 16 }}>
            {state.message}
          </p>
          <button onClick={reset} style={btnStyle}>
            Try again
          </button>
        </>
      )}
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  display: "block",
  margin: "20px auto 0",
  padding: "10px 32px",
  background: "#4f46e5",
  color: "#fff",
  border: "none",
  borderRadius: 8,
  fontSize: 16,
  cursor: "pointer",
};
