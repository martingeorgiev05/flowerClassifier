export interface PredictionResult {
  flower: string;
  confidence: number;
}

export async function classifyImage(file: File): Promise<PredictionResult> {
  const body = new FormData();
  body.append("file", file);

  const res = await fetch("/predict", { method: "POST", body });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }

  return res.json();
}
