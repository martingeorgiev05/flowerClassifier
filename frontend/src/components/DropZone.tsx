import { useRef, useState } from "react";

interface Props {
  onFile: (file: File) => void;
  onError: (msg: string) => void;
  disabled: boolean;
}

const ALLOWED = ["image/jpeg", "image/png", "image/webp"];
const MAX_SIZE = 5 * 1024 * 1024;

export function DropZone({ onFile, onError, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function validate(file: File): string | null {
    if (!ALLOWED.includes(file.type))
      return "Only JPEG, PNG, and WEBP images are allowed.";
    if (file.size > MAX_SIZE) return "File must be under 5 MB.";
    return null;
  }

  function handleFile(file: File) {
    const err = validate(file);
    if (err) {
      onError(err);
      return;
    }
    onFile(file);
  }

  return (
    <div
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
      }}
      style={{
        border: `2px dashed ${dragging ? "#4f46e5" : "#ccc"}`,
        borderRadius: 12,
        padding: 40,
        textAlign: "center",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.5 : 1,
        transition: "border-color 0.2s",
      }}>
      <p>Drag & drop a flower image here, or click to select</p>
      <p style={{ fontSize: 12, color: "#888" }}>JPEG, PNG, WEBP · max 5 MB</p>
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        style={{ display: "none" }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
          e.target.value = "";
        }}
      />
    </div>
  );
}
