interface Props {
  flower: string;
  confidence: number;
}

export function ResultCard({ flower, confidence }: Props) {
  return (
    <div style={{ textAlign: "center", marginTop: 24 }}>
      <h2
        style={{
          textTransform: "capitalize",
          fontSize: 28,
          margin: "0 0 8px",
        }}>
        {flower}
      </h2>
      <div
        style={{
          background: "#f3f4f6",
          borderRadius: 8,
          height: 12,
          overflow: "hidden",
          margin: "0 auto",
          maxWidth: 300,
        }}>
        <div
          style={{
            width: `${confidence}%`,
            height: "100%",
            background: "#4f46e5",
            transition: "width 0.6s ease",
          }}
        />
      </div>
      <p style={{ color: "#555", marginTop: 8 }}>{confidence}% confidence</p>
    </div>
  );
}
