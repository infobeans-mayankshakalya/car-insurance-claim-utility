import React, { useState } from "react";

export default function CarDamageAnalyzer() {
  const [file, setFile] = useState(null);
  const [make, setMake] = useState("");
  const [model, setModel] = useState("");
  const [year, setYear] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [imagePreview, setImagePreview] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !make || !model || !year) {
      alert("Please fill all fields and select an image");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("make", make);
    formData.append("model", model);
    formData.append("year", year);

    setLoading(true);
    try {
      setImagePreview(URL.createObjectURL(file));

      const res = await fetch("http://localhost:8000/inference/local", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error("Error uploading image:", err);
      alert("Error uploading image");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 700, margin: "30px auto", fontFamily: "Arial, sans-serif" }}>
      <h1 style={{ textAlign: "center", marginBottom: 20, color: "#333" }}>Car Damage Analyzer</h1>

      <div
        style={{
          background: "#fff",
          padding: 20,
          borderRadius: 12,
          boxShadow: "0 4px 10px rgba(0,0,0,0.1)",
        }}
      >
        <form
          onSubmit={handleSubmit}
          style={{ display: "grid", gap: 15, marginBottom: 20 }}
        >
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setFile(e.target.files[0])}
            style={{ padding: 10, borderRadius: 6, border: "1px solid #ccc" }}
          />
          <input
            type="text"
            placeholder="Make"
            value={make}
            onChange={(e) => setMake(e.target.value)}
            style={{ padding: 10, borderRadius: 6, border: "1px solid #ccc" }}
          />
          <input
            type="text"
            placeholder="Model"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            style={{ padding: 10, borderRadius: 6, border: "1px solid #ccc" }}
          />
          <input
            type="number"
            placeholder="Year"
            value={year}
            onChange={(e) => setYear(e.target.value)}
            style={{ padding: 10, borderRadius: 6, border: "1px solid #ccc" }}
          />
          <button
            type="submit"
            disabled={loading}
            style={{
              padding: 12,
              borderRadius: 6,
              border: "none",
              backgroundColor: "#007bff",
              color: "#fff",
              fontWeight: "bold",
              cursor: loading ? "not-allowed" : "pointer",
            }}
          >
            {loading ? "Analyzing..." : "Submit"}
          </button>
        </form>

        {result && imagePreview && (
          <div
            style={{
              marginTop: 20,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            <div
              style={{
                position: "relative",
                display: "inline-block",
                borderRadius: 12,
                overflow: "hidden",
                boxShadow: "0 4px 10px rgba(0,0,0,0.1)",
              }}
            >
              <img
                src={imagePreview}
                alt="Car Damage"
                style={{ maxWidth: "100%", display: "block" }}
              />
              <div
                style={{
                  position: "absolute",
                  top: 10,
                  left: 10,
                  backgroundColor: "rgba(255,0,0,0.7)",
                  color: "#fff",
                  padding: "6px 12px",
                  borderRadius: 6,
                  fontWeight: "bold",
                  fontSize: 14,
                }}
              >
                {result.damage_label} ({(result.confidence * 100).toFixed(1)}%) - {result.severity}
              </div>
            </div>

            <div
              style={{
                background: "#f8f9fa",
                marginTop: 15,
                padding: 15,
                borderRadius: 10,
                width: "100%",
                textAlign: "left",
                boxShadow: "0 2px 6px rgba(0,0,0,0.05)",
              }}
            >
              <p>
                <strong>Estimated Repair Cost:</strong> ₹{result.cost_estimate}
              </p>
              <p>
                <strong>Car:</strong> {result.car.make} {result.car.model} ({result.car.year})
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
