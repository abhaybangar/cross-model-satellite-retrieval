import React, { useState } from "react";
import axios from "axios";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:5000";

function App() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setResults([]);
    setError("");
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please choose an image before uploading.");
      return;
    }

    setStatus("Uploading image and searching...");
    setError("");

    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await axios.post(`${API_BASE}/api/search`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResults(response.data.results);
      setStatus("Search completed.");
    } catch (err) {
      setError(err.response?.data?.error || err.message);
      setStatus("");
    }
  };

  return (
    <div className="app-shell">
      <header>
        <h1>Optical Image Search</h1>
        <p>Upload an optical image to find matching images from the SAR gallery.</p>
      </header>
      <main>
        <section className="upload-panel">
          <input type="file" accept="image/*" onChange={handleFileChange} />
          <button onClick={handleUpload}>Search</button>
          {status && <p className="status">{status}</p>}
          {error && <p className="error">{error}</p>}
        </section>

        {results.length > 0 && (
          <section className="results-panel">
            <h2>Top Matches</h2>
            <ul>
              {results.map((item) => (
                <li key={item.filename}>
                  <div>
                    <strong>{item.filename}</strong>
                    <span>score: {(item.score * 100).toFixed(2)}%</span>
                  </div>
                  <img
                    src={`${API_BASE}/dataset/${item.filename}`}
                    alt={item.filename}
                    loading="lazy"
                  />
                </li>
              ))}
            </ul>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
