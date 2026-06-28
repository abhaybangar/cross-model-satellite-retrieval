import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./styles.css";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:5000";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isTiff, setIsTiff] = useState(false);
  const [localPreviewSrc, setLocalPreviewSrc] = useState("#");
  const [queryPreviewUrl, setQueryPreviewUrl] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [results, setResults] = useState([]);
  const [showResults, setShowResults] = useState(false);

  // Background Preprocessing State
  const [isPreprocessing, setIsPreprocessing] = useState(false);
  const [isPreprocessed, setIsPreprocessed] = useState(false);

  // Stats & Timer state
  const [showStats, setShowStats] = useState(false);
  const [statsStatus, setStatsStatus] = useState("Idle");
  const [elapsedTime, setElapsedTime] = useState("0.000");

  // Modal State
  const [modalOpen, setModalOpen] = useState(false);
  const [modalImg, setModalImg] = useState("");
  const [modalTitle, setModalTitle] = useState("");
  const [modalScore, setModalScore] = useState("");

  // Sample images state
  const [sampleImages, setSampleImages] = useState([]);
  const [displayedSamples, setDisplayedSamples] = useState([]);

  const fileInputRef = useRef(null);
  const timerRef = useRef(null);
  const startTimeRef = useRef(null);

  // Mount effect to fetch test2 samples
  useEffect(() => {
    const fetchSamples = async () => {
      try {
        const response = await axios.get(`${API_BASE}/api/test2-samples`);
        if (Array.isArray(response.data) && response.data.length > 0) {
          setSampleImages(response.data);
          // Pick 5 random ones initially
          const shuffled = [...response.data].sort(() => 0.5 - Math.random());
          setDisplayedSamples(shuffled.slice(0, 5));
        }
      } catch (err) {
        console.error("Failed to fetch sample images:", err);
      }
    };
    fetchSamples();
  }, []);

  const handleShuffle = () => {
    if (sampleImages.length === 0) return;
    const shuffled = [...sampleImages].sort(() => 0.5 - Math.random());
    setDisplayedSamples(shuffled.slice(0, 5));
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length) {
      handleFileSelect(files[0]);
    }
  };

  const triggerPreprocessing = async (file) => {
    setIsPreprocessing(true);
    setIsPreprocessed(false);
    setStatusMsg("Preprocessing image in background...");
    setErrorMsg("");

    try {
      let response;
      if (file && file.isSample) {
        response = await axios.post(`${API_BASE}/api/preprocess`, {
          image_path: file.path
        });
      } else {
        const formData = new FormData();
        formData.append("image", file);
        response = await axios.post(`${API_BASE}/api/preprocess`, formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
      }
      setIsPreprocessing(false);
      setIsPreprocessed(true);
      setStatusMsg("Preprocessing complete. Ready to retrieve.");
      
      if (response.data.queryPreview) {
        setQueryPreviewUrl(response.data.queryPreview);
      }
    } catch (err) {
      setIsPreprocessing(false);
      let msg = "Error preprocessing query image.";
      if (err.response && err.response.data) {
        msg = err.response.data.error || err.response.data.message || msg;
      } else if (err.message) {
        msg = err.message;
      }
      setErrorMsg(msg);
      setStatusMsg("");
    }
  };

  const handleFileSelect = (file) => {
    const isTiffFile =
      file.name.toLowerCase().endsWith(".tif") ||
      file.name.toLowerCase().endsWith(".tiff") ||
      file.type === "image/tiff" ||
      file.type === "image/x-tiff";
    const isImage = file.type.startsWith("image/") || isTiffFile;

    if (!isImage) {
      setErrorMsg("Invalid file type. Please upload an image file.");
      setStatusMsg("");
      return;
    }

    setSelectedFile(file);
    setIsTiff(isTiffFile);
    setQueryPreviewUrl(null);
    setErrorMsg("");
    setStatusMsg("");
    setShowStats(false);
    setElapsedTime("0.000");
    setStatsStatus("Idle");

    if (isTiffFile) {
      setLocalPreviewSrc("#");
    } else {
      const reader = new FileReader();
      reader.onload = (e) => {
        setLocalPreviewSrc(e.target.result);
      };
      reader.readAsDataURL(file);
    }

    // Trigger instant preprocessing in the background
    triggerPreprocessing(file);
  };

  const handleSelectSample = (samplePath) => {
    const filename = samplePath.split("/").pop();
    const isTiffFile = filename.toLowerCase().endsWith(".tif") || filename.toLowerCase().endsWith(".tiff");
    
    const sampleObj = {
      name: filename,
      path: samplePath,
      isSample: true
    };

    setSelectedFile(sampleObj);
    setIsTiff(isTiffFile);
    setQueryPreviewUrl(null);
    setErrorMsg("");
    setStatusMsg("");
    setShowStats(false);
    setElapsedTime("0.000");
    setStatsStatus("Idle");

    if (isTiffFile) {
      setLocalPreviewSrc("#");
    } else {
      setLocalPreviewSrc(`${API_BASE}/image?path=${encodeURIComponent(samplePath)}`);
    }

    triggerPreprocessing(sampleObj);
  };

  const triggerFileInput = () => {
    if (!selectedFile && fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files.length) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const removeSelectedFile = (e) => {
    e.stopPropagation();
    setSelectedFile(null);
    setIsTiff(false);
    setLocalPreviewSrc("#");
    setQueryPreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    setErrorMsg("");
    setStatusMsg("");
    setShowStats(false);
    setElapsedTime("0.000");
    setStatsStatus("Idle");
    setShowResults(false);
    setIsPreprocessing(false);
    setIsPreprocessed(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
    }
  };

  const executeSearch = async () => {
    if (!selectedFile) return;

    if (isPreprocessing) {
      setStatusMsg("Still preprocessing query image in background. Please wait...");
      return;
    }

    setStatusMsg("Searching SAR gallery using preprocessed query...");
    setIsSearching(true);
    setErrorMsg("");
    setShowResults(true);

    // Setup timer
    setShowStats(true);
    setStatsStatus("Scanning...");
    setElapsedTime("0.000");

    if (timerRef.current) clearInterval(timerRef.current);
    startTimeRef.current = performance.now();
    timerRef.current = setInterval(() => {
      const elapsed = ((performance.now() - startTimeRef.current) / 1000).toFixed(3);
      setElapsedTime(elapsed);
    }, 10);

    try {
      // Call search without image file payload to reuse the cached RAM embedding
      const response = await axios.post(`${API_BASE}/api/search`);

      if (timerRef.current) clearInterval(timerRef.current);
      const finalTime = ((performance.now() - startTimeRef.current) / 1000).toFixed(3);
      setElapsedTime(finalTime);
      setStatsStatus("Completed");

      setStatusMsg("");
      setIsSearching(false);

      if (response.data.queryPreview) {
        setQueryPreviewUrl(response.data.queryPreview);
      }

      setResults(response.data.results || []);
    } catch (err) {
      if (timerRef.current) clearInterval(timerRef.current);
      setStatsStatus("Failed");
      setIsSearching(false);

      let msg = "Network error while connecting to search API.";
      if (err.response && err.response.data) {
        msg = err.response.data.error || err.response.data.message || msg;
      } else if (err.message) {
        msg = err.message;
      }
      setErrorMsg(msg);
      setStatusMsg("");
    }
  };

  const openModal = (imgSrc, name, score) => {
    setModalImg(imgSrc);
    setModalTitle(name);
    setModalScore(`Match Similarity: ${score}`);
    setModalOpen(true);
  };

  const closeModal = () => {
    setModalOpen(false);
  };

  // Close modal on escape key
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        closeModal();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  return (
    <div className="app-container">
      {/* Top Header */}
      <header>
        <div className="brand">
          <img src="/logo.png" alt="DRISHTIKON Logo" className="brand-logo" />
          <div className="brand-text">
            <h1>DRISHTIKON</h1>
            <p className="subtitle-logo">Cross-Model Retrieval</p>
          </div>
        </div>

        <div className="system-status">
          <div className="status-dot"></div>
          <span>Online (Port 5000)</span>
        </div>
      </header>

      {/* Main Dashboard */}
      <div className="dashboard-grid">
        {/* Left side: Upload & Parameters */}
        <div className="panel">
          <div className="panel-title">
            <i className="fa-solid fa-upload"></i>
            <h2>Input Optical Query</h2>
          </div>

          {/* Drag & Drop Zone */}
          <div
            className={`dropzone ${isDragOver ? "dragover" : ""}`}
            onDragEnter={handleDragOver}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={triggerFileInput}
          >
            {!selectedFile && (
              <div className="dropzone-content">
                <i className="fa-regular fa-image dropzone-icon"></i>
                <p className="dropzone-text">Drag & drop your optical image</p>
                <p className="dropzone-subtext">or click to browse local files</p>
              </div>
            )}

            <input
              type="file"
              ref={fileInputRef}
              className="file-input"
              accept="image/*"
              onChange={handleFileChange}
            />

            {/* Selected Image Preview */}
            {selectedFile && (
              <div className="preview-container" style={{ display: "flex" }}>
                {!isTiff && (
                  <img
                    className="image-preview"
                    src={localPreviewSrc}
                    alt="Query preview"
                  />
                )}
                {isTiff && !queryPreviewUrl && (
                  <div className="tiff-placeholder">
                    <i className="fa-solid fa-file-image"></i>
                    <span>TIFF Image Format</span>
                    <span className="tiff-sub">No browser preview, ready to search</span>
                  </div>
                )}
                {isTiff && queryPreviewUrl && (
                  <img
                    className="image-preview"
                    src={queryPreviewUrl}
                    alt="Query preview"
                  />
                )}
                <div className="file-details">
                  <span className="file-name">{selectedFile.name}</span>
                  <button
                    className="remove-file-btn"
                    onClick={removeSelectedFile}
                    title="Remove image"
                  >
                    <i className="fa-solid fa-trash-can"></i>
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Search button */}
          <button
            className="action-btn"
            onClick={executeSearch}
            disabled={!selectedFile || isSearching}
          >
            <i className="fa-solid fa-magnifying-glass"></i>
            <span>Retrieve Matches</span>
          </button>

          {/* Status Messaging */}
          {statusMsg && (
            <div className={`status-msg ${statusMsg.includes("complete") ? "success" : statusMsg.includes("Error") || statusMsg.includes("failed") ? "error" : "loading"}`} style={{ display: "flex" }}>
              {!statusMsg.includes("complete") && !statusMsg.includes("Error") && !statusMsg.includes("failed") && <div className="spinner"></div>}
              {statusMsg.includes("complete") && <i className="fa-solid fa-circle-check" style={{ color: "var(--success)", fontSize: "1rem" }}></i>}
              {statusMsg.includes("Error") || statusMsg.includes("failed") ? <i className="fa-solid fa-circle-exclamation"></i> : null}
              <span>{statusMsg}</span>
            </div>
          )}

          {errorMsg && (
            <div className="status-msg error" style={{ display: "flex" }}>
              <i className="fa-solid fa-circle-exclamation"></i>
              <span>{errorMsg}</span>
            </div>
          )}

          {/* Stats Panel */}
          {showStats && (
            <div className="stats-panel">
              <span>Status: {statsStatus}</span>
              <span className="stats-time">{elapsedTime}s</span>
            </div>
          )}
        </div>

        {/* Right side: Search Results */}
        <div className="panel" style={{ minHeight: "480px" }}>
          <div className="results-container">
            {/* Before Search placeholder */}
            {!showResults && (
              <div className="results-placeholder">
                <i className="fa-regular fa-folder-open"></i>
                <h3>No Active Query</h3>
                <p>Upload an optical image and click retrieve to search the SAR satellite database.</p>
              </div>
            )}

            {/* Radar Scanning Animation */}
            {showResults && isSearching && (
              <div className="results-placeholder">
                <div className="radar-container">
                  <div className="radar-sweep"></div>
                  <i className="fa-solid fa-satellite-dish radar-center-icon"></i>
                </div>
                <h3>Scanning SAR Database...</h3>
                <p>Matching feature embeddings via DINOv2 engine.</p>
              </div>
            )}

            {/* Dynamic Results Container */}
            {showResults && !isSearching && (
              <div>
                <div className="results-header">
                  <div className="panel-title" style={{ marginBottom: 0 }}>
                    <i className="fa-solid fa-circle-nodes"></i>
                    <h2>SAR Retrieval Results</h2>
                  </div>
                  <span className="results-info">Found {results.length} matches</span>
                </div>

                {/* List of Results */}
                {results.length === 0 ? (
                  <div className="results-placeholder">
                    <i className="fa-regular fa-folder-open"></i>
                    <h3>No Matches Found</h3>
                    <p>No matches were found in the database. Please try another query image.</p>
                  </div>
                ) : (
                  <ul className="results-grid">
                    {results.map((item, index) => {
                      const scorePct = `${(item.score * 100).toFixed(2)}%`;
                      const displayFilename = item.filename.split("/").pop();
                      const imageSrc = `${API_BASE}/image?path=${encodeURIComponent(
                        item.filename
                      )}`;

                      return (
                        <li key={item.filename} className="result-card">
                          <div className="card-rank">{index + 1}</div>
                          <div
                            className="card-img-wrapper"
                            onClick={() =>
                              openModal(imageSrc, displayFilename, scorePct)
                            }
                          >
                            <img
                              className="card-img"
                              src={imageSrc}
                              alt={displayFilename}
                              loading="lazy"
                            />
                            <div className="card-hover-overlay">
                              <i className="fa-solid fa-magnifying-glass-plus"></i>
                            </div>
                          </div>
                          <div className="card-content">
                            <span className="card-title" title={item.filename}>
                              {displayFilename}
                            </span>
                            <div className="card-meta">
                              <span>Path: {item.filename}</span>
                            </div>
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Sample Test Queries Section */}
      <div className="samples-panel">
        <div className="samples-header">
          <div className="panel-title" style={{ marginBottom: 0 }}>
            <i className="fa-solid fa-images"></i>
            <h2>Sample Test Queries</h2>
          </div>
          <button className="shuffle-btn" onClick={handleShuffle} disabled={sampleImages.length === 0}>
            <i className="fa-solid fa-arrows-rotate"></i>
            <span>Shuffle Samples</span>
          </button>
        </div>

        {displayedSamples.length === 0 ? (
          <div style={{ textAlign: "center", color: "var(--text-muted)", padding: "1rem" }}>
            Loading sample queries...
          </div>
        ) : (
          <ul className="samples-grid">
            {displayedSamples.map((samplePath) => {
              const filename = samplePath.split("/").pop();
              const imageSrc = `${API_BASE}/image?path=${encodeURIComponent(samplePath)}`;
              const isActive = selectedFile && selectedFile.isSample && selectedFile.path === samplePath;

              return (
                <li
                  key={samplePath}
                  className={`sample-card ${isActive ? "active" : ""}`}
                  onClick={() => handleSelectSample(samplePath)}
                >
                  <div className="sample-card-img-wrapper">
                    <img
                      className="sample-card-img"
                      src={imageSrc}
                      alt={filename}
                      loading="lazy"
                    />
                  </div>
                  <div className="sample-card-content">
                    <span className="sample-card-title" title={filename}>
                      {filename}
                    </span>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {/* High-resolution Image Modal */}
      {modalOpen && (
        <div className="modal show" onClick={closeModal}>
          <div className="modal-container" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={closeModal}>
              <i className="fa-solid fa-xmark"></i>
            </button>
            <img id="modalImg" className="modal-img" src={modalImg} alt="Full preview" />
            <h3 className="modal-title">{modalTitle}</h3>
            <span className="modal-score">{modalScore}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
