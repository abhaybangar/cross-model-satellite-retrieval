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
  const [activeTab, setActiveTab] = useState("search");

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


      </header>

      {/* Navigation Tabs */}
      <nav className="nav-tabs">
        <button
          className={`nav-tab-btn ${activeTab === "search" ? "active" : ""}`}
          onClick={() => setActiveTab("search")}
        >
          <i className="fa-solid fa-magnifying-glass"></i>
          Search Dashboard
        </button>
        <button
          className={`nav-tab-btn ${activeTab === "about" ? "active" : ""}`}
          onClick={() => setActiveTab("about")}
        >
          <i className="fa-solid fa-chart-line"></i>
          Model Analysis
        </button>
      </nav>

      {/* Tab 1: Search Dashboard */}
      {activeTab === "search" && (
        <>
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
                              <div className="card-score-badge">{scorePct}</div>
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
        </>
      )}

      {/* Tab 2: Model Analysis */}
      {activeTab === "about" && (
        <div className="about-container">
          <div className="about-header">
            <h2>Cross-Modal Latent Space Alignment</h2>
            <p>
              Dual-pathway deep contrastive retrieval architecture for bridging Sentinel-2 Multispectral and Sentinel-1 SAR observations.
            </p>
          </div>

          <div className="about-grid">
            {/* Left Column: Physics & Math */}
            <div className="about-section">
              <div className="about-card">
                <h3><i className="fa-solid fa-satellite"></i> 1. Sensor Physics & Modality Gap</h3>
                <p>
                  Cross-modal satellite retrieval is challenging due to the fundamentally different physical phenomena captured by optical and microwave sensors:
                </p>
                <div className="badge-row">
                  <span className="physics-badge optical">Sentinel-2 MSI (Optical)</span>
                  <span className="physics-badge sar">Sentinel-1 C-Band SAR (Radar)</span>
                </div>
                <p>
                  <strong>Optical Reflectance (MSI)</strong> measures solar radiance reflected off the Earth's surface across the visual, near-infrared, and shortwave infrared spectral bands (10m–60m resolution). It records the chemical composition of targets (e.g. chlorophyll absorption in vegetation canopies).
                </p>
                <p>
                  <strong>Active Microwave Scattering (SAR)</strong> operates at C-band (5.405 GHz) and measures surface dielectric properties (such as soil moisture and liquid water content) and physical geometry (roughness, orientation, and volume structure).
                </p>
                <ul>
                  <li><i className="fa-solid fa-circle-check"></i> <strong>VV Polarization:</strong> Highly sensitive to surface specular reflection (water bodies) and surface roughness.</li>
                  <li><i className="fa-solid fa-circle-check"></i> <strong>VH Polarization:</strong> Sensitive to depolarization caused by volume scattering (forest canopy structure, agricultural biomass).</li>
                  <li><i className="fa-solid fa-circle-check"></i> <strong>VV-VH Ratio:</strong> Stretched as a 3rd channel [VV, VH, VV-VH] to isolate vegetation thickness and moisture levels.</li>
                </ul>
              </div>

              <div className="about-card">
                <h3><i className="fa-solid fa-calculator"></i> 2. Joint Contrastive Optimization</h3>
                <p>
                  Because optical and SAR inputs reside in disjoint spaces, we align them into a shared latent metric space using a <strong>Symmetric InfoNCE Contrastive Loss</strong>. The model is trained on matched pairs to maximize the cosine similarity of true targets while minimizing it for mismatched pairs in the batch.
                </p>
                <p>
                  For a batch of N coincident optical-SAR pairs, the contrastive loss function L is defined as:
                </p>
                <div className="math-block">
                  L = -<span className="math-font">1</span>/<span className="math-font">2N</span> &Sigma;<span className="math-sub">i=1</span><span className="math-sup">N</span> [ log( e<span className="math-sup">sim(u<span className="math-sub">i</span>, v<span className="math-sub">i</span>)/&tau;</span> / &Sigma;<span className="math-sub">j=1</span><span className="math-sup">N</span> e<span className="math-sup">sim(u<span className="math-sub">i</span>, v<span className="math-sub">j</span>)/&tau;</span> ) + log( e<span className="math-sup">sim(u<span className="math-sub">i</span>, v<span className="math-sub">i</span>)/&tau;</span> / &Sigma;<span className="math-sub">j=1</span><span className="math-sup">N</span> e<span className="math-sup">sim(u<span className="math-sub">j</span>, v<span className="math-sub">i</span>)/&tau;</span> ) ]
                </div>
                <p>
                  Where:
                </p>
                <ul>
                  <li><i className="fa-solid fa-circle-check"></i> <strong>u<span className="math-sub">i</span>, v<span className="math-sub">i</span>:</strong> L2-normalized 256-D projected vectors for optical and SAR respectively.</li>
                  <li><i className="fa-solid fa-circle-check"></i> <strong>sim(u, v):</strong> Cosine similarity given by the dot product u &middot; v.</li>
                  <li><i className="fa-solid fa-circle-check"></i> <strong>&tau;:</strong> Learnable temperature parameter scaling similarity logits.</li>
                </ul>
              </div>
            </div>

            {/* Right Column: Pipeline & Performance */}
            <div className="about-section">
              <div className="about-card">
                <h3><i className="fa-solid fa-network-wired"></i> 3. Dual-Pathway Encoder Architecture</h3>
                <p>
                  The system leverages a frozen self-supervised foundation model, <strong>DINOv2 (ViT-B/14)</strong>, to extract general spatial feature embeddings (768-D). Modality alignment is achieved by training lightweight, non-linear MLP Projection Heads.
                </p>
                
                <div className="diagram-container">
                  <svg viewBox="0 0 400 240" className="svg-content">
                    {/* Optical Pathway */}
                    <rect x="10" y="20" width="100" height="30" rx="5" fill="#3b82f6" fillOpacity="0.2" stroke="#3b82f6" strokeWidth="1"/>
                    <text x="60" y="38" textAnchor="middle" className="svg-title" fontSize="10">Optical Image</text>
                    
                    <path d="M 60 50 L 60 80" className="svg-arrow" stroke="var(--text-muted)" strokeWidth="1.5" fill="none"/>
                    
                    <rect x="10" y="80" width="100" height="30" rx="5" fill="rgba(79, 70, 229, 0.2)" stroke="var(--accent)" strokeWidth="1"/>
                    <text x="60" y="98" textAnchor="middle" className="svg-title" fontSize="10">DINOv2 (768-D)</text>
                    
                    <path d="M 60 110 L 60 140" className="svg-arrow" stroke="var(--text-muted)" strokeWidth="1.5" fill="none"/>
                    
                    <rect x="10" y="140" width="100" height="30" rx="5" fill="rgba(16, 185, 129, 0.2)" stroke="var(--success)" strokeWidth="1"/>
                    <text x="60" y="158" textAnchor="middle" className="svg-title" fontSize="10">Projection Head</text>
                    
                    <path d="M 60 170 L 60 200" className="svg-arrow" stroke="var(--text-muted)" strokeWidth="1.5" fill="none"/>
                    
                    <rect x="10" y="200" width="100" height="30" rx="5" fill="rgba(255, 255, 255, 0.05)" stroke="var(--text-secondary)" strokeWidth="1"/>
                    <text x="60" y="218" textAnchor="middle" className="svg-title" fontSize="10">Opt. Vector (256-D)</text>
                    
                    {/* SAR Pathway */}
                    <rect x="290" y="20" width="100" height="30" rx="5" fill="#fe5e03" fillOpacity="0.2" stroke="#fe5e03" strokeWidth="1"/>
                    <text x="340" y="38" textAnchor="middle" className="svg-title" fontSize="10">SAR Image</text>
                    
                    <path d="M 340 50 L 340 80" className="svg-arrow" stroke="var(--text-muted)" strokeWidth="1.5" fill="none"/>
                    
                    <rect x="290" y="80" width="100" height="30" rx="5" fill="rgba(79, 70, 229, 0.2)" stroke="var(--accent)" strokeWidth="1"/>
                    <text x="340" y="98" textAnchor="middle" className="svg-title" fontSize="10">DINOv2 (768-D)</text>
                    
                    <path d="M 340 110 L 340 140" className="svg-arrow" stroke="var(--text-muted)" strokeWidth="1.5" fill="none"/>
                    
                    <rect x="290" y="140" width="100" height="30" rx="5" fill="rgba(16, 185, 129, 0.2)" stroke="var(--success)" strokeWidth="1"/>
                    <text x="340" y="158" textAnchor="middle" className="svg-title" fontSize="10">Projection Head</text>
                    
                    <path d="M 340 170 L 340 200" className="svg-arrow" stroke="var(--text-muted)" strokeWidth="1.5" fill="none"/>
                    
                    <rect x="290" y="200" width="100" height="30" rx="5" fill="rgba(255, 255, 255, 0.05)" stroke="var(--text-secondary)" strokeWidth="1"/>
                    <text x="340" y="218" textAnchor="middle" className="svg-title" fontSize="10">SAR Vector (256-D)</text>
                    
                    {/* Middle Similarity */}
                    <circle cx="200" cy="215" r="20" fill="rgba(254, 94, 3, 0.15)" stroke="#fe5e03" strokeWidth="1.5"/>
                    <text x="200" y="219" textAnchor="middle" fontSize="10" fontWeight="800" fill="#fe5e03">Dot</text>
                    
                    <path d="M 110 215 L 180 215" stroke="var(--text-muted)" strokeWidth="1.5" strokeDasharray="3,3"/>
                    <path d="M 290 215 L 220 215" stroke="var(--text-muted)" strokeWidth="1.5" strokeDasharray="3,3"/>
                    
                    {/* InfoNCE Link in training */}
                    <rect x="140" y="125" width="120" height="40" rx="5" fill="rgba(255, 255, 255, 0.03)" stroke="var(--border)" strokeWidth="1"/>
                    <text x="200" y="142" textAnchor="middle" fontSize="10" fontWeight="700" fill="var(--text-primary)">InfoNCE Loss</text>
                    <text x="200" y="153" textAnchor="middle" fontSize="8" fill="var(--text-secondary)">Symmetric Contrastive</text>
                    
                    <path d="M 110 145 L 140 145" stroke="var(--accent)" strokeWidth="1"/>
                    <path d="M 290 145 L 260 145" stroke="var(--accent)" strokeWidth="1"/>
                  </svg>
                </div>
              </div>

              <div className="about-card">
                <h3><i className="fa-solid fa-chart-simple"></i> 4. Training Progression & Benchmark Results</h3>
                <p>
                  Training of the projection heads was carried out on paired historical datasets over 60 epochs. The validation accuracy was tracked on the unseen <strong>test2</strong> dataset, verifying robust cross-sensor generalization:
                </p>

                <div className="about-metrics-grid">
                  <div className="about-metric-card">
                    <div className="metric-val">38.0%</div>
                    <div className="metric-label">Recall@1</div>
                  </div>
                  <div className="about-metric-card">
                    <div className="metric-val">84.0%</div>
                    <div className="metric-label">Recall@10</div>
                  </div>
                  <div className="about-metric-card">
                    <div className="metric-val">211ms</div>
                    <div className="metric-label">Inference Latency</div>
                  </div>
                </div>

                <div className="diagram-container">
                  <svg viewBox="0 0 400 220" className="svg-content">
                    {/* Grid Lines */}
                    <line x1="40" y1="20" x2="360" y2="20" stroke="rgba(255,255,255,0.05)" strokeWidth="1"/>
                    <line x1="40" y1="60" x2="360" y2="60" stroke="rgba(255,255,255,0.05)" strokeWidth="1"/>
                    <line x1="40" y1="100" x2="360" y2="100" stroke="rgba(255,255,255,0.05)" strokeWidth="1"/>
                    <line x1="40" y1="140" x2="360" y2="140" stroke="rgba(255,255,255,0.05)" strokeWidth="1"/>
                    <line x1="40" y1="180" x2="360" y2="180" stroke="rgba(255,255,255,0.08)" strokeWidth="1.5"/>
                    
                    {/* Axes Labels (Left: Loss) */}
                    <text x="18" y="24" className="svg-label" textAnchor="end">4.0</text>
                    <text x="18" y="64" className="svg-label" textAnchor="end">3.0</text>
                    <text x="18" y="104" className="svg-label" textAnchor="end">2.0</text>
                    <text x="18" y="144" className="svg-label" textAnchor="end">1.0</text>
                    <text x="18" y="184" className="svg-label" textAnchor="end">0.0</text>
                    <text x="5" y="100" className="svg-label" transform="rotate(-90 5 100)" textAnchor="middle" fill="#ef4444">Loss</text>
                    
                    {/* Axes Labels (Right: Accuracy) */}
                    <text x="382" y="24" className="svg-label" textAnchor="start">40%</text>
                    <text x="382" y="64" className="svg-label" textAnchor="start">30%</text>
                    <text x="382" y="104" className="svg-label" textAnchor="start">20%</text>
                    <text x="382" y="144" className="svg-label" textAnchor="start">10%</text>
                    <text x="382" y="184" className="svg-label" textAnchor="start">0%</text>
                    <text x="395" y="100" className="svg-label" transform="rotate(90 395 100)" textAnchor="middle" fill="#10b981">Accuracy</text>
                    
                    {/* X Axis Labels (Epochs) */}
                    <text x="40" y="198" className="svg-label" textAnchor="middle">1</text>
                    <text x="104" y="198" className="svg-label" textAnchor="middle">10</text>
                    <text x="168" y="198" className="svg-label" textAnchor="middle">20</text>
                    <text x="232" y="198" className="svg-label" textAnchor="middle">30</text>
                    <text x="296" y="198" className="svg-label" textAnchor="middle">40</text>
                    <text x="360" y="198" className="svg-label" textAnchor="middle">60</text>
                    <text x="200" y="214" className="svg-label" textAnchor="middle">Training Epochs</text>
                    
                    {/* Loss Path (Red) */}
                    <path d="M 40 20 L 93.3 52 L 146.6 108 L 200 140 L 253.3 156 L 360 164" fill="none" stroke="#ef4444" strokeWidth="2.5" strokeLinecap="round"/>
                    
                    {/* Accuracy Path (Green) */}
                    <path d="M 40 166 L 93.3 132 L 146.6 92 L 200 64 L 253.3 40 L 360 28" fill="none" stroke="#10b981" strokeWidth="2.5" strokeLinecap="round"/>
                    
                    {/* Data Points */}
                    <circle cx="360" cy="164" r="4" fill="#ef4444"/>
                    <circle cx="360" cy="28" r="4" fill="#10b981"/>
                  </svg>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

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
