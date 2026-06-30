const express = require("express");
const cors = require("cors");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { spawn, exec } = require("child_process");
const os = require("os");
const mongoose = require("mongoose");
const QueryLog = require("./models/QueryLog");

const app = express();
const PORT = process.env.PORT || 5000;
const rawPython = process.env.PYTHON || "../venv/Scripts/python.exe";
let resolvedPython = path.isAbsolute(rawPython) ? rawPython : path.resolve(__dirname, rawPython);
if (!process.env.PYTHON && !fs.existsSync(resolvedPython)) {
  resolvedPython = "python";
}
const PYTHON_EXEC = resolvedPython;
const uploadDir = path.join(__dirname, "uploads");
const datasetDir = path.join(__dirname, "..", "dataset");
const tempDir = os.tmpdir();

if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));
app.use("/dataset", express.static(datasetDir));

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, `${Date.now()}-${file.originalname}`),
});
const upload = multer({ storage });

const mongoUri = process.env.MONGO_URI || "";
if (mongoUri) {
  mongoose
    .connect(mongoUri, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    })
    .then(() => console.log("Connected to MongoDB"))
    .catch((err) => console.warn("MongoDB connection failed:", err.message));
} else {
  console.log("MONGO_URI is not set. MongoDB features are disabled.");
}

// Persistent FastAPI inference server process reference
let uvicornProcess = null;

function startFastApiServer() {
  if (process.env.SPAWN_FASTAPI === "false" || process.env.NODE_ENV === "production") {
    console.log("ℹ️ Spawning FastAPI inference server is disabled.");
    return;
  }
  const version = (process.env.MODEL_VERSION || "V2").toUpperCase();
  console.log(`🤖 Selected Model Version: ${version}`);
  
  let serverApp = "inference_server:app";
  if (version === "V4") {
    serverApp = "inference_server_remoteclip_lora:app";
  } else if (version === "V5") {
    serverApp = "inference_server_dinov2_lora:app";
  }
  
  console.log(`🚀 Spawning FastAPI inference server with app '${serverApp}'...`);
  const pythonScriptDir = path.join(__dirname, "python");
  
  // Start uvicorn python.inference_server:app --port 8000 --host 127.0.0.1
  uvicornProcess = spawn(
    PYTHON_EXEC,
    ["-m", "uvicorn", serverApp, "--port", "8000", "--host", "127.0.0.1"],
    { cwd: pythonScriptDir }
  );

  uvicornProcess.stdout.on("data", (data) => {
    process.stdout.write(`[FastAPI] ${data}`);
  });

  uvicornProcess.stderr.on("data", (data) => {
    process.stderr.write(`[FastAPI] ${data}`);
  });

  uvicornProcess.on("close", (code) => {
    console.log(`[FastAPI] Server process exited with code ${code}`);
  });
}

function cleanUp() {
  if (uvicornProcess) {
    console.log("🧹 Terminating FastAPI inference server...");
    uvicornProcess.kill("SIGTERM");
    uvicornProcess = null;
  }
}

// Register lifecycle cleanup handlers to avoid orphaned python processes
process.on("exit", cleanUp);
process.on("SIGINT", () => {
  cleanUp();
  process.exit();
});
process.on("SIGTERM", () => {
  cleanUp();
  process.exit();
});

app.post("/api/preprocess", upload.single("image"), async (req, res) => {
  const reqStart = performance.now();
  
  let imagePath = null;
  let isUploaded = false;
  
  if (req.file) {
    imagePath = req.file.path;
    isUploaded = true;
  } else if (req.body && req.body.image_path) {
    let resolvedPath = req.body.image_path;
    if (resolvedPath.startsWith("test2/optical/")) {
      resolvedPath = resolvedPath.replace("test2/optical/", "optical/");
    } else if (resolvedPath.startsWith("train2/optical/")) {
      resolvedPath = resolvedPath.replace("train2/optical/", "train/optical/");
    }
    imagePath = path.join(datasetDir, resolvedPath);
  }

  if (!imagePath) {
    return res.status(400).json({ error: "No image file or image path provided for preprocessing." });
  }

  try {
    const fetchStart = performance.now();
    let FASTAPI_URL = process.env.FASTAPI_URL || process.env.FAST_API_URL || process.env.FASTAPIURL || "http://127.0.0.1:8000";
    if (FASTAPI_URL.endsWith("/")) FASTAPI_URL = FASTAPI_URL.slice(0, -1);
    if (!FASTAPI_URL.startsWith("http://") && !FASTAPI_URL.startsWith("https://")) {
      FASTAPI_URL = "https://" + FASTAPI_URL;
    }
    console.log("🔗 Connecting to FastAPI at:", `${FASTAPI_URL}/preprocess`);
    
    let response;
    const formData = new FormData();
    let fileBuffer;
    let filename;
    let mimetype = "image/tiff";

    if (req.file) {
      fileBuffer = fs.readFileSync(req.file.path);
      filename = req.file.originalname;
      mimetype = req.file.mimetype;
    } else {
      fileBuffer = fs.readFileSync(imagePath);
      filename = path.basename(imagePath);
    }

    const blob = new Blob([fileBuffer], { type: mimetype });
    formData.append("file", blob, filename);

    response = await fetch(`${FASTAPI_URL}/preprocess`, {
      method: "POST",
      body: formData
    });

    const fetchEnd = performance.now();
    const networkTime = (fetchEnd - fetchStart) / 1000;

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`FastAPI server returned ${response.status}: ${errorText}`);
    }

    const result = await response.json();
    let queryPreview = result.query_preview || null;
    let tiffConversionTime = 0;

    // Clean up temporary upload file once preprocessed & cached in Python memory
    if (isUploaded && req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }

    if (queryPreview) {
      result.queryPreview = queryPreview;
    }

    const reqEnd = performance.now();
    const totalApiTime = (reqEnd - reqStart) / 1000;

    result.timings = {
      ...result.timings,
      node_network_request: parseFloat(networkTime.toFixed(4)),
      tiff_conversion: parseFloat(tiffConversionTime.toFixed(4)),
      total_api: parseFloat(totalApiTime.toFixed(4))
    };

    console.log("\n⏱️  [Express Server] Query Preprocessing Timing Breakdown:");
    console.log(`  - Image Preprocessing:     ${(result.timings.image_preprocessing || 0).toFixed(4)}s`);
    console.log(`  - Feature Extraction:      ${(result.timings.compute_query_embedding || 0).toFixed(4)}s`);
    console.log(`  -----------------------------------`);
    console.log(`  - Python Preprocess Total: ${(result.timings.total_python || 0).toFixed(4)}s`);
    console.log(`  - Express Network Roundtrip: ${result.timings.node_network_request.toFixed(4)}s`);
    if (tiffConversionTime > 0) {
      console.log(`  - TIFF-to-PNG Conversion:    ${tiffConversionTime.toFixed(4)}s`);
    }
    console.log(`  -----------------------------------`);
    console.log(`  - Total Backend Request Time: ${totalApiTime.toFixed(4)}s\n`);

    return res.json(result);
  } catch (error) {
    console.error("❌ Preprocess API Error:", error);
    if (isUploaded && req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    return res.status(500).json({ error: error.message });
  }
});

app.post("/api/search", upload.single("image"), async (req, res) => {
  const reqStart = performance.now();
  
  try {
    const fetchStart = performance.now();
    let FASTAPI_URL = process.env.FASTAPI_URL || process.env.FAST_API_URL || process.env.FASTAPIURL || "http://127.0.0.1:8000";
    if (FASTAPI_URL.endsWith("/")) FASTAPI_URL = FASTAPI_URL.slice(0, -1);
    if (!FASTAPI_URL.startsWith("http://") && !FASTAPI_URL.startsWith("https://")) {
      FASTAPI_URL = "https://" + FASTAPI_URL;
    }
    console.log("🔗 Connecting to FastAPI at:", `${FASTAPI_URL}/search`);
    
    let response;
    const formData = new FormData();
    formData.append("top_k", String(req.body.top_k || req.query.top_k || 5));

    let imagePath = null;
    if (req.body.image_path) {
      let resolvedPath = req.body.image_path;
      if (resolvedPath.startsWith("test2/optical/")) {
        resolvedPath = resolvedPath.replace("test2/optical/", "optical/");
      } else if (resolvedPath.startsWith("train2/optical/")) {
        resolvedPath = resolvedPath.replace("train2/optical/", "train/optical/");
      }
      imagePath = path.join(datasetDir, resolvedPath);
    } else if (req.query.image_path) {
      let resolvedPath = req.query.image_path;
      if (resolvedPath.startsWith("test2/optical/")) {
        resolvedPath = resolvedPath.replace("test2/optical/", "optical/");
      } else if (resolvedPath.startsWith("train2/optical/")) {
        resolvedPath = resolvedPath.replace("train2/optical/", "train/optical/");
      }
      imagePath = path.join(datasetDir, resolvedPath);
    }

    if (req.file || (imagePath && fs.existsSync(imagePath))) {
      let fileBuffer;
      let filename;
      let mimetype = "image/tiff";

      if (req.file) {
        fileBuffer = fs.readFileSync(req.file.path);
        filename = req.file.originalname;
        mimetype = req.file.mimetype;
      } else {
        fileBuffer = fs.readFileSync(imagePath);
        filename = path.basename(imagePath);
      }

      const blob = new Blob([fileBuffer], { type: mimetype });
      formData.append("file", blob, filename);

      response = await fetch(`${FASTAPI_URL}/search`, {
        method: "POST",
        body: formData
      });
    } else {
      // Fallback: search using cache without image data if no local file exists
      response = await fetch(`${FASTAPI_URL}/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ top_k: req.body.top_k || req.query.top_k || 5 })
      });
    }

    const fetchEnd = performance.now();
    const networkTime = (fetchEnd - fetchStart) / 1000;

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`FastAPI server returned ${response.status}: ${errorText}`);
    }

    const result = await response.json();
    let queryPreview = result.query_preview || null;
    let tiffConversionTime = 0;

    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }

    if (queryPreview) {
      result.queryPreview = queryPreview;
    }

    if (mongoose.connection.readyState === 1) {
      const log = new QueryLog({
        queryFilename: result.query || "preprocessed_query",
        results: result.results,
      });
      log.save().catch(() => {});
    }

    const reqEnd = performance.now();
    const totalApiTime = (reqEnd - reqStart) / 1000;
    
    // Add additional timings to the response
    result.timings = {
      ...result.timings,
      node_network_request: parseFloat(networkTime.toFixed(4)),
      tiff_conversion: parseFloat(tiffConversionTime.toFixed(4)),
      total_api: parseFloat(totalApiTime.toFixed(4))
    };

    console.log("\n⏱️  [Express Server] Query Search Timing Breakdown (FastAPI Active):");
    console.log(`  - Image Preprocessing:     ${(result.timings.image_preprocessing || 0).toFixed(4)}s`);
    console.log(`  - Feature Extraction:      ${(result.timings.compute_query_embedding || 0).toFixed(4)}s`);
    console.log(`  - Projection Head:         ${(result.timings.projection_head || 0).toFixed(4)}s`);
    console.log(`  - FAISS Search:            ${(result.timings.search_gallery || 0).toFixed(4)}s`);
    console.log(`  -----------------------------------`);
    console.log(`  - Python Inference Total:  ${(result.timings.total_python || 0).toFixed(4)}s`);
    console.log(`  - Express Network Roundtrip: ${result.timings.node_network_request.toFixed(4)}s`);
    if (tiffConversionTime > 0) {
      console.log(`  - TIFF-to-PNG Conversion:    ${tiffConversionTime.toFixed(4)}s`);
    }
    console.log(`  -----------------------------------`);
    console.log(`  - Total Backend Request Time: ${totalApiTime.toFixed(4)}s\n`);

    return res.json(result);
  } catch (error) {
    console.error("❌ Search API Error:", error);
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    return res.status(500).json({
      error: error.message
    });
  }
});

app.get("/api/logs", async (req, res) => {
  if (mongoose.connection.readyState !== 1) {
    return res.status(503).json({ error: "MongoDB is not enabled." });
  }

  const logs = await QueryLog.find().sort({ createdAt: -1 }).limit(20);
  return res.json(logs);
});

app.get("/api/status", (req, res) => {
  return res.json({ status: "ok", dataset: fs.existsSync(datasetDir) });
});

app.get("/api/test2-samples", (req, res) => {
  const test2OpticalDir = path.join(datasetDir, "optical");
  if (!fs.existsSync(test2OpticalDir)) {
    return res.status(404).json({ error: "Optical folder not found" });
  }

  try {
    const files = fs.readdirSync(test2OpticalDir)
      .filter(file => {
        const ext = path.extname(file).toLowerCase();
        return ext === ".tif" || ext === ".tiff" || ext === ".png" || ext === ".jpg" || ext === ".jpeg";
      })
      .map(file => `optical/${file}`);
    
    return res.json(files);
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
});

app.get("/image", (req, res) => {
  let imagePath = req.query.path;
  if (!imagePath) {
    return res.status(400).json({ error: "Missing path parameter" });
  }

  // Translate test2 and train2 preprocessed paths to raw folders
  if (imagePath.startsWith("test2/optical/")) {
    imagePath = imagePath.replace("test2/optical/", "optical/");
  } else if (imagePath.startsWith("test2/sar/")) {
    imagePath = imagePath.replace("test2/sar/", "sar/");
  } else if (imagePath.startsWith("train2/optical/")) {
    imagePath = imagePath.replace("train2/optical/", "train/optical/");
  } else if (imagePath.startsWith("train2/sar/")) {
    imagePath = imagePath.replace("train2/sar/", "train/sar/");
  }

  const fullPath = path.join(datasetDir, imagePath);

  if (!fullPath.startsWith(datasetDir)) {
    return res.status(403).json({ error: "Invalid path" });
  }

  if (!fs.existsSync(fullPath)) {
    return res.status(404).json({ error: "Image not found" });
  }

  const ext = path.extname(fullPath).toLowerCase();

  if (ext === ".tif" || ext === ".tiff") {
    const uniqueName = `img_${Date.now()}_${Math.random().toString(36).substr(2, 9)}.png`;
    const tempPath = path.join(tempDir, uniqueName);
    
    const python = spawn(PYTHON_EXEC, [
      "-c",
      `
from PIL import Image
import sys
try:
    img = Image.open(sys.argv[1])
    if 'sar' in sys.argv[1].lower():
        if img.mode == 'RGB':
            img = img.getchannel(0)
        else:
            img = img.convert('L')
    else:
        img = img.convert('RGB')
    img.save(sys.argv[2], 'PNG')
    print('OK')
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
`,
      fullPath,
      tempPath,
    ]);

    let stderr = "";
    python.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    python.on("close", (code) => {
      if (code !== 0) {
        console.error("Conversion failed:", stderr);
        return res.status(500).json({ error: "Failed to convert image" });
      }
      
      if (!fs.existsSync(tempPath)) {
        return res.status(500).json({ error: "Conversion output not found" });
      }
      
      res.setHeader("Content-Type", "image/png");
      const stream = fs.createReadStream(tempPath);
      stream.on("end", () => {
        fs.unlink(tempPath, (err) => {
          if (err) console.error("Failed to delete temp file:", err);
        });
      });
      stream.pipe(res);
    });
  } else {
    res.setHeader("Content-Type", `image/${ext.slice(1)}`);
    fs.createReadStream(fullPath).pipe(res);
  }
});

app.listen(PORT, () => {
  console.log(`Backend is running on http://localhost:${PORT}`);
  
  // Start FastAPI inference server process
  startFastApiServer();

  if (process.env.NODE_ENV !== "production") {
    // Automatically open browser on startup
    const url = `http://localhost:${PORT}`;
    const startCmd = process.platform === "win32" ? `start ${url}` : process.platform === "darwin" ? `open ${url}` : `xdg-open ${url}`;
    exec(startCmd, (err) => {
      if (err) console.error("Failed to open browser:", err);
    });
  }
});
