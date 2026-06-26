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
const PYTHON_EXEC = process.env.PYTHON || "python";
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
  console.log("🚀 Spawning FastAPI inference server...");
  const pythonScriptDir = path.join(__dirname, "python");
  
  const resolvedPython = path.resolve(__dirname, PYTHON_EXEC);
  
  // Start uvicorn python.inference_server:app --port 8000 --host 127.0.0.1
  uvicornProcess = spawn(
    resolvedPython,
    ["-m", "uvicorn", "inference_server:app", "--port", "8000", "--host", "127.0.0.1"],
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
  if (!req.file) {
    return res.status(400).json({ error: "No image file provided for preprocessing." });
  }

  try {
    const fetchStart = performance.now();
    
    // Call FastAPI /preprocess
    const response = await fetch("http://127.0.0.1:8000/preprocess", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        image_path: req.file.path
      })
    });

    const fetchEnd = performance.now();
    const networkTime = (fetchEnd - fetchStart) / 1000;

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`FastAPI server returned ${response.status}: ${errorText}`);
    }

    const result = await response.json();

    // Convert query image to base64 preview if it is TIFF
    const ext = path.extname(req.file.path).toLowerCase();
    let queryPreview = null;
    let tiffConversionTime = 0;
    if (ext === ".tif" || ext === ".tiff") {
      const convertStart = performance.now();
      const uniqueName = `query_preview_${Date.now()}.png`;
      const tempPath = path.join(tempDir, uniqueName);
      
      const pythonConvert = spawn(PYTHON_EXEC, [
        "-c",
        `
from PIL import Image
import sys
try:
    img = Image.open(sys.argv[1]).convert('RGB')
    img.save(sys.argv[2], 'PNG')
    print('OK')
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
`,
        req.file.path,
        tempPath,
      ]);
      
      await new Promise((resolve) => {
        pythonConvert.on("close", (convertCode) => {
          if (convertCode === 0 && fs.existsSync(tempPath)) {
            try {
              const fileBuffer = fs.readFileSync(tempPath);
              queryPreview = `data:image/png;base64,${fileBuffer.toString("base64")}`;
              fs.unlinkSync(tempPath);
            } catch (e) {
              console.error("Read temp file error:", e);
            }
          }
          resolve();
        });
      });
      tiffConversionTime = (performance.now() - convertStart) / 1000;
    }

    // Clean up temporary upload file once preprocessed & cached in Python memory
    if (req.file && fs.existsSync(req.file.path)) {
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
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    return res.status(500).json({ error: error.message });
  }
});

app.post("/api/search", upload.single("image"), async (req, res) => {
  const reqStart = performance.now();
  
  try {
    const fetchStart = performance.now();
    const bodyPayload = { top_k: 5 };
    if (req.file) {
      bodyPayload.image_path = req.file.path;
    }
    
    // Call the persistent FastAPI server search endpoint
    const response = await fetch("http://127.0.0.1:8000/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(bodyPayload)
    });

    const fetchEnd = performance.now();
    const networkTime = (fetchEnd - fetchStart) / 1000;

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`FastAPI server returned ${response.status}: ${errorText}`);
    }

    const result = await response.json();

    // Convert query image to base64 preview if it is TIFF and we uploaded one
    let queryPreview = null;
    let tiffConversionTime = 0;
    if (req.file) {
      const ext = path.extname(req.file.path).toLowerCase();
      if (ext === ".tif" || ext === ".tiff") {
        const convertStart = performance.now();
        const uniqueName = `query_preview_${Date.now()}.png`;
        const tempPath = path.join(tempDir, uniqueName);
        
        const pythonConvert = spawn(PYTHON_EXEC, [
          "-c",
          `
from PIL import Image
import sys
try:
    img = Image.open(sys.argv[1]).convert('RGB')
    img.save(sys.argv[2], 'PNG')
    print('OK')
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
`,
          req.file.path,
          tempPath,
        ]);
        
        await new Promise((resolve) => {
          pythonConvert.on("close", (convertCode) => {
            if (convertCode === 0 && fs.existsSync(tempPath)) {
              try {
                const fileBuffer = fs.readFileSync(tempPath);
                queryPreview = `data:image/png;base64,${fileBuffer.toString("base64")}`;
                fs.unlinkSync(tempPath);
              } catch (e) {
                console.error("Read temp file error:", e);
              }
            }
            resolve();
          });
        });
        tiffConversionTime = (performance.now() - convertStart) / 1000;
      }

      if (fs.existsSync(req.file.path)) {
        fs.unlinkSync(req.file.path);
      }
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

app.get("/image", (req, res) => {
  const imagePath = req.query.path;
  if (!imagePath) {
    return res.status(400).json({ error: "Missing path parameter" });
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
    img = Image.open(sys.argv[1]).convert('RGB')
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

  // Automatically open browser on startup
  const url = `http://localhost:${PORT}`;
  const startCmd = process.platform === "win32" ? `start ${url}` : process.platform === "darwin" ? `open ${url}` : `xdg-open ${url}`;
  exec(startCmd, (err) => {
    if (err) console.error("Failed to open browser:", err);
  });
});
