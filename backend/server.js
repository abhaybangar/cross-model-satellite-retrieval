const express = require("express");
const cors = require("cors");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
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

function parseJsonFromOutput(output) {
  const start = output.indexOf("{");
  const end = output.lastIndexOf("}");
  if (start === -1 || end === -1) {
    throw new Error("Could not parse JSON output from Python script.");
  }
  return JSON.parse(output.slice(start, end + 1));
}

app.post("/api/search", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "Upload an image file to search." });
  }

  const pythonScript = path.join(__dirname, "python", "search.py");
  const child = spawn(PYTHON_EXEC, [pythonScript, "--query", req.file.path, "--top_k", "5"]);

  let stdout = "";
  let stderr = "";

  child.stdout.on("data", (chunk) => {
    stdout += chunk.toString();
  });

  child.stderr.on("data", (chunk) => {
    stderr += chunk.toString();
  });

  child.on("close", async (code) => {
    try {
      if (req.file && fs.existsSync(req.file.path)) {
        fs.unlinkSync(req.file.path);
      }

      if (code !== 0) {
        return res.status(500).json({ error: stderr || `Python script exited with code ${code}` });
      }

      const result = parseJsonFromOutput(stdout);

      if (mongoose.connection.readyState === 1) {
        const log = new QueryLog({
          queryFilename: req.file.originalname,
          results: result.results,
        });
        log.save().catch(() => {});
      }

      return res.json(result);
    } catch (error) {
      return res.status(500).json({
        error: error.message,
        stdout: stdout.trim(),
        stderr: stderr.trim(),
      });
    }
  });
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
});
