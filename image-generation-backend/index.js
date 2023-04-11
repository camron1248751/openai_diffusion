const express = require("express");
const cors = require("cors");
const morgan = require("morgan");
const { PythonShell } = require("python-shell");
const path = require("path");
const multer = require("multer");
const http = require("http");
const { Server } = require("socket.io");
const fs = require("fs");

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/");
  },
  filename: (req, file, cb) => {
    cb(null, file.fieldname + "-" + Date.now() + path.extname(file.originalname));
  },
});

const upload = multer({ storage: storage });

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*",
  },
});

app.use(cors());
app.use(express.json());
app.use(morgan("dev"));

app.use("/uploads", express.static(path.join(__dirname, "uploads")));

app.get("/uploads/list", (req, res) => {
  fs.readdir(path.join(__dirname, "uploads"), (err, files) => {
    if (err) {
      console.error(err);
      res.status(500).send("Error reading uploads folder.");
    } else {
      res.json(files);
    }
  });
});

app.post("/generate", upload.single("uploadedImage"), async (req, res) => {
  if (!req.file) {
    return res.status(400).send("No image file uploaded");
  }

  // Call the guided_generation function with the uploaded image
  let options = {
    mode: "text",
    pythonOptions: ["-u"], // get print results in real-time
    scriptPath: "../scripts/",
    args: ["--input_image_path", req.file.path],
  };

  const pythonShellInstance = new PythonShell("gen_function.py", options);

  pythonShellInstance.on("message", (message) => {
   io.emit("progress_update", message);
  });

  // pythonShellInstance.end(function (err, results) {
  //   if (err) {
  //     console.error(err);
  //     return res.status(500).send("An error occurred while generating images");
  //   }

  //   // Extract the generated filenames from the Python script output
  //   const generatedFilenames = JSON.parse(results[results.length - 1]);

  //   // Send the generated filenames back to the client
  //   res.json({ generatedFilenames });
  // });
  PythonShell.run("gen_function.py", options).then(results => {
    const generatedFilenames = JSON.parse(results[results.length - 1]);
    res.json({ generatedFilenames })
  })
});

io.on("connection", (socket) => {
  console.log("a user connected");

  socket.on("disconnect", () => {
    console.log("user disconnected");
  });
});

const port = process.env.PORT || 5000;

server.listen(port, () => console.log(`Server listening on port ${port}`));
