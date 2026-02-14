const worker = new Worker("worker.js", { type: "module" });

worker.onerror = (error) => {
  console.error("Worker error:", error);
  showNotification("Worker failed to start: " + error.message, "error");
  trainBtn.disabled = false;
  trainBtn.textContent = "Train Model";
};

function showNotification(message, type = "info") {
  const container = document.getElementById("notification-area");
  if (!container) return; // Safety check

  const notification = document.createElement("div");
  notification.className = `notification-toast ${type}`;
  notification.textContent = message;

  container.appendChild(notification);

  // Trigger reflow
  requestAnimationFrame(() => {
    notification.classList.add("show");
  });

  setTimeout(() => {
    notification.classList.remove("show");
    setTimeout(() => {
      notification.remove();
    }, 300);
  }, 3000);
}

let isModelTrained = false;

const fileInput = document.getElementById("fileInput");
const exampleSelect = document.getElementById("exampleSelect");
const uploadContainer = document.getElementById("upload-container");
const exampleContainer = document.getElementById("example-container");
const filePreview = document.getElementById("filePreview");
const sourceRadios = document.getElementsByName("source");

const logsEl = document.getElementById("logs");
const outputEl = document.getElementById("output");
const trainBtn = document.getElementById("trainBtn");
const generateBtn = document.getElementById("generateBtn");
const temperatureInput = document.getElementById("temperature");
const stepsInput = document.getElementById("steps");
const nEmbdInput = document.getElementById("nEmbd");
const nLayerInput = document.getElementById("nLayer");
const samplesInput = document.getElementById("samples");
const progressBar = document.getElementById("progressBar");

let currentFileContent = null;

// Toggle UI based on selection
Array.from(sourceRadios).forEach((radio) => {
  radio.addEventListener("change", (e) => {
    if (e.target.value === "upload") {
      uploadContainer.classList.remove("hidden");
      exampleContainer.classList.add("hidden");
      handleFileUpload(); // Update preview based on current file input
    } else {
      uploadContainer.classList.add("hidden");
      exampleContainer.classList.remove("hidden");
      handleExampleSelect(); // Update preview based on current example selection
    }
  });
});

// Handle File Upload
fileInput.addEventListener("change", handleFileUpload);

async function handleFileUpload() {
  if (fileInput.files.length) {
    const file = fileInput.files[0];
    const text = await file.text();
    currentFileContent = text;
    updatePreview(text);
  } else {
    currentFileContent = null;
    document.getElementById("fileStats").textContent = "";
    filePreview.textContent = "No file selected...";
  }
}

// Handle Example Selection
exampleSelect.addEventListener("change", handleExampleSelect);

async function handleExampleSelect() {
  const filename = exampleSelect.value;
  try {
    filePreview.textContent = "Loading preview...";
    const response = await fetch(`/data/${filename}`);
    if (!response.ok) throw new Error("Failed to load example file");
    const text = await response.text();
    currentFileContent = text;
    updatePreview(text);
  } catch (error) {
    console.error(error);
    filePreview.textContent = "Error loading example file.";
    currentFileContent = null;
  }
}

function updatePreview(text) {
  const lines = text.split("\n").filter((l) => l.trim().length > 0);
  const totalLines = lines.length;

  const fileStats = document.getElementById("fileStats");
  fileStats.textContent = `(${totalLines} names)`;

  const previewLines = lines.slice(0, 10);
  filePreview.textContent = previewLines.join("\n") + (totalLines > 10 ? "\n..." : "");
}

worker.onmessage = (e) => {
  const { type, message, names, value } = e.data;

  switch (type) {
    case "log":
      const logLine = document.createElement("div");
      logLine.textContent = message;
      logsEl.appendChild(logLine);
      logsEl.scrollTop = logsEl.scrollHeight;
      break;

    case "progress":
      const percent = Math.floor(value * 100);
      progressBar.style.width = percent + "%";
      if (progressText) progressText.textContent = percent + "%";
      break;

    case "TRAINING_COMPLETE":
      isModelTrained = true;
      trainBtn.disabled = false;
      trainBtn.textContent = "Train Model";
      generateBtn.disabled = false;
      progressBar.style.width = "100%";
      showNotification("Training complete 🚀", "success");


      logsEl.appendChild(document.createElement("br"));
      const logLine1 = document.createElement("div");
      logLine1.textContent = "Training complete 🚀, Model is ready to generate names";
      logsEl.appendChild(logLine1);
      logsEl.scrollTop = logsEl.scrollHeight;

      break;

    case "result":
      outputEl.textContent = names.join("\n");
      generateBtn.disabled = false;
      generateBtn.textContent = "Generate Names";
      break;

    case "error":
      showNotification(`Error: ${message}`, "error");
      trainBtn.disabled = false;
      trainBtn.textContent = "Train Model";
      generateBtn.disabled = false;
      generateBtn.textContent = "Generate Names";
      break;
  }
};


trainBtn.addEventListener("click", async (e) => {
  e.preventDefault();
  if (!currentFileContent) {
    showNotification("Please select a file or example first.", "error");
    return;
  }

  progressBar.style.width = "0%";

  logsEl.innerHTML = "";
  outputEl.textContent = "";
  isModelTrained = false; // Reset state on new training

  trainBtn.disabled = true;
  generateBtn.disabled = true;
  trainBtn.textContent = "Training...";

  // scroll window to bottom
  window.scrollTo(0, document.body.scrollHeight);

  const numSteps = parseInt(stepsInput.value);
  const nEmbd = parseInt(nEmbdInput.value);
  const nLayer = parseInt(nLayerInput.value);

  worker.postMessage({
    type: "TRAIN",
    payload: {
      fileContent: currentFileContent,
      numSteps,
      nEmbd,
      nLayer,
    },
  });
});

generateBtn.addEventListener("click", () => {
  if (!isModelTrained) {
    showNotification("Model is not trained yet!", "error");
    return;
  }

  generateBtn.disabled = true;
  generateBtn.textContent = "Generating...";
  outputEl.textContent = "Generating...";

  const temperature = parseFloat(temperatureInput.value);
  const numSamples = parseInt(samplesInput.value);

  worker.postMessage({
    type: "GENERATE",
    payload: {
      temperature,
      numSamples,
    },
  });
});

// Initialize with Upload mode or force valid state
if (fileInput.files.length) {
  handleFileUpload();
}

// View Source Logic
const viewSourceBtn = document.getElementById("viewSourceBtn");
const sourceModal = document.getElementById("sourceModal");
const closeModalBtn = document.getElementById("closeModalBtn");
const sourceCode = document.getElementById("sourceCode");

viewSourceBtn.addEventListener("click", async () => {
  sourceModal.classList.remove("hidden");
  if (sourceCode.textContent === "Loading...") {
    try {
      const response = await fetch("/model.js");
      if (!response.ok) throw new Error("Failed to load source code");
      const text = await response.text();
      sourceCode.textContent = text;
    } catch (error) {
      sourceCode.textContent = "Error loading source code: " + error.message;
    }
  }
});

closeModalBtn.addEventListener("click", () => {
  sourceModal.classList.add("hidden");
});

sourceModal.addEventListener("click", (e) => {
  if (e.target === sourceModal) {
    sourceModal.classList.add("hidden");
  }
});
