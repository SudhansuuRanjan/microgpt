const worker = new Worker("worker.js", { type: "module" });

const fileInput = document.getElementById("fileInput");
const exampleSelect = document.getElementById("exampleSelect");
const uploadContainer = document.getElementById("upload-container");
const exampleContainer = document.getElementById("example-container");
const filePreview = document.getElementById("filePreview");
const sourceRadios = document.getElementsByName("source");

const logsEl = document.getElementById("logs");
const outputEl = document.getElementById("output");
const trainBtn = document.getElementById("trainBtn");
const temperatureInput = document.getElementById("temperature");
const stepsInput = document.getElementById("steps");
const samplesInput = document.getElementById("samples");

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
  const { type, message, names } = e.data;

  if (type === "log") {
    const p = document.createElement("div"); // Changed to div for better styling control if needed
    p.textContent = message;
    logsEl.appendChild(p);
    logsEl.scrollTop = logsEl.scrollHeight;
  } else if (type === "result") {
    outputEl.textContent = names.join("\n");
    trainBtn.disabled = false;
    trainBtn.textContent = "Train & Generate";
  } else if (type === "error") {
    alert(`Error: ${message}`);
    trainBtn.disabled = false;
    trainBtn.textContent = "Train & Generate";
  }
};

trainBtn.addEventListener("click", async () => {
  logsEl.innerHTML = "";
  outputEl.textContent = "";

  if (!currentFileContent) {
    alert("Please select a file or example first.");
    return;
  }

  trainBtn.disabled = true;
  trainBtn.textContent = "Training...";

  const temperature = parseFloat(temperatureInput.value);
  const numSteps = parseInt(stepsInput.value);
  const numSamples = parseInt(samplesInput.value);

  worker.postMessage({
    fileContent: currentFileContent,
    temperature,
    numSteps,
    numSamples,
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
