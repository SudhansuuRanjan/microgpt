import { trainAndGenerate } from "./model.js";

self.onmessage = async (e) => {
  const { fileContent, temperature, numSteps, numSamples } = e.data;

  try {
    const names = await trainAndGenerate(
      {
        fileContent,
        temperature,
        numSteps,
        numSamples,
      },
      (log) => {
        self.postMessage({ type: "log", message: log });
      },
      (name) => {
        // Optional: stream generated names one by one if needed, 
        // but currently main.js expects all at once or ignores this callback
      }
    );

    self.postMessage({ type: "result", names });
  } catch (error) {
    self.postMessage({ type: "error", message: error.message });
  }
};
