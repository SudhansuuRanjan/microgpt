import { MicroGPT } from "./model.js";

const model = new MicroGPT();

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  try {
    if (type === "TRAIN") {
      const { fileContent, numSteps } = payload;
      await model.train(
        { fileContent, numSteps },
        (log) => {
          self.postMessage({ type: "log", message: log });
        },
        (progress) => {
          self.postMessage({ type: "progress", value: progress });
        }
      );

      self.postMessage({ type: "TRAINING_COMPLETE" });
    } else if (type === "GENERATE") {
      const { temperature, numSamples } = payload;
      const names = model.generate(
        { temperature, numSamples },
        (name) => {
          // Optional: stream if needed
        }
      );
      self.postMessage({ type: "result", names });
    }
  } catch (error) {
    self.postMessage({ type: "error", message: error.message });
  }
};
