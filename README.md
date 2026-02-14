# Micro GPT

A browser-based implementation of a miniature GPT model, ported to Javascript for browser from Andrej Karpathy's minGPT.

**Live Demo:** [https://micro-gpt.sudhanshuranjan2k18.workers.dev/](https://micro-gpt.sudhanshuranjan2k18.workers.dev/)

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── baby-names.txt
│   │   ├── indian-names.txt
│   │   ├── indian-boys-names.txt
│   │   └── pokemon-names.txt
│   ├── index.html       # Main UI
│   ├── style.css        # Styling (Dark Theme)
│   ├── main.js          # UI Logic & Worker Communication
│   ├── worker.js        # Model Training (Web Worker)
│   ├── model.js         # The GPT Model (Logic)
│   └── favicon.png      # Custom Icon
├── server.js            # Hono Server (Cloudflare Worker Entry)
└── wrangler.jsonc       # Cloudflare Configuration
```

## Local Development

1.  **Install Dependencies:**
    ```bash
    bun install
    ```

2.  **Run Locally:**
    ```bash
    bun run dev
    ```
    Access at `http://localhost:8787`
