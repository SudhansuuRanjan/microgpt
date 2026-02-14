// Micro GPT - Paragraph Version (Browser Compatible)

import random from "./random.js";

export async function trainAndGenerate(options, log, emitText) {
  const { fileContent, temperature, numSamples, numSteps } = options;

  random.seed(42);

  const text = fileContent.trim();
  log(`Loaded ${text.length} characters`);

  // ----------------------------
  // TOKENIZER (character-level)
  // ----------------------------
  const uchars = [...new Set(text)].sort();
  const char_to_id = new Map(uchars.map((ch, i) => [ch, i]));
  const BOS = uchars.length;
  const vocab_size = uchars.length + 1;

  log(`vocab size: ${vocab_size}`);

  const tokens_full = Array.from(text, (ch) => char_to_id.get(ch));

  // ----------------------------
  // MODEL SIZE (Bigger than name version)
  // ----------------------------
  const n_embd = 16;
  const n_head = 2;
  const n_layer = 1;
  const block_size = 32;
  const head_dim = Math.floor(n_embd / n_head);
  const scale = 1 / Math.sqrt(head_dim);

  // ----------------------------
  // AUTOGRAD ENGINE
  // ----------------------------
  let _gen = 0;

  class Value {
    constructor(data, children = [], grads = []) {
      this.data = data;
      this.grad = 0;
      this._c = children;
      this._g = grads;
      this._gen = 0;
    }

    add(o) {
      if (o instanceof Value)
        return new Value(this.data + o.data, [this, o], [1, 1]);
      return new Value(this.data + o, [this], [1]);
    }

    mul(o) {
      if (o instanceof Value)
        return new Value(this.data * o.data, [this, o], [o.data, this.data]);
      return new Value(this.data * o, [this], [o]);
    }

    pow(n) {
      return new Value(this.data ** n, [this], [n * this.data ** (n - 1)]);
    }

    exp() {
      const e = Math.exp(this.data);
      return new Value(e, [this], [e]);
    }

    log() {
      return new Value(Math.log(this.data), [this], [1 / this.data]);
    }

    relu() {
      return new Value(Math.max(0, this.data), [this], [+(this.data > 0)]);
    }

    div(o) {
      return this.mul(o instanceof Value ? o.pow(-1) : 1 / o);
    }

    backward() {
      const gen = ++_gen;
      const topo = [];

      function build(v) {
        if (v._gen === gen) return;
        v._gen = gen;
        v._c.forEach(build);
        topo.push(v);
      }

      build(this);
      this.grad = 1;

      for (let i = topo.length - 1; i >= 0; --i) {
        const v = topo[i];
        v._c.forEach((c, idx) => {
          c.grad += v._g[idx] * v.grad;
        });
      }
    }
  }

  const matrix = (nout, nin, std = 0.08) =>
    Array.from({ length: nout }, () =>
      Array.from({ length: nin }, () => new Value(random.gauss(0, std)))
    );

  const state = {
    wte: matrix(vocab_size, n_embd),
    wpe: matrix(block_size, n_embd),
    lm_head: matrix(vocab_size, n_embd),
  };

  for (let i = 0; i < n_layer; ++i) {
    state[`layer${i}.attn_wq`] = matrix(n_embd, n_embd);
    state[`layer${i}.attn_wk`] = matrix(n_embd, n_embd);
    state[`layer${i}.attn_wv`] = matrix(n_embd, n_embd);
    state[`layer${i}.attn_wo`] = matrix(n_embd, n_embd);
    state[`layer${i}.mlp_fc1`] = matrix(4 * n_embd, n_embd);
    state[`layer${i}.mlp_fc2`] = matrix(n_embd, 4 * n_embd);
  }

  const params = Object.values(state).flat(Infinity);
  log(`num params: ${params.length}`);

  const sum = (arr) => arr.reduce((a, b) => a.add(b));

  function linear(x, w) {
    return w.map((row) => sum(row.map((wi, i) => wi.mul(x[i]))));
  }

  function softmax(logits) {
    const max = Math.max(...logits.map((v) => v.data));
    const exps = logits.map((v) => v.add(-max).exp());
    const total = sum(exps);
    return exps.map((e) => e.div(total));
  }

  function rmsnorm(x) {
    const ms = sum(x.map((v) => v.mul(v))).mul(1 / x.length);
    const s = ms.add(1e-5).pow(-0.5);
    return x.map((v) => v.mul(s));
  }

  function gpt(token_id, pos_id, keys, values) {
    let x = state.wte[token_id].map((v, i) =>
      v.add(state.wpe[pos_id][i])
    );
    x = rmsnorm(x);

    for (let li = 0; li < n_layer; ++li) {
      let residual = x;
      x = rmsnorm(x);

      const q = linear(x, state[`layer${li}.attn_wq`]);
      const k = linear(x, state[`layer${li}.attn_wk`]);
      const v = linear(x, state[`layer${li}.attn_wv`]);

      keys[li].push(k);
      values[li].push(v);

      const attn_out = [];

      for (let h = 0; h < n_head; ++h) {
        const hs = h * head_dim;

        const qh = q.slice(hs, hs + head_dim);
        const kh = keys[li].map((kk) => kk.slice(hs, hs + head_dim));
        const vh = values[li].map((vv) => vv.slice(hs, hs + head_dim));

        const logits = kh.map((kt) =>
          sum(qh.map((qi, i) => qi.mul(kt[i]))).mul(scale)
        );

        const weights = softmax(logits);

        for (let j = 0; j < head_dim; ++j) {
          attn_out.push(
            sum(weights.map((w, t) => w.mul(vh[t][j])))
          );
        }
      }

      x = linear(attn_out, state[`layer${li}.attn_wo`]);
      x = x.map((v, i) => v.add(residual[i]));

      residual = x;
      x = rmsnorm(x);
      x = linear(x, state[`layer${li}.mlp_fc1`]);
      x = x.map((v) => v.relu());
      x = linear(x, state[`layer${li}.mlp_fc2`]);
      x = x.map((v, i) => v.add(residual[i]));
    }

    return linear(x, state.lm_head);
  }

  // ----------------------------
  // TRAINING
  // ----------------------------
  for (let step = 0; step < numSteps; ++step) {
    const start = Math.floor(
      Math.random() * (tokens_full.length - block_size - 1)
    );

    const chunk = tokens_full.slice(start, start + block_size + 1);

    const keys = Array.from({ length: n_layer }, () => []);
    const values = Array.from({ length: n_layer }, () => []);

    const losses = [];

    for (let pos = 0; pos < block_size; ++pos) {
      const logits = gpt(chunk[pos], pos, keys, values);
      const probs = softmax(logits);
      losses.push(probs[chunk[pos + 1]].log().mul(-1));
    }

    const loss = sum(losses).mul(1 / block_size);
    loss.backward();

    const lr = 0.01 * (1 - step / numSteps);

    for (const p of params) {
      p.data -= lr * p.grad;
      p.grad = 0;
    }

    if (step % 10 === 0)
      log(`step ${step} | loss ${loss.data.toFixed(4)}`);
  }

  // ----------------------------
  // GENERATION (Paragraph Mode)
  // ----------------------------
  for (let s = 0; s < numSamples; ++s) {
    const keys = Array.from({ length: n_layer }, () => []);
    const values = Array.from({ length: n_layer }, () => []);

    let token_id = BOS;
    let paragraph = "";

    for (let pos = 0; pos < 400; ++pos) {
      const logits = gpt(token_id, pos % block_size, keys, values);
      const probs = softmax(
        logits.map((l) => l.div(temperature))
      );

      token_id = random.choices(
        [...Array(vocab_size).keys()],
        probs.map((p) => p.data)
      );

      if (token_id === BOS) break;

      paragraph += uchars[token_id];
    }

    emitText(paragraph + "\n\n");
  }
}
