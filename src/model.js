// Ported to JavaScript from Andrej Karpathy's minGPT (Python)
// Original: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
import random from "./random.js";

class Value {
  constructor(data, children = [], local_grads = []) {
    this.data = data; // scalar value of this node calculated during forward pass
    this.grad = 0; // derivative of the loss w.r.t. this node, calculated in backward pass
    this._c0 = children[0]; // children of this node in the computation graph
    this._c1 = children[1];
    this._lg0 = local_grads[0]; // local derivative of this node w.r.t. its children
    this._lg1 = local_grads[1];
    this._nch = children.length; // number of children (0, 1, or 2)
    this._gen = 0;
  }

  add(other) {
    if (other instanceof Value)
      return new Value(this.data + other.data, [this, other], [1, 1]);
    return new Value(this.data + other, [this], [1]);
  }

  mul(other) {
    if (other instanceof Value)
      return new Value(
        this.data * other.data,
        [this, other],
        [other.data, this.data],
      );
    return new Value(this.data * other, [this], [other]);
  }

  pow(other) {
    return new Value(
      this.data ** other,
      [this],
      [other * this.data ** (other - 1)],
    );
  }
  log() {
    return new Value(Math.log(this.data), [this], [1 / this.data]);
  }
  exp() {
    const e = Math.exp(this.data);
    return new Value(e, [this], [e]);
  }
  relu() {
    return new Value(Math.max(0, this.data), [this], [+(this.data > 0)]);
  }
  neg() {
    return new Value(-this.data, [this], [-1]);
  }
  sub(other) {
    return this.add(other instanceof Value ? other.neg() : -other);
  }
  div(other) {
    return this.mul(other instanceof Value ? other.pow(-1) : 1 / other);
  }

  backward() {
    let _gen = 0;
    const gen = ++_gen;
    const topo = [];
    function build_topo(v) {
      if (v._gen === gen) return;
      v._gen = gen;
      if (v._nch >= 1) build_topo(v._c0);
      if (v._nch === 2) build_topo(v._c1);
      topo.push(v);
    }
    build_topo(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; --i) {
      const v = topo[i],
        g = v.grad;
      if (v._nch >= 1) v._c0.grad += v._lg0 * g;
      if (v._nch === 2) v._c1.grad += v._lg1 * g;
    }
  }
}

export class MicroGPT {
  constructor() {
    this.state_dict = null;
    this.uchars = null;
    this.char_to_id = null;
    this.vocab_size = 0;
    this.BOS = 0;
    this.block_size = 16;
    this.n_embd = 16;
    this.n_head = 4;
    this.n_layer = 1;
    this.head_dim = Math.floor(this.n_embd / this.n_head);
    this.scale = 1 / this.head_dim ** 0.5;
  }

  // Helpers
  matrix(nout, nin, std = 0.08) {
    return Array.from({ length: nout }, () =>
      Array.from({ length: nin }, () => new Value(random.gauss(0, std)))
    );
  }

  sum(arr) {
    return arr.reduce((a, b) => a.add(b));
  }

  zip(a, b) {
    return a.map((ai, i) => [ai, b[i]]);
  }

  linear(x, w) {
    return w.map((wo) => this.sum(wo.map((wi, i) => wi.mul(x[i]))));
  }

  softmax(logits) {
    const max_val = Math.max(...logits.map((v) => v.data));
    const exps = logits.map((v) => v.sub(max_val).exp());
    const total = this.sum(exps);
    return exps.map((e) => e.div(total));
  }

  rmsnorm(x) {
    const ms = this.sum(x.map((xi) => xi.mul(xi))).mul(1 / x.length);
    const s = ms.add(1e-5).pow(-0.5);
    return x.map((xi) => xi.mul(s));
  }

  gpt(token_id, pos_id, keys, values) {
    const tok_emb = this.state_dict["wte"][token_id];
    const pos_emb = this.state_dict["wpe"][pos_id];
    let x = this.zip(tok_emb, pos_emb).map(([t, p]) => t.add(p));
    x = this.rmsnorm(x);

    for (let li = 0; li < this.n_layer; ++li) {
      // 1) Multi-head attention block
      let x_residual = x;
      x = this.rmsnorm(x);
      const q = this.linear(x, this.state_dict[`layer${li}.attn_wq`]);
      const k = this.linear(x, this.state_dict[`layer${li}.attn_wk`]);
      const v = this.linear(x, this.state_dict[`layer${li}.attn_wv`]);
      keys[li].push(k);
      values[li].push(v);
      const x_attn = [];
      for (let h = 0; h < this.n_head; ++h) {
        const hs = h * this.head_dim;
        const q_h = q.slice(hs, hs + this.head_dim);
        const k_h = keys[li].map((ki) => ki.slice(hs, hs + this.head_dim));
        const v_h = values[li].map((vi) => vi.slice(hs, hs + this.head_dim));
        const attn_logits = k_h.map((kt) =>
          this.sum(this.zip(q_h, kt).map(([qi, ki]) => qi.mul(ki))).mul(this.scale)
        );
        const attn_weights = this.softmax(attn_logits);
        for (let j = 0; j < this.head_dim; ++j)
          x_attn.push(this.sum(attn_weights.map((aw, t) => aw.mul(v_h[t][j]))));
      }
      x = this.linear(x_attn, this.state_dict[`layer${li}.attn_wo`]);
      x = x.map((a, i) => a.add(x_residual[i]));
      // 2) MLP block
      x_residual = x;
      x = this.rmsnorm(x);
      x = this.linear(x, this.state_dict[`layer${li}.mlp_fc1`]);
      x = x.map((xi) => xi.relu());
      x = this.linear(x, this.state_dict[`layer${li}.mlp_fc2`]);
      x = x.map((a, i) => a.add(x_residual[i]));
    }

    return this.linear(x, this.state_dict["lm_head"]);
  }

  async train(options, log, onProgress) {
    const { fileContent, numSteps, nEmbd, nLayer } = options;

    // Update model configuration if provided
    if (nEmbd) this.n_embd = nEmbd;
    if (nLayer) this.n_layer = nLayer;

    // Recalculate derived properties
    this.head_dim = Math.floor(this.n_embd / this.n_head);
    this.scale = 1 / this.head_dim ** 0.5;

    random.seed(42);

    const docs = fileContent
      .trim()
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.length > 0);

    random.shuffle(docs);
    log(`Loaded ${docs.length} docs`);

    // Tokenizer
    this.uchars = [...new Set(docs.join(""))].sort();
    this.char_to_id = new Map(this.uchars.map((ch, i) => [ch, i]));
    this.BOS = this.uchars.length;
    this.vocab_size = this.uchars.length + 1;
    log(`vocab size: ${this.vocab_size}`);

    // Initialize parameters
    this.state_dict = {
      wte: this.matrix(this.vocab_size, this.n_embd),
      wpe: this.matrix(this.block_size, this.n_embd),
      lm_head: this.matrix(this.vocab_size, this.n_embd),
    };

    for (let i = 0; i < this.n_layer; ++i) {
      this.state_dict[`layer${i}.attn_wq`] = this.matrix(this.n_embd, this.n_embd);
      this.state_dict[`layer${i}.attn_wk`] = this.matrix(this.n_embd, this.n_embd);
      this.state_dict[`layer${i}.attn_wv`] = this.matrix(this.n_embd, this.n_embd);
      this.state_dict[`layer${i}.attn_wo`] = this.matrix(this.n_embd, this.n_embd);
      this.state_dict[`layer${i}.mlp_fc1`] = this.matrix(4 * this.n_embd, this.n_embd);
      this.state_dict[`layer${i}.mlp_fc2`] = this.matrix(this.n_embd, 4 * this.n_embd);
    }

    const params = Object.values(this.state_dict).flat(Infinity);
    log(`num params: ${params.length}`);

    // Optimization
    const learning_rate = 0.01,
      beta1 = 0.85,
      beta2 = 0.99,
      eps_adam = 1e-8;
    const m_buf = new Float64Array(params.length);
    const v_buf = new Float64Array(params.length);

    // Training Loop
    const num_steps = numSteps;
    for (let step = 0; step < num_steps; ++step) {
      const doc = docs[step % docs.length];
      const tokens = [this.BOS, ...Array.from(doc, (ch) => this.char_to_id.get(ch)), this.BOS];
      const n = Math.min(this.block_size, tokens.length - 1);

      const keys = Array.from({ length: this.n_layer }, () => []);
      const values = Array.from({ length: this.n_layer }, () => []);
      const losses = [];

      for (let pos_id = 0; pos_id < n; ++pos_id) {
        const token_id = tokens[pos_id];
        const target_id = tokens[pos_id + 1];
        const logits = this.gpt(token_id, pos_id, keys, values);
        const probs = this.softmax(logits);
        const loss_t = probs[target_id].log().neg();
        losses.push(loss_t);
      }

      const loss = this.sum(losses).mul(1 / n);
      loss.backward();

      // Adam Update
      const lr_t = learning_rate * (1 - step / num_steps);
      const bc1 = 1 - beta1 ** (step + 1);
      const bc2 = 1 - beta2 ** (step + 1);

      for (let i = 0; i < params.length; ++i) {
        const p = params[i];
        m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad;
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2;
        const m_hat = m_buf[i] / bc1;
        const v_hat = v_buf[i] / bc2;
        p.data -= (lr_t * m_hat) / (Math.sqrt(v_hat) + eps_adam);
        p.grad = 0;
      }

      if (step % 200 === 0 || step === num_steps - 1) {
        log(
          `step ${step + 1}/${num_steps} | loss ${loss.data.toFixed(4)}`
        );
      }

      if (step % 5 === 0 || step === num_steps - 1) {
        if (onProgress) {
          onProgress((step + 1) / numSteps);
        }
      }

      if (step % 200 === 0) {
        await new Promise(requestAnimationFrame);
      }
    }
  }

  generate(options, emitName) {
    if (!this.state_dict) {
      throw new Error("Model is not trained yet.");
    }
    const { temperature, numSamples } = options;
    const token_ids = Array.from({ length: this.vocab_size }, (_, i) => i);
    const generatedNames = [];

    for (let sample_idx = 0; sample_idx < numSamples; ++sample_idx) {
      const keys = Array.from({ length: this.n_layer }, () => []);
      const values = Array.from({ length: this.n_layer }, () => []);
      let token_id = this.BOS;
      const sample = [];

      for (let pos_id = 0; pos_id < this.block_size; ++pos_id) {
        const logits = this.gpt(token_id, pos_id, keys, values);
        const probs = this.softmax(logits.map((l) => l.div(temperature)));

        token_id = random.choices(
          token_ids,
          probs.map((p) => p.data)
        );

        if (token_id === this.BOS) break;
        sample.push(this.uchars[token_id]);
      }

      const name = sample.join("");
      generatedNames.push(name);

      if (emitName) emitName(name);
    }
    return generatedNames;
  }
}
