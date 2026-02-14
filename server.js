import { Hono } from "hono";

const app = new Hono();

app.get("/status", (c) => c.text("ok"));

export default app;
