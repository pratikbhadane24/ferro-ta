const fs = require("node:fs");
const path = require("node:path");
const { performance } = require("node:perf_hooks");

const wasm = require("./pkg/ferro_ta_wasm.js");

function parseArgs(argv) {
  const args = { bars: 100000, json: null };
  for (let idx = 0; idx < argv.length; idx += 1) {
    const token = argv[idx];
    if (token === "--bars") {
      args.bars = Number(argv[idx + 1]);
      idx += 1;
    } else if (token === "--json") {
      args.json = argv[idx + 1];
      idx += 1;
    }
  }
  return args;
}

function makeSeries(length) {
  const close = new Float64Array(length);
  const high = new Float64Array(length);
  const low = new Float64Array(length);
  const volume = new Float64Array(length);
  let value = 100.0;
  for (let idx = 0; idx < length; idx += 1) {
    value += Math.sin(idx / 13.0) * 0.35 + Math.cos(idx / 29.0) * 0.18;
    close[idx] = value;
    high[idx] = value + 1.25;
    low[idx] = value - 1.10;
    volume[idx] = 1000.0 + Math.abs(Math.sin(idx / 7.0) * 300.0) + (idx % 100);
  }
  return { close, high, low, volume };
}

function timeMin(fn, rounds = 7) {
  fn();
  let best = Number.POSITIVE_INFINITY;
  for (let round = 0; round < rounds; round += 1) {
    const started = performance.now();
    fn();
    best = Math.min(best, performance.now() - started);
  }
  return best;
}

function runBenchmark({ bars }) {
  const { close, high, low, volume } = makeSeries(bars);
  const cases = [
    ["SMA", () => wasm.sma(close, 20)],
    ["EMA", () => wasm.ema(close, 20)],
    ["WMA", () => wasm.wma(close, 20)],
    ["RSI", () => wasm.rsi(close, 14)],
    ["ADX", () => wasm.adx(high, low, close, 14)],
    ["MFI", () => wasm.mfi(high, low, close, volume, 14)],
    ["ATR", () => wasm.atr(high, low, close, 14)],
    ["BBANDS", () => wasm.bbands(close, 20, 2.0, 2.0)],
  ];

  const results = cases.map(([name, fn]) => {
    const elapsedMs = timeMin(fn);
    return {
      indicator: name,
      elapsed_ms: Number(elapsedMs.toFixed(4)),
      ns_per_bar: Number(((elapsedMs * 1e6) / bars).toFixed(2)),
      million_bars_per_second: Number((((bars / 1e6) / (elapsedMs / 1000))).toFixed(2)),
    };
  });

  return {
    metadata: {
      suite: "wasm",
      runtime: {
        generated_at_utc: new Date().toISOString(),
        node_version: process.version,
        platform: process.platform,
        arch: process.arch,
      },
      dataset: {
        bars,
      },
    },
    results,
  };
}

function printResults(payload) {
  const bars = payload.metadata.dataset.bars;
  console.log(`WASM Benchmark: ${bars} bars`);
  console.log("----------------------------------------------------------------");
  console.log(
    `${"Indicator".padEnd(12)}${"Elapsed (ms)".padStart(14)}${"ns/bar".padStart(12)}${"M bars/s".padStart(12)}`
  );
  console.log("----------------------------------------------------------------");
  for (const row of payload.results) {
    console.log(
      `${row.indicator.padEnd(12)}${row.elapsed_ms.toFixed(2).padStart(14)}${row.ns_per_bar
        .toFixed(2)
        .padStart(12)}${row.million_bars_per_second.toFixed(2).padStart(12)}`
    );
  }
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const payload = runBenchmark(args);
  printResults(payload);

  if (args.json) {
    const outputPath = path.resolve(args.json);
    fs.writeFileSync(outputPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
    console.log(`\nWrote JSON results to ${outputPath}`);
  }
}

main();
