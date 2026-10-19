from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Binomial distribution</title>
<style>
  body { max-width: 960px; margin: 36px auto; padding: 0 20px; font: 16px sans-serif; color: #18332f; }
  h1 { margin-bottom: 24px; font-size: 24px; }
  .controls { display: flex; gap: 36px; flex-wrap: wrap; margin-bottom: 18px; }
  label { display: grid; grid-template-columns: 32px minmax(150px, 280px) 48px; gap: 12px; align-items: center; }
  input { accent-color: #087f68; }
  output { font-variant-numeric: tabular-nums; }
  canvas { display: block; width: 100%; height: 460px; }
</style>
<h1>Binomial distribution</h1>
<div class="controls">
  <label>n <input id="n" type="range" min="1" max="100" value="20"><output id="n-value"></output></label>
  <label>p <input id="p" type="range" min="0" max="1" step="0.01" value="0.20"><output id="p-value"></output></label>
</div>
<canvas id="plot"></canvas>
<script>
const nSlider = document.querySelector("#n");
const pSlider = document.querySelector("#p");
const canvas = document.querySelector("#plot");
const ctx = canvas.getContext("2d");

function draw() {
  const n = Number(nSlider.value);
  const p = Number(pSlider.value);
  document.querySelector("#n-value").textContent = n;
  document.querySelector("#p-value").textContent = p.toFixed(2);

  const probabilities = Array(n + 1).fill(0);
  if (p === 1) probabilities[n] = 1;
  else {
    probabilities[0] = (1 - p) ** n;
    for (let k = 0; k < n; k++)
      probabilities[k + 1] = probabilities[k] * (n - k) / (k + 1) * p / (1 - p);
  }

  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const left = 54, right = 16, top = 20, bottom = 42;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const max = Math.max(...probabilities) * 1.08;
  ctx.strokeStyle = "#afbfbb";
  ctx.fillStyle = "#52645f";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "right";
  for (let tick = 0; tick <= 4; tick++) {
    const y = top + plotHeight * (1 - tick / 4);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(width - right, y);
    ctx.stroke();
    ctx.fillText((max * tick / 4).toFixed(2), left - 8, y + 4);
  }

  const slot = plotWidth / (n + 1);
  probabilities.forEach((probability, k) => {
    const barHeight = probability / max * plotHeight;
    ctx.fillStyle = "#087f68";
    ctx.fillRect(left + k * slot + slot * 0.12, top + plotHeight - barHeight,
      Math.max(1, slot * 0.76), barHeight);
  });
  ctx.fillStyle = "#52645f";
  ctx.textAlign = "center";
  [0, Math.round(n / 2), n].forEach(k =>
    ctx.fillText(k, left + (k + 0.5) * slot, height - 14));
}

nSlider.addEventListener("input", draw);
pSlider.addEventListener("input", draw);
window.addEventListener("resize", draw);
draw();
</script>
</html>"""


class PageHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    parser = ArgumentParser(description="Serve an interactive binomial plot.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), PageHandler)
    print(f"Serving on http://127.0.0.1:{args.port} (Ctrl+C to stop)")
    server.serve_forever()
