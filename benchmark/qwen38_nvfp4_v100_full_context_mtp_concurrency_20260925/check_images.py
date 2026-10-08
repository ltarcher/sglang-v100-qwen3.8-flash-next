"""Exercise concurrent image requests against an already running local server."""

import argparse
import base64
import io
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image, ImageDraw


def make_image(color: str, number: int) -> dict:
    image = Image.new("RGB", (1536, 1024), "white")
    draw = ImageDraw.Draw(image)
    for x in range(0, 1536, 128):
        draw.line((x, 0, x, 1024), fill="#d0d0d0", width=3)
    for y in range(0, 1024, 128):
        draw.line((0, y, 1536, y), fill="#d0d0d0", width=3)
    draw.rectangle((360, 160, 1176, 864), fill=color)
    draw.text((50, 50), f"Image {number}", fill="black", stroke_width=1)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}}


def sample_gpu(stop: threading.Event, samples: list) -> None:
    while not stop.is_set():
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        samples.append(
            [dict(zip(("gpu", "used_mib", "free_mib"), map(int, line.split(","))))
             for line in result.stdout.splitlines()]
        )
        stop.wait(0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8083")
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--images", type=int, default=3)
    parser.add_argument("--filler-repeats", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    colors = ("red", "blue", "green", "yellow")
    stop = threading.Event()
    samples = []
    sampler = threading.Thread(target=sample_gpu, args=(stop, samples), daemon=True)
    rows = []

    def run(index: int) -> dict:
        color = colors[index % len(colors)]
        items = [make_image(color, i + 1) for i in range(args.images)]
        items.append({
            "type": "text",
            "text": ("Background notes for the agent. " * args.filler_repeats)
            + "What color is the large central rectangle in each image? Reply with just the color name.",
        })
        payload = {
            "model": "qwen",
            "messages": [{"role": "user", "content": items}],
            "temperature": 0,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        start = time.monotonic()
        try:
            response = requests.post(
                args.base_url + "/v1/chat/completions", json=payload, timeout=900
            )
            row = {"request": index, "status": response.status_code,
                   "elapsed_s": round(time.monotonic() - start, 2)}
            if response.ok:
                body = response.json()
                answer = body["choices"][0]["message"]["content"] or ""
                row.update(answer=answer, expected_color=color, usage=body["usage"],
                           finish_reason=body["choices"][0]["finish_reason"])
                row["passed"] = color in answer.lower() and body["usage"]["completion_tokens"] > 0
            else:
                row.update(passed=False, error=response.text[:500])
            return row
        except Exception as exc:
            return {"request": index, "passed": False, "error": repr(exc),
                    "elapsed_s": round(time.monotonic() - start, 2)}

    sampler.start()
    start = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=args.requests) as pool:
            rows = list(pool.map(run, range(args.requests)))
    finally:
        stop.set()
        sampler.join(timeout=2)
    peak = []
    for gpu in range(4):
        record = [sample[gpu] for sample in samples]
        peak.append({"gpu": gpu, "peak_used_mib": max(s["used_mib"] for s in record),
                     "minimum_free_mib": min(s["free_mib"] for s in record)})
    summary = {"requests": args.requests, "images_per_request": args.images,
               "filler_repeats": args.filler_repeats,
               "elapsed_s": round(time.monotonic() - start, 2),
               "peak_gpu_memory": peak, "results": rows,
               "passed": all(row["passed"] for row in rows)}
    with open(args.output, "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
