#!/usr/bin/env python3
"""
MCP server exposing VoxTell and nnInteractive as callable segmentation tools.

Speaks JSON-RPC 2.0 over stdio — the MCP wire protocol — with no SDK dependency,
so it runs under any Python that can import nibabel and numpy. Keeping the
protocol visible also makes this readable as a worked example of what MCP is.

Tools:
  check_request        validate an image/prompt pair without spending any compute
  voxtell_segment      text-prompted segmentation; validates first, refuses on mismatch
  nninteractive_segment bbox-prompted segmentation
  list_models          what is available and what each model can and cannot do

The point of check_request, and of the validation gate inside voxtell_segment:
VoxTell will accept "brain tumor" against an abdominal CT and return a mask of
nothing. The model does not refuse. This server does.
"""

from __future__ import annotations
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import validate as V  # noqa: E402

PROTOCOL_VERSION = "2024-11-05"
MODEL_DIR   = os.environ.get("VOXTELL_MODEL_DIR", "")
NNI_WEIGHTS = os.environ.get("NNINTERACTIVE_WEIGHTS", "")
OUT_DIR     = Path(os.environ.get("VOXTELL_OUTPUT_DIR", Path.home() / ".voxtell_mcp"))


# ── Protocol plumbing ─────────────────────────────────────────────────────────

def send(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def reply(rid, result):
    send({"jsonrpc": "2.0", "id": rid, "result": result})


def error(rid, code, message):
    send({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}})


def text_result(s: str, is_error: bool = False):
    return {"content": [{"type": "text", "text": s}], "isError": is_error}


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(path: str):
    """Return (array, spacing). Supports NIfTI and the CVPR .npz format."""
    import numpy as np
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {path}")

    if p.suffix == ".npz":
        d = np.load(p, allow_pickle=True)
        if "imgs" not in d.files:
            raise ValueError(f"{p.name} has no 'imgs' key (found: {list(d.files)})")
        arr = d["imgs"]
        spacing = tuple(d["spacing"].tolist()) if "spacing" in d.files else None
        return arr, spacing

    import nibabel as nib
    img = nib.load(str(p))
    return img.get_fdata(), tuple(float(z) for z in img.header.get_zooms()[:3])


# ── Tools ─────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "check_request",
        "description": (
            "Check whether a segmentation prompt is anatomically possible for a given image, "
            "without running any model. Returns the inferred modality and body region, what the "
            "prompt asks for, and whether they are compatible. Use this before spending GPU time, "
            "or to explain why a request was refused."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Path to a .nii/.nii.gz or CVPR .npz volume"},
                "prompt": {"type": "string", "description": "The text prompt, e.g. 'the spleen'"},
            },
            "required": ["image_path", "prompt"],
        },
    },
    {
        "name": "voxtell_segment",
        "description": (
            "Segment a structure from a 3-D medical image using a free-text prompt (VoxTell). "
            "Validates the request first and REFUSES anatomically impossible pairings — asking for "
            "a brain structure in an abdominal CT returns an error, not a meaningless mask. "
            "Set force=true only to override a refusal deliberately."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "prompt": {"type": "string", "description": "What to segment, e.g. 'the liver'"},
                "force": {"type": "boolean", "description": "Run even if validation refuses. Default false.", "default": False},
            },
            "required": ["image_path", "prompt"],
        },
    },
    {
        "name": "nninteractive_segment",
        "description": (
            "Segment using a 3-D bounding box prompt (nnInteractive). No text encoder, so there is "
            "no anatomy to validate — the box defines the target. Faster per object than VoxTell."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "bbox": {
                    "type": "array",
                    "description": "[[z_min,z_max],[y_min,y_max],[x_min,x_max]], max exclusive",
                    "items": {"type": "array", "items": {"type": "integer"}},
                },
            },
            "required": ["image_path", "bbox"],
        },
    },
    {
        "name": "list_models",
        "description": "List available segmentation models, how each is prompted, and their known limits.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def tool_check_request(args):
    arr, spacing = load_image(args["image_path"])
    v = V.check(arr, args["prompt"], spacing, Path(args["image_path"]).name)
    lines = [
        f"Request: {'ALLOWED' if v.allowed else 'REFUSED'}",
        "",
        f"  {v.reason}",
        "",
        "Image",
        f"  modality : {v.image.modality}",
        f"  region   : {v.image.region}",
        f"  shape    : {v.image.shape}" + (f"   spacing: {v.image.spacing}" if v.image.spacing else ""),
    ]
    for n in v.image.notes:
        lines.append(f"  - {n}")
    lines += ["", "Prompt", f"  region  : {v.prompt_region}"]
    if v.matched_terms:
        lines.append(f"  matched : {', '.join(v.matched_terms[:5])}")
    return text_result("\n".join(lines), is_error=not v.allowed)


def tool_voxtell_segment(args):
    image_path = args["image_path"]
    prompt = args["prompt"]
    force = bool(args.get("force", False))

    arr, spacing = load_image(image_path)
    v = V.check(arr, prompt, spacing, Path(image_path).name)

    if not v.allowed and not force:
        return text_result(
            "REFUSED — no compute was spent.\n\n"
            f"{v.reason}\n\n"
            f"Image reads as: {v.image.modality} of the {v.image.region}.\n"
            f"Prompt targets: {v.prompt_region}.\n\n"
            "If this is deliberate — testing behaviour on out-of-distribution input, say — "
            "call again with force=true. The mask will not be meaningful.",
            is_error=True,
        )

    if not MODEL_DIR or not Path(MODEL_DIR).exists():
        return text_result(
            f"Validation passed ({v.reason})\n\n"
            f"Cannot run inference: VOXTELL_MODEL_DIR is not set or does not exist "
            f"({MODEL_DIR!r}). Set it in the plugin's user config.",
            is_error=True,
        )

    import numpy as np
    import torch
    from voxtell.inference.predictor import VoxTellPredictor

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    predictor = VoxTellPredictor(model_dir=MODEL_DIR, device=device)
    raw = arr[None].astype(np.float32) if arr.ndim == 3 else arr.astype(np.float32)
    data, bbox, orig_shape = predictor.preprocess(raw)
    emb = predictor.embed_text_prompts([prompt])
    logits = predictor.predict_sliding_window_return_logits(data, emb)

    mask = (torch.sigmoid(logits.float().cpu()) > 0.5).numpy()
    n_vox = int(mask.sum())

    out = OUT_DIR / f"{Path(image_path).stem}__{prompt.replace(' ', '_')[:40]}.npz"
    np.savez_compressed(out, mask=mask.astype(np.uint8))

    warn = ""
    if force and not v.allowed:
        warn = "\n\nNOTE: validation was overridden with force=true. Treat this mask as meaningless.\n"
    if n_vox == 0:
        warn += "\nThe mask is empty — the model found nothing matching this prompt."

    return text_result(
        f"Segmented '{prompt}'\n"
        f"  device      : {device}\n"
        f"  image       : {Path(image_path).name}  {tuple(arr.shape)}\n"
        f"  validation  : {v.reason}\n"
        f"  voxels      : {n_vox:,}\n"
        f"  saved       : {out}"
        + warn
    )


def tool_nninteractive_segment(args):
    if not NNI_WEIGHTS or not Path(NNI_WEIGHTS).exists():
        return text_result(
            f"nnInteractive weights not configured or missing ({NNI_WEIGHTS!r}). "
            "Set nninteractive_weights in the plugin's user config.",
            is_error=True,
        )
    import numpy as np
    import torch
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

    arr, _ = load_image(args["image_path"])
    bbox = [[int(a), int(b)] for a, b in args["bbox"]]

    session = nnInteractiveInferenceSession(
        device=torch.device("cuda", 0),
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=os.cpu_count(),   # SLURM_CPUS_PER_TASK hangs with fold='all'
        do_autozoom=True,
        use_pinned_memory=True,
    )
    session.initialize_from_trained_model_folder(NNI_WEIGHTS, use_fold="all")

    buf = torch.zeros(arr.shape, dtype=torch.uint8, device="cpu")
    session.set_image(arr[None].astype(np.float32))
    session.set_target_buffer(buf)
    session.reset_interactions()
    session.add_bbox_interaction(bbox, include_interaction=True, run_prediction=False)
    session.new_interaction_centers = [session.new_interaction_centers[-1]]
    session.new_interaction_zoom_out_factors = [session.new_interaction_zoom_out_factors[-1]]
    session._predict()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{Path(args['image_path']).stem}__bbox.npz"
    np.savez_compressed(out, mask=buf.numpy())
    return text_result(
        f"Segmented from bounding box {bbox}\n"
        f"  voxels : {int((buf.numpy() > 0).sum()):,}\n"
        f"  saved  : {out}"
    )


def tool_list_models(args):
    return text_result(
        "VoxTell — text-prompted 3-D segmentation\n"
        "  prompt      : free text, e.g. 'the spleen'\n"
        "  encoder     : Qwen3-Embedding-4B, INT4 (NF4) by default\n"
        "  limits      : segments what the prompt names IF it is in the field of view. It does not\n"
        "                verify that itself, which is why check_request exists.\n"
        "  measured    : 2.7x faster after optimization on abdominal CT (n=4), DSC +0.0003\n"
        f"  checkpoint  : {MODEL_DIR or 'NOT CONFIGURED'}\n"
        "\n"
        "nnInteractive — bounding-box prompted 3-D segmentation\n"
        "  prompt      : [[z0,z1],[y0,y1],[x0,x1]]\n"
        "  limits      : must use fold='all'; fold=0 gives ~0.33 DSC and is not a valid baseline\n"
        "  measured    : 1.33x with torch.compile (n=4), DSC +0.0002\n"
        f"  weights     : {NNI_WEIGHTS or 'NOT CONFIGURED'}\n"
    )


HANDLERS = {
    "check_request": tool_check_request,
    "voxtell_segment": tool_voxtell_segment,
    "nninteractive_segment": tool_nninteractive_segment,
    "list_models": tool_list_models,
}


# ── Dispatch, shared by both transports ───────────────────────────────────────
# MCP's protocol semantics are identical on every transport; a transport only
# decides how messages are framed and delivered. So the routing below is written
# once and both stdio and HTTP call into it.

def handle(req: dict):
    """Return a response dict, or None for a notification (which gets no reply)."""
    method, rid = req.get("method"), req.get("id")

    def ok(result):
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    if method == "initialize":
        return ok({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "voxtell-seg", "version": "0.1.0"},
        })
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return ok({"tools": TOOLS})
    if method == "ping":
        return ok({})
    if method == "tools/call":
        params = req.get("params", {})
        name = params.get("name")
        handler = HANDLERS.get(name)
        if handler is None:
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32601, "message": f"unknown tool: {name}"}}
        try:
            return ok(handler(params.get("arguments", {})))
        except Exception as e:
            return ok(text_result(
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc(limit=3)}", is_error=True))
    if rid is not None:
        return {"jsonrpc": "2.0", "id": rid,
                "error": {"code": -32601, "message": f"unknown method: {method}"}}
    return None


# ── Transport: stdio ──────────────────────────────────────────────────────────
# Newline-delimited JSON-RPC over the standard streams of a client-launched
# subprocess. This is what Claude Code and Claude Desktop use.

def serve_stdio():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle(req)
        if resp is not None:
            send(resp)


# ── Transport: Streamable HTTP ────────────────────────────────────────────────
# Each message is an HTTP POST to a single endpoint. A web client such as
# ChatGPT cannot launch a subprocess on your machine, so stdio is unavailable to
# it and this is the binding it needs.

def serve_http(host: str, port: int):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):        # keep stdout clean
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, code: int, body: bytes = b"", ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self):
            # Convenience for humans checking the server is up. The spec's GET
            # is for opening an SSE stream, which these tools do not need.
            if self.path.rstrip("/") in ("", "/mcp", "/health"):
                self._send(200, json.dumps({
                    "server": "voxtell-seg",
                    "transport": "streamable-http",
                    "endpoint": "/mcp",
                    "tools": [t["name"] for t in TOOLS],
                }).encode())
            else:
                self._send(404, b'{"error":"not found"}')

        def do_POST(self):
            if self.path.rstrip("/") not in ("", "/mcp"):
                self._send(404, b'{"error":"not found"}')
                return
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._send(400, json.dumps({
                    "jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": "parse error"}}).encode())
                return

            # A batch is a JSON array; a single message is an object.
            if isinstance(req, list):
                out = [r for r in (handle(m) for m in req) if r is not None]
                self._send(200 if out else 202, json.dumps(out).encode() if out else b"")
                return

            resp = handle(req)
            if resp is None:
                self._send(202)                    # notification: accepted, no body
            else:
                self._send(200, json.dumps(resp).encode())

    srv = ThreadingHTTPServer((host, port), Handler)
    sys.stderr.write(
        f"voxtell-seg MCP server on http://{host}:{port}/mcp\n"
        f"  tools: {', '.join(t['name'] for t in TOOLS)}\n"
        f"  bound to {host}; to reach it from a hosted client you need a tunnel\n"
        f"  (e.g. cloudflared / ngrok) and an auth layer in front. Do not expose\n"
        f"  this directly: it reads local files and runs GPU jobs, with no auth.\n"
    )
    srv.serve_forever()


def main():
    args = sys.argv[1:]
    if "--http" in args:
        host = "127.0.0.1"
        port = 8765
        if "--port" in args:
            port = int(args[args.index("--port") + 1])
        if "--host" in args:
            host = args[args.index("--host") + 1]
        serve_http(host, port)
    else:
        serve_stdio()


if __name__ == "__main__":
    main()
