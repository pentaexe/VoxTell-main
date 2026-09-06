# voxtell-seg — a Claude Code plugin

Makes VoxTell and nnInteractive directly callable as segmentation tools, and
refuses requests that are anatomically impossible before spending any compute.

## Why this exists

VoxTell is text-prompted, so it accepts any string. Hand it an abdominal CT and
ask for a brain tumour and it will run, and return a mask. The mask is of
nothing, but nothing in the pipeline says so — you get a plausible-looking
result and no warning.

That is the gap this plugin closes. `check_request` inspects the image and the
prompt independently and reports whether they are compatible; `voxtell_segment`
runs that check first and returns an error instead of a meaningless mask.

```
Request: REFUSED

  The prompt asks for a structure in the brain (brain, brain tumor), but the
  image is a CT study of the torso. That structure is not in this field of
  view, so any mask returned would be meaningless.

Image
  modality : CT
  region   : torso
  shape    : (406, 512, 512)   spacing: (1.25, 0.82, 0.82)
  - intensities reach -2048, consistent with Hounsfield units (air ~ -1000)
  - 25.0% of voxels below -700 HU (air)
  - 508 mm of coverage spans thorax and abdomen together

Prompt
  region  : brain
  matched : brain, brain tumor
```

## What each piece does

This plugin bundles all three extension mechanisms, which is the clearest way to
see how they differ:

| Mechanism | What it is | Here |
|---|---|---|
| **Skill** | Instructions loaded into the model's context. Shapes *how it works* — workflow, protocol, standards. Advisory. | `skills/voxtell-inference`, `skills/nninteractive-inference`: cluster procedure, measurement rules, submission standards |
| **MCP server** | A process exposing *callable tools* over JSON-RPC. Real code with real return values. Deterministic — it can refuse. | `mcp_server/server.py`: five tools, including the validation gate |
| **Plugin** | A package bundling skills, MCP servers, commands and hooks into one installable unit | this directory |

The distinction that matters: a skill can *tell* the model that a brain prompt
on an abdominal CT is wrong. Only a tool can *make the call fail*. One is
guidance the model may or may not follow; the other is a gate it cannot walk
past.

Because MCP is an open protocol, the same server works in any MCP client, not
only Claude Code.

## Tools

| Tool | Purpose |
|---|---|
| `check_request` | Validate an image/prompt pair. No compute. Returns inferred modality, region, and the reasoning. |
| `voxtell_segment` | Text-prompted segmentation. Validates first; refuses on mismatch unless `force=true`. |
| `nninteractive_segment` | Bounding-box prompted segmentation. No text, so no anatomy check needed. |
| `list_models` | What is available and what each model can and cannot do. |
| `setup` | Check the machine, report which VoxTell build is live, and fetch the checkpoint. |

## How validation works

Deliberately simple and inspectable — intensity statistics and geometry, not a
classifier. A validator nobody can audit is a validator nobody should trust.

- **Modality** from the intensity distribution. CT is calibrated in Hounsfield
  units, so air sits near -1000; MR intensities are arbitrary and rarely
  negative.
- **Region** from air fraction and field of view. A head has almost no internal
  air; a torso has lungs or bowel gas. Thresholds are stated in `validate.py`.
- **Prompt** against an anatomy vocabulary mapping terms to body regions.

It refuses only on a clear mismatch. Unknown on either side proceeds with a
note — a validator that blocks whatever it cannot classify is useless. Adjacent
regions (thorax/abdomen, abdomen/pelvis) pass, since they routinely share a
field of view.

**Known limits.** The region heuristic is thresholds on one CT convention. A
FLARE abdominal CT with lung in frame comes back as `torso` rather than
`abdomen` — honest about the 508 mm field of view, but coarser than the label a
radiologist would use, and the adjacency rule is what makes it behave. It has
not been tested on MR beyond the modality check, and the vocabulary covers
common structures only.

## Install

In Claude Code:

```
/plugin marketplace add pentaexe/VoxTell-main
/plugin install voxtell-seg@voxtell
```

Then, in any conversation:

> Run the setup tool with download true

That checks your machine and fetches the ~1.7 GB checkpoint from Hugging Face
(`mrokuss/VoxTell`) into `~/.cache/voxtell_models`. The checkpoint is not in the
repository, so a fresh install always needs this once.

### Prerequisites

> **Install VoxTell from this repository, not from PyPI.**
> `pip install voxtell` fetches the upstream DKFZ package. The optimizations
> measured here — the INT4 text backbone, the embedding cache, Numba
> preprocessing and `tile_step=0.75` — live in this fork's
> `voxtell/inference/predictor.py` and are **not** in the PyPI build. A plugin
> pointed at the PyPI copy still segments, and returns masks that look fine, with
> none of the speedups. The `setup` and `list_models` tools report which build is
> live so this is visible rather than silent.

```bash
git clone https://github.com/pentaexe/VoxTell-main
cd VoxTell-main
pip install -e .                                  # the optimized build
pip install nibabel accelerate huggingface_hub
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

A CUDA GPU is effectively required; CPU inference on a 3-D volume takes minutes.

`accelerate` looks optional and is not: without it VoxTell's INT4 loader catches
the ImportError and serves FP16 while still logging INT4.

### Configuration

**`voxtell_python` must be a full path.** There is no bare interpreter name that
works everywhere: on Ubuntu `python` usually does not exist, and on Windows
`python`, `python3` and `py` all commonly resolve to the Microsoft Store stub,
which exits without running anything.

To find the right value:

```bash
python3 skills/voxtell-inference/scripts/doctor.py
```

It tests every interpreter it can find, reports which have voxtell and whether
it is the optimized build, and prints the exact path to paste.

| Key | Default | Notes |
|---|---|---|
| `voxtell_python` | `python3` | **Set this.** Full path to the env where you ran `pip install -e .` |
| `voxtell_model_dir` | blank | Optional. Blank means download to `~/.cache/voxtell_models` |
| `nninteractive_weights` | blank | Optional. Needs `fold_all` |

### Installed and enabled, but no tools appear

That is almost always `voxtell_python`. Claude Code launches the MCP server as a
subprocess; if the command does not resolve to a real interpreter, the process
never starts, so there is no server to write a log. The plugin looks healthy and
has no tools.

Confirm by running the command yourself:

```bash
<voxtell_python> voxtell-plugin/mcp_server/server.py --http --port 8765
```

A working interpreter prints a startup line. A stub exits silently.

### Developing on it

```bash
claude plugin validate ./voxtell-plugin --strict
claude --plugin-dir ./voxtell-plugin
```

> **Every `userConfig` key the MCP server references needs a `default`.**
> `--plugin-dir` does not prompt for configuration, so an unresolved
> `${user_config.…}` makes the server entry invalid, and an invalid entry is
> **skipped silently** with nothing in `--debug` to explain it. The symptom is a
> plugin that validates and loads while its tools simply do not exist.

## Two transports: Claude Code and ChatGPT

MCP's protocol semantics are identical on every transport — a transport only
decides how messages are framed and delivered. So the same five tools are
reachable two ways, and the routing code is written once.

**stdio** (default) — newline-delimited JSON-RPC over the standard streams of a
subprocess the client launches. This is what Claude Code and Claude Desktop use,
and it is what `.mcp.json` configures.

```bash
printf '%s\n' \
 '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
 '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
 | python mcp_server/server.py
```

**Streamable HTTP** — each message is an HTTP POST to one endpoint. A hosted
client such as ChatGPT cannot launch a subprocess on your machine, so stdio is
unavailable to it and this is the binding it needs.

```bash
python mcp_server/server.py --http --port 8765

curl -X POST http://127.0.0.1:8765/mcp -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

`GET /mcp` returns a small status document, which is handy for confirming the
server is up.

> **Before exposing this.** It binds to `127.0.0.1` deliberately. Reaching it
> from a hosted client needs a tunnel (cloudflared, ngrok) *and* an auth layer in
> front. The server reads arbitrary local file paths and starts GPU work, and it
> has no authentication of its own. Do not put it on a public URL as-is.

## Status

Both branches verified end to end through the MCP layer on an RTX 4070 SUPER.

**Allowed request** — MNI T1 with prompt `"brain"`:

```
Segmented 'brain'
  device      : cuda:0
  image       : mni_icbm152_t1_tal_nlin_sym_09a.nii.gz  (189, 233, 197)
  validation  : Prompt targets the brain, and the image looks like a brain study.
  voxels      : 1,829,613
  saved       : ...\mni_icbm152_t1_tal_nlin_sym_09a__brain.nii.gz  (aligned to the input image)
```

That voxel count is 0.07% from the 1,828,296 measured for the same prompt in
the INT4 comparison, so the mask is real rather than an artifact of the plumbing.

**Refused request** — FLARE abdominal CT with prompt `"brain tumor"`: returns
`isError: true`, and no model is loaded and no GPU touched. The refusal happens
before any compute.

`nninteractive_segment` has not been exercised — it needs the weights directory,
which lives on the cluster.
