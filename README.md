# BHAV

A Bengali-speaking voice assistant backend. BHAV accepts a Bengali text prompt over HTTP, answers it using a Groq-hosted LLM, and returns the reply as both Bengali text and synthesized Bengali speech.

The LLM reasons in English — Bengali in and Bengali out is handled by a translation layer on either side of the model call.

## How a request flows

```
Bengali prompt
  → mtranslate (bn → en)
  → Groq chat completion
  → mtranslate (en → bn)
  → strip TTS-hostile characters
  → Edge TTS (bn-BD-NabanitaNeural)
  → base64 MP3 + Bengali text
```

Text cleaning before synthesis removes `*`, `= \ /`, and bracket characters so the voice doesn't read markdown punctuation aloud.

## API

### `POST /chat`

**Request**

```json
{ "prompt": "তোমার নাম কি?" }
```

**Response**

```json
{
  "response": "আমার নাম ভাব।",
  "audio": "<base64-encoded MP3>",
  "call": null
}
```

`audio` is `null` if speech synthesis fails; the text reply is still returned, so callers should handle a missing audio field rather than assume it.

### `GET /ping`

Returns `OK` with status 200. Intended for uptime and cold-start checks.

## Configuration

Set via environment variables.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | yes | — | Groq API key. The app raises on startup if unset. |
| `GROQ_MODEL` | no | `openai/gpt-oss-120b` | Groq model ID. |
| `PORT` | no | `5000` | Listen port. Render sets this automatically. |

### On choosing a model

`GROQ_MODEL` exists because Groq retires models on its free and developer tiers with real regularity, and each retirement breaks this app with a `model_not_found` 404. This project has already been moved four times: `llama-3.3-70b-versatile` → `openai/gpt-oss-120b` → `llama-4-maverick` → `llama-4-scout` → back to `openai/gpt-oss-120b`.

As of August 2026 the entire Llama family is gone from the free and developer tiers. `openai/gpt-oss-120b` is on Groq's production tier and is their recommended replacement. When it eventually goes too, set `GROQ_MODEL` on the host instead of editing and redeploying code — check [Groq's deprecation page](https://console.groq.com/docs/deprecations) for the current list.

If you switch to a reasoning model, note that this app sends `message.content` straight to translation and TTS. On Groq, `gpt-oss` returns reasoning in a separate `message.reasoning` field, so `content` stays clean. Other reasoning models may need `reasoning_format="hidden"` to avoid the model's scratchpad being spoken out loud.

## Running locally

```bash
pip install -r requirements.txt

export GROQ_API_KEY=gsk_...   # PowerShell: $env:GROQ_API_KEY = "gsk_..."
python app.py
```

Serves on `http://localhost:5000`.

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "তোমার নাম কি?"}'
```

## Docker

```bash
docker build -t bhav .
docker run -p 5000:5000 -e GROQ_API_KEY=gsk_... bhav
```

Built on `python:3.10` (Debian) so `apt-get` is available for the PortAudio system dependency.

## Deployment

Configured for Render. The Dockerfile declares `PORT=5000` as a default and Render overrides it at runtime, which `app.py` reads on startup. Set `GROQ_API_KEY` as a Render environment variable — never commit it.

## Repository layout

| Path | Purpose |
|---|---|
| `app.py` | Entire application: routes, translation, LLM call, TTS |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container build |
| `spec-file.txt` | Conda environment spec (win-64), for local Windows setup |

## Notes for future work

A few things in the current code are worth knowing about before you build on it:

- **The container runs Flask's development server.** `gunicorn` is in `requirements.txt` but the Dockerfile's `CMD` is `python app.py`. For production traffic, switch to `gunicorn app:app`.
- **TLS certificate verification is disabled globally** at the top of `app.py`, along with urllib3 warnings. This was presumably a workaround for a local certificate problem. It applies to every outbound HTTPS request the process makes, including the Groq API call.
- **CORS is open to all origins.** Fine for coursework, worth restricting if this is ever exposed publicly.
- **Conversations are stateless.** Each `/chat` call sends only the system prompt and the current message, so BHAV has no memory of earlier turns.
- **The `call` field is always `null`** — reserved by the response shape but not currently populated.
