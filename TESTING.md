# Testing pcb locally — step by step

This guide walks you through installing the package, running every command, and wiring it into Claude Code as an MCP tool.

---

## 1. Install the package

From the project directory:

```bash
cd prompt_compression_benchmarker_0625
pip install .
```

Verify it worked:

```bash
pcb --help
```

You should see:

```
Usage: pcb [OPTIONS] COMMAND [ARGS]...

Commands:
  compress          Compress text from a file or stdin
  run               Run the benchmark
  list-compressors  List available compressors
  list-models       List supported LLM judge models
  show-schema       Show JSONL schema for a task type
```

> The package bundles sample data for all three task types. Every command below works from any directory — no `--data-dir` needed.

---

## 2. Run the benchmark

### Quickest possible run (2 samples, RAG only)

```bash
pcb run --task rag --max-samples 2
```

### Full benchmark across all task types

```bash
pcb run --max-samples 5
```

### With cost projection

Replace the model name with whatever you actually call:

```bash
pcb run --max-samples 5 --daily-tokens 1000000 --cost-model claude-sonnet-4-6
```

Other model names: `claude-opus-4-7`, `gpt-4.1`, `gpt-4.1-mini`, `gemini-2.5-pro`, `deepseek-v3.2`

### Save output to a file

```bash
pcb run --max-samples 10 --output results.json
pcb run --max-samples 10 --output results.csv
pcb run --max-samples 10 --output results.html
```

### Target a different compression rate

`--rate` is the fraction of tokens to remove. Default is 0.5.

```bash
pcb run --rate 0.3 --max-samples 5   # lighter compression
pcb run --rate 0.6 --max-samples 5   # aggressive compression
```

---

## 3. Run with LLM-as-judge

This calls a real model to score quality. Requires an OpenRouter API key.

```bash
export OPENROUTER_API_KEY=sk-or-...

pcb run --llm-judge --max-samples 5 --task rag
```

With a specific judge model:

```bash
pcb run --llm-judge --judge-model claude-haiku-4.5 --max-samples 5
```

See all available judge models:

```bash
pcb list-models
```

---

## 4. Compress a file directly

```bash
# Create a test file
echo "The transformer architecture introduced in 2017 by Vaswani et al. replaced recurrent neural networks for sequence modeling. The core innovation is the self-attention mechanism, which allows each token to attend to every other token simultaneously. Multi-head attention runs several attention computations in parallel with different learned projections. The encoder-decoder structure processes inputs through stacked blocks each containing self-attention and feed-forward sublayers with residual connections and layer normalization. GPT uses only the decoder stack with causal attention. BERT uses only the encoder with bidirectional attention." > /tmp/test_context.txt

# Compress it
pcb compress /tmp/test_context.txt --stats

# Try different compressors
pcb compress /tmp/test_context.txt --compressor llmlingua2 --rate 0.4 --stats
pcb compress /tmp/test_context.txt --compressor tfidf --rate 0.5 --stats

# Save compressed output
pcb compress /tmp/test_context.txt --compressor llmlingua2 -o /tmp/compressed.txt --stats
cat /tmp/compressed.txt
```

### Pipe it (works with any script)

```bash
cat /tmp/test_context.txt | pcb compress --compressor llmlingua2
```

---

## 5. Use your own data

Data is JSONL — one JSON object per line.

### See the schema for each task type

```bash
pcb show-schema rag
pcb show-schema summarization
pcb show-schema coding
```

### Create a sample RAG file

```bash
cat > /tmp/my_rag.jsonl << 'EOF'
{"id": "test_001", "context": "Redis supports several persistence mechanisms. RDB (Redis Database) creates point-in-time snapshots of the dataset at specified intervals. AOF (Append Only File) logs every write operation, with fsync options ranging from every write to every second. RDB restarts faster but may lose data since the last snapshot. AOF provides better durability but is slower. Redis 4.0 introduced a hybrid persistence mode combining both approaches. The BGSAVE command triggers a background RDB save. The CONFIG REWRITE command persists runtime config changes to disk.", "question": "What are the two main Redis persistence modes and which one restarts faster?", "answer": "RDB and AOF. RDB restarts faster but may lose more data."}
{"id": "test_002", "context": "Python's Global Interpreter Lock (GIL) is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecode simultaneously. This means CPU-bound tasks do not benefit from threading in CPython. The GIL was introduced to simplify memory management and make the CPython implementation thread-safe without fine-grained locking. I/O-bound tasks can still benefit from threading because the GIL is released during I/O operations. To achieve true parallelism for CPU-bound work, Python developers use multiprocessing, which spawns separate processes each with their own GIL, or use C extensions like NumPy that release the GIL during computation.", "question": "Why does Python's GIL prevent CPU-bound tasks from benefiting from threading, and what is the recommended workaround?", "answer": "The GIL prevents multiple threads from executing Python bytecode simultaneously. The workaround is multiprocessing, which spawns separate processes each with their own GIL."}
EOF

pcb run --data-dir /tmp --task rag --max-samples 2
```

---

## 6. Wire into Claude Code (MCP)

This adds four compression tools directly inside Claude Code conversations.

### Step 1 — Add the MCP server

Run this from inside any project you want it available in:

```bash
claude mcp add pcb -s project -- python -m pcb.mcp_server
```

For all projects (user scope):

```bash
claude mcp add pcb -s user -- python -m pcb.mcp_server
```

### Step 2 — Verify it registered

```bash
claude mcp list
```

You should see `pcb` in the list.

### Step 3 — Open Claude Code and try these prompts

Once the MCP server is connected, paste these into the Claude Code conversation:

**List available compressors:**
```
Use the pcb list_compressors tool to show me all available compression algorithms
```

**Compress a piece of text:**
```
Use the pcb compress_text tool to compress this text with llmlingua2 at rate 0.45:

"The CRISPR-Cas9 gene editing system was first adapted for use in eukaryotic cells 
by Jennifer Doudna and Emmanuelle Charpentier. The system derives from a natural 
bacterial immune mechanism where bacteria incorporate viral DNA snippets into their 
genome in regions called CRISPRs. On subsequent infections, the bacteria use guide 
RNAs to direct Cas9 proteins to destroy matching viral DNA. Doudna and Charpentier 
showed this targeting mechanism could be reprogrammed by designing synthetic guide 
RNAs. The Cas9 protein creates a double-strand break that the cell repairs via NHEJ 
(error-prone) or HDR (precise, requires a DNA template)."
```

**Get a recommendation:**
```
Use pcb recommend to tell me the best compressor for a RAG use case where I want 
at least 90% quality retention
```

**Estimate savings:**
```
Use pcb estimate_savings to calculate how much I'd save compressing this text 
on claude-opus-4-7 at 2000 calls per day with llmlingua2 at rate 0.4:

"The transformer architecture introduced in the 2017 paper Attention Is All You Need 
replaced recurrent networks for sequence modeling. The core innovation is self-attention, 
which allows each token to attend to every other token simultaneously. Multi-head 
attention runs several computations in parallel with different learned projections. 
Positional encodings add sequence order information to permutation-invariant attention."
```

### Step 4 — Use it while working on code

Once wired up, you can ask Claude Code things like:

```
Before answering my next question, use pcb compress_text to compress the contents 
of src/utils/parser.py with llmlingua2 at rate 0.4, then use the compressed version 
as context
```

```
I'm about to send a large codebase context to gpt-4.1 at about 500 calls per day. 
Use pcb estimate_savings with a sample of the attached file to tell me if it's 
worth compressing
```

---

## 7. Test the Python SDK wrapper

This verifies the middleware works without needing an API key — it tests the compression logic directly.

```python
# save as /tmp/test_wrapper.py and run: python3 /tmp/test_wrapper.py

from pcb.middleware.anthropic_client import _compress_content, CompressionStats
from pcb.compressors import ALL_COMPRESSORS

# Set up compressor
c = ALL_COMPRESSORS["llmlingua2"]()
c.initialize()

# Simulate what CompressingAnthropic does to your messages
messages = [
    {
        "role": "user",
        "content": (
            "Redis is an open-source in-memory data store supporting multiple persistence "
            "mechanisms. RDB (Redis Database Backup) creates point-in-time snapshots of the "
            "entire dataset at configured intervals using a fork-and-save approach. The BGSAVE "
            "command triggers an RDB save in the background without blocking client requests. "
            "RDB files are compact binary snapshots that load quickly on restart, making RDB "
            "the faster option for recovery but potentially losing writes since the last snapshot. "
            "AOF (Append Only File) logs every write command received by the server. The fsync "
            "policy controls durability: 'always' fsyncs on every command for maximum safety, "
            "'everysec' fsyncs once per second balancing safety and performance, and 'no' lets "
            "the OS decide. AOF provides better durability guarantees but produces larger files "
            "and has slower restart times due to replaying all recorded commands. Redis 4.0 "
            "introduced a hybrid persistence mode that embeds an RDB snapshot inside the AOF "
            "file, combining fast restart with AOF durability. Based on this context: "
            "which persistence mode has faster restart time and why?"
        )
    },
    {"role": "assistant", "content": "I'll answer based on the context provided."},
]

print("Compression test (no API key needed):\n")
for msg in messages:
    if msg["role"] == "user":
        new_content, orig, comp = _compress_content(msg["content"], c, rate=0.45)
        print(f"  role:     user")
        print(f"  before:   {orig} tokens")
        print(f"  after:    {comp} tokens  ({(orig-comp)/orig*100:.1f}% reduction)")
        print(f"  output:   {new_content[:120]}...")
    else:
        print(f"\n  role:     {msg['role']} — skipped (not in compress_roles)")

# Show stats tracking
stats = CompressionStats()
stats.calls = 100
stats.original_tokens = 45000
stats.compressed_tokens = 25000
print(f"\nAfter 100 calls:")
print(f"  {stats}")
print(f"  Monthly savings at $15/1M, 2000 calls/day: ${stats.monthly_savings_usd(15.0, 2000):,.0f}")
```

Run it:
```bash
python3 /tmp/test_wrapper.py
```

---

## 8. End-to-end with real API (optional)

If you want to test the actual SDK wrapper making real API calls, install the Anthropic SDK and provide a key:

```bash
pip install anthropic

python3 - <<'EOF'
import os
from pcb.middleware import CompressingAnthropic

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."  # your key

client = CompressingAnthropic(compressor="llmlingua2", rate=0.45, verbose=True)

response = client.messages.create(
    model="claude-haiku-4-5",
    messages=[{
        "role": "user",
        "content": (
            "Redis supports two main persistence modes. RDB creates snapshots at "
            "configured intervals and restarts faster but may lose recent writes. "
            "AOF logs every operation with configurable fsync and is more durable "
            "but slower. A hybrid mode in Redis 4.0 combines both approaches. "
            "Based on this context: which persistence mode has faster restart time?"
        )
    }],
    max_tokens=100,
)

print("Response:", response.content[0].text)
print("Stats:", client.stats)
EOF
```

---

## Quick reference

| Command | What it does |
|---|---|
| `pcb run` | Benchmark all compressors on all tasks |
| `pcb run --task rag` | RAG task only |
| `pcb run --max-samples 5` | Limit to 5 samples per task |
| `pcb run --rate 0.4` | 40% token removal target |
| `pcb run --daily-tokens 1000000 --cost-model claude-sonnet-4-6` | Add cost projection |
| `pcb run --llm-judge` | Add real model quality scoring |
| `pcb run --output results.json` | Save to JSON/CSV/HTML |
| `pcb compress file.txt --stats` | Compress a file, show token counts |
| `cat file.txt \| pcb compress` | Pipe compression |
| `pcb list-compressors` | Show all algorithms |
| `pcb list-models` | Show all LLM judge models |
| `pcb show-schema rag` | Show data format |
| `claude mcp add pcb -s project -- python -m pcb.mcp_server` | Add to Claude Code |
