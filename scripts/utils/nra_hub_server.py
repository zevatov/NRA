import os
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn
import json

app = FastAPI(title="NRA Hub Demo")

OUT_DIR = Path("/tmp/nra_ultimate_benchmarks")

def parse_range_header(range_header, file_size):
    if not range_header or not range_header.startswith('bytes='):
        return None
    range_match = range_header.replace('bytes=', '').split('-')
    start = int(range_match[0]) if range_match[0] else 0
    end = int(range_match[1]) if len(range_match) > 1 and range_match[1] else file_size - 1
    return start, end

@app.get("/datasets/{file_name}")
def download_dataset(request: Request, file_name: str):
    file_path = OUT_DIR / file_name
    if not file_path.exists():
        return Response(status_code=404)
        
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")
    
    if range_header:
        # Partial content (HTTP 206) - Critical for Zero-Download CloudArchive!
        start, end = parse_range_header(range_header, file_size)
        length = end - start + 1
        
        def file_iterator():
            with open(file_path, "rb") as f:
                f.seek(start)
                bytes_left = length
                while bytes_left > 0:
                    chunk = f.read(min(bytes_left, 1024 * 64))
                    if not chunk:
                        break
                    bytes_left -= len(chunk)
                    yield chunk
                    
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Type": "application/octet-stream"
        }
        return StreamingResponse(file_iterator(), status_code=206, headers=headers)
    else:
        # Full content
        headers = {"Accept-Ranges": "bytes", "Content-Length": str(file_size)}
        return StreamingResponse(open(file_path, "rb"), headers=headers)

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
    <head>
        <title>NRA Hub Demo</title>
        <style>
            body { font-family: -apple-system, system-ui; background: #0f111a; color: white; margin: 40px; }
            .card { background: #1e2133; padding: 20px; border-radius: 12px; margin-bottom: 20px; }
            h1 { color: #58a6ff; }
            code { background: #000; padding: 10px; border-radius: 6px; display: block; color: #7ee787; }
            .badge { background: #3fb950; color: #000; padding: 4px 8px; border-radius: 20px; font-weight: bold; font-size: 12px;}
        </style>
    </head>
    <body>
        <h1>🔥 NRA Hub Demo</h1>
        <p>The Global Dataset Library with <b>Zero-Download Cloud Streaming</b>.</p>
        
        <div class="card">
            <h2>Dataset A: Vision (Real High-Res Images)</h2>
            <p>1000 real photos of cats and dogs. <span class="badge">42.8 MB</span></p>
            <code>
            import nra<br>
            archive = nra.CloudArchive("http://localhost:8080/datasets/A_Vision.nra")<br>
            print(f"Files: {len(archive.file_ids())}")
            </code>
        </div>
        
        <div class="card">
            <h2>Dataset B: Heavy Duplication (LLM Logs)</h2>
            <p>5000 JSON files (40% duplicates). Raw Size: 10.4 MB -> <span class="badge">NRA Size: 1.3 MB (CDC Dedup)</span></p>
            <code>
            archive = nra.CloudArchive("http://localhost:8080/datasets/B_Dedup.nra")
            </code>
        </div>
        
        <div class="card">
            <h2>Dataset C: Multimodal Chaos</h2>
            <p>Mix of text, images, and JSON metadata. <span class="badge">11.2 MB</span></p>
            <code>
            archive = nra.CloudArchive("http://localhost:8080/datasets/C_Multi.nra")
            </code>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
