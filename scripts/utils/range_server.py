import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

class RangeRequestHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        if 'Range' not in self.headers:
            self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            return super().send_head()
        try:
            # Simplistic Range support
            range_header = self.headers['Range']
            range_match = range_header.replace('bytes=', '').split('-')
            start = int(range_match[0])
            end = int(range_match[1]) if len(range_match) > 1 and range_match[1] else None
            
            path = self.translate_path(self.path)
            f = open(path, 'rb')
            fs = os.fstat(f.fileno())
            file_len = fs[6]
            if end is None or end >= file_len:
                end = file_len - 1
            length = end - start + 1
            
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_len}")
            self.send_header("Content-Length", str(length))
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.end_headers()
            return f
        except Exception:
            return super().send_head()

    def copyfile(self, source, outputfile):
        if 'Range' not in self.headers:
            super().copyfile(source, outputfile)
            return
        range_header = self.headers['Range']
        range_match = range_header.replace('bytes=', '').split('-')
        start = int(range_match[0])
        end = int(range_match[1]) if len(range_match) > 1 and range_match[1] else None
        
        source.seek(start)
        fs = os.fstat(source.fileno())
        file_len = fs[6]
        if end is None or end >= file_len:
            end = file_len - 1
        length = end - start + 1
        
        buf_size = 64 * 1024
        while length > 0:
            read_len = min(length, buf_size)
            data = source.read(read_len)
            if not data:
                break
            outputfile.write(data)
            length -= len(data)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    httpd = HTTPServer(('localhost', port), RangeRequestHandler)
    httpd.serve_forever()
