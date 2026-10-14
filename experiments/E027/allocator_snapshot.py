#!/usr/bin/env python3
"""Standalone allocator endpoint helper.

WARNING: its stats describe this helper process only. It is not attached to an
SGLang model worker and must not be used as worker VRAM evidence. For valid
allocator accounting, invoke the snapshot inside the model worker process.
"""
from http.server import BaseHTTPRequestHandler,HTTPServer
import json,os,torch
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  if self.path!='/allocator': self.send_error(404);return
  d={'pid':os.getpid(),'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'max_allocated':torch.cuda.max_memory_allocated(),'max_reserved':torch.cuda.max_memory_reserved(),'stats':{k:v for k,v in torch.cuda.memory_stats().items() if 'allocated_bytes' in k or 'reserved_bytes' in k}}
  b=json.dumps(d).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
 def log_message(self,*a): pass
if __name__ == '__main__':
 # Standalone helper for local API/debug only; this is not model-worker accounting.
 HTTPServer(('127.0.0.1',int(os.getenv('G018_ALLOCATOR_PORT','30004'))),H).serve_forever()
