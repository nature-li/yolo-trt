# 先把图片转成 Triton 的请求格式
python3 - << 'EOF'
import json, base64

with open("../image.jpg", "rb") as f:
    data = list(f.read())

body = {
    "inputs": [{
        "name": "raw_image",
        "shape": [len(data)],
        "datatype": "UINT8",
        "data": data
    }]
}
with open("payload.json", "w") as f:
    json.dump(body, f)
print("done, size:", len(data))
EOF

# wrk 压测
wrk -t4 -c16 -d10s -s triton.lua http://localhost:8000