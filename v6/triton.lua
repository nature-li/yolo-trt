local f = io.open("payload.json", "rb")
local body = f:read("*all")
f:close()

wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.headers["Host"] = "localhost:8000"
wrk.path = "/v2/models/yolov8n_ensemble/infer"
wrk.body = body