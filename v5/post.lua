wrk.method = "POST"
wrk.headers["Content-Type"] = "application/octet-stream"
wrk.body = io.open("../image.jpg", "rb"):read("*all")