"""Stops oversized uploads before the multipart parser spools them to disk."""
from __future__ import annotations

import json


class BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    def __init__(self, app, path: str, max_bytes: int) -> None:
        self.app, self.path, self.max_bytes = app, path, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != self.path or scope["method"] != "POST":
            return await self.app(scope, receive, send)

        headers = dict(scope["headers"])
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = 0
        if declared > self.max_bytes:
            body = json.dumps({"code": "file_too_large", "message": "File is larger than the limit"}).encode()
            await send({"type": "http.response.start", "status": 413,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode()),
                                    (b"connection", b"close")]})
            await send({"type": "http.response.body", "body": body})
            return None

        # Chunked uploads declare no length, so also count the bytes as they arrive.
        seen = 0

        async def limited_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise BodyTooLarge()
            return message

        return await self.app(scope, limited_receive, send)
