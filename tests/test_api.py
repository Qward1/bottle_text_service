from __future__ import annotations

import json
import socket
import threading
import time
import unittest
import urllib.request
import zipfile
from email import policy
from email.parser import BytesParser
from io import BytesIO

import cv2
import numpy as np
import uvicorn

from app.main import app
from app.processor import process_image
from tests.smoke_test import build_challenging_bottle


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_multipart_body(filename: str, content: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = "----bottle-text-service-boundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")
    return body, boundary


class UvicornServer:
    def __init__(self) -> None:
        self.port = get_free_port()
        self.server = uvicorn.Server(
            uvicorn.Config(app=app, host="127.0.0.1", port=self.port, log_level="error")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        deadline = time.time() + 10.0
        while not self.server.started:
            if time.time() > deadline:
                raise TimeoutError("Uvicorn did not start in time")
            time.sleep(0.05)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10.0)


class ProcessEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.http_server = UvicornServer()
        cls.http_server.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.http_server.stop()

    def test_process_returns_zip_with_images_only(self) -> None:
        content, _ = build_challenging_bottle()
        expected = process_image(content, detector_backend="craft")
        body, boundary = build_multipart_body("bottle.jpg", content, "image/jpeg")

        request = urllib.request.Request(
            url=f"http://127.0.0.1:{self.http_server.port}/process?detector_backend=craft&response_format=zip",
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
            content_type = response.headers.get_content_type()

        self.assertEqual(content_type, "application/zip")

        archive = zipfile.ZipFile(BytesIO(payload))
        self.assertEqual(
            sorted(archive.namelist()),
            ["bw.png", "crop_preview.jpg", "debug_roi.jpg", "high_contrast.jpg", "improved.jpg"],
        )

        crop = cv2.imdecode(np.frombuffer(archive.read("crop_preview.jpg"), np.uint8), cv2.IMREAD_COLOR)
        improved = cv2.imdecode(np.frombuffer(archive.read("improved.jpg"), np.uint8), cv2.IMREAD_COLOR)
        bw = cv2.imdecode(np.frombuffer(archive.read("bw.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
        high_contrast = cv2.imdecode(np.frombuffer(archive.read("high_contrast.jpg"), np.uint8), cv2.IMREAD_COLOR)
        debug_roi = cv2.imdecode(np.frombuffer(archive.read("debug_roi.jpg"), np.uint8), cv2.IMREAD_COLOR)

        self.assertIsNotNone(crop)
        self.assertIsNotNone(improved)
        self.assertIsNotNone(bw)
        self.assertIsNotNone(high_contrast)
        self.assertIsNotNone(debug_roi)
        assert crop is not None
        assert improved is not None
        assert bw is not None
        assert high_contrast is not None
        assert debug_roi is not None

        original = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(original)
        assert original is not None

        self.assertEqual(crop.shape[:2], expected.crop_bgr.shape[:2])
        self.assertEqual(improved.shape[:2], expected.improved_bgr.shape[:2])
        self.assertEqual(bw.shape[:2], expected.bw.shape[:2])
        self.assertEqual(high_contrast.shape[:2], expected.high_contrast.shape[:2])
        self.assertEqual(debug_roi.shape[:2], original.shape[:2])

    def test_process_returns_multipart_files_by_default(self) -> None:
        content, _ = build_challenging_bottle()
        expected = process_image(content, detector_backend="craft")
        body, boundary = build_multipart_body("bottle.jpg", content, "image/jpeg")

        request = urllib.request.Request(
            url=f"http://127.0.0.1:{self.http_server.port}/process?detector_backend=craft",
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
            content_type = response.headers.get("Content-Type", "")

        self.assertIn("multipart/form-data", content_type)

        message = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + payload
        )
        parts = list(message.iter_parts())

        self.assertEqual(
            [part.get_filename() for part in parts],
            ["crop_preview.jpg", "improved.jpg", "bw.png", "high_contrast.jpg", "debug_roi.jpg"],
        )

        images = {
            part.get_filename(): cv2.imdecode(
                np.frombuffer(part.get_payload(decode=True), np.uint8),
                cv2.IMREAD_GRAYSCALE if part.get_filename() == "bw.png" else cv2.IMREAD_COLOR,
            )
            for part in parts
        }

        for image in images.values():
            self.assertIsNotNone(image)

        crop = images["crop_preview.jpg"]
        improved = images["improved.jpg"]
        bw = images["bw.png"]
        high_contrast = images["high_contrast.jpg"]
        debug_roi = images["debug_roi.jpg"]
        assert crop is not None
        assert improved is not None
        assert bw is not None
        assert high_contrast is not None
        assert debug_roi is not None

        original = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(original)
        assert original is not None

        self.assertEqual(crop.shape[:2], expected.crop_bgr.shape[:2])
        self.assertEqual(improved.shape[:2], expected.improved_bgr.shape[:2])
        self.assertEqual(bw.shape[:2], expected.bw.shape[:2])
        self.assertEqual(high_contrast.shape[:2], expected.high_contrast.shape[:2])
        self.assertEqual(debug_roi.shape[:2], original.shape[:2])

    def test_single_file_process_endpoints_return_expected_images(self) -> None:
        content, _ = build_challenging_bottle()
        expected = process_image(content, detector_backend="craft")
        body, boundary = build_multipart_body("bottle.jpg", content, "image/jpeg")

        original = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(original)
        assert original is not None

        expectations = [
            ("/process1", "image/jpeg", expected.crop_bgr.shape[:2], cv2.IMREAD_COLOR),
            ("/process2", "image/jpeg", expected.improved_bgr.shape[:2], cv2.IMREAD_COLOR),
            ("/process3", "image/png", expected.bw.shape[:2], cv2.IMREAD_GRAYSCALE),
            ("/process4", "image/jpeg", expected.high_contrast.shape[:2], cv2.IMREAD_COLOR),
            ("/process5", "image/jpeg", original.shape[:2], cv2.IMREAD_COLOR),
        ]

        for path, media_type, expected_shape, imread_mode in expectations:
            with self.subTest(path=path):
                request = urllib.request.Request(
                    url=f"http://127.0.0.1:{self.http_server.port}{path}?detector_backend=craft",
                    data=body,
                    method="POST",
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                )

                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = response.read()
                    content_type = response.headers.get_content_type()

                self.assertEqual(content_type, media_type)
                image = cv2.imdecode(np.frombuffer(payload, np.uint8), imread_mode)
                self.assertIsNotNone(image)
                assert image is not None
                self.assertEqual(image.shape[:2], expected_shape)

    def test_process_returns_json_links_for_dify(self) -> None:
        content, _ = build_challenging_bottle()
        expected = process_image(content, detector_backend="craft")
        body, boundary = build_multipart_body("bottle.jpg", content, "image/jpeg")

        request = urllib.request.Request(
            url=f"http://127.0.0.1:{self.http_server.port}/process?detector_backend=craft&response_format=json_links",
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
            content_type = response.headers.get_content_type()

        self.assertEqual(content_type, "application/json")

        links = json.loads(payload.decode("utf-8"))
        self.assertEqual(
            [link.rsplit("/", 1)[-1] for link in links],
            ["crop_preview.jpg", "improved.jpg", "bw.png", "high_contrast.jpg", "debug_roi.jpg"],
        )

        images = {}
        for link in links:
            with urllib.request.urlopen(link, timeout=30) as image_response:
                filename = link.rsplit("/", 1)[-1]
                image_bytes = image_response.read()
            images[filename] = cv2.imdecode(
                np.frombuffer(image_bytes, np.uint8),
                cv2.IMREAD_GRAYSCALE if filename == "bw.png" else cv2.IMREAD_COLOR,
            )

        for image in images.values():
            self.assertIsNotNone(image)

        crop = images["crop_preview.jpg"]
        improved = images["improved.jpg"]
        bw = images["bw.png"]
        high_contrast = images["high_contrast.jpg"]
        debug_roi = images["debug_roi.jpg"]
        assert crop is not None
        assert improved is not None
        assert bw is not None
        assert high_contrast is not None
        assert debug_roi is not None

        original = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(original)
        assert original is not None

        self.assertEqual(crop.shape[:2], expected.crop_bgr.shape[:2])
        self.assertEqual(improved.shape[:2], expected.improved_bgr.shape[:2])
        self.assertEqual(bw.shape[:2], expected.bw.shape[:2])
        self.assertEqual(high_contrast.shape[:2], expected.high_contrast.shape[:2])
        self.assertEqual(debug_roi.shape[:2], original.shape[:2])


if __name__ == "__main__":
    unittest.main()
