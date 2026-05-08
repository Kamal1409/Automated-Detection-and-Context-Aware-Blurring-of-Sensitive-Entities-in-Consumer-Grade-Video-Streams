import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request


class LLM:
    def __init__(self, api_base=None, api_key=None, model=None, timeout_seconds=None):
        base = api_base or os.getenv(
            "LLM_API_BASE", "https://generativelanguage.googleapis.com/v1beta"
        )
        self.api_base = str(base).strip().rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "gemini-2.5-flash")
        if timeout_seconds is None:
            env_timeout = os.getenv("LLM_TIMEOUT_SECONDS", "")
            if env_timeout:
                try:
                    timeout_seconds = int(env_timeout)
                except ValueError:
                    timeout_seconds = None
        self.timeout_seconds = int(timeout_seconds) if timeout_seconds else 18
        env_retries = os.getenv("LLM_MAX_RETRIES", "2")
        try:
            self.max_retries = max(0, min(6, int(env_retries)))
        except ValueError:
            self.max_retries = 2
        self.enabled = bool(self.api_key and self.api_base and self.model)

    def _normalize_model(self, model):
        raw = str(model).strip()
        if not raw:
            return "models/gemini-2.5-flash"
        if raw.startswith("models/"):
            return raw
        return f"models/{raw}"

    def _post_json(self, url, payload):
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        request = urllib.request.Request(url, data=data, method="POST")
        request.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)

    def _post_json_with_retries(self, url, payload):
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._post_json(url, payload)
            except urllib.error.HTTPError as ex:
                status = getattr(ex, "code", None)
                try:
                    body = ex.read().decode("utf-8", errors="ignore")
                except Exception:
                    body = ""
                last_error = f"HTTP {status}: {body or ex.reason}"
                retryable = status in {408, 429, 500, 502, 503, 504}
                if not retryable or attempt >= self.max_retries:
                    raise RuntimeError(last_error) from ex
            except urllib.error.URLError as ex:
                last_error = f"URL error: {ex.reason}"
                if attempt >= self.max_retries:
                    raise RuntimeError(last_error) from ex

            delay_seconds = min(8.0, 0.6 * (2**attempt))
            time.sleep(delay_seconds)

        raise RuntimeError(last_error or "request_failed")

    def select_primary_tracks(
        self, video_context, track_summaries, primary_subject_count=1, user_prompt=None
    ):
        meta = {"error": None, "reason": None, "confidence": None}
        if not self.enabled:
            meta["error"] = "disabled"
            return None, meta
        if not track_summaries:
            meta["error"] = "no_tracks"
            return None, meta

        payload_data = {
            "video_context": video_context or {},
            "primary_subject_count": int(primary_subject_count),
            "track_summaries": track_summaries,
        }

        system_prompt = (
            "You decide which people are the primary subjects of a video. "
            "Return JSON only with keys: keep_track_ids (list of ints), reason, confidence (0..1). "
        )
        if user_prompt:
            system_prompt += str(user_prompt)
        else:
            system_prompt += (
                "Detect all human faces in the video and keep only the primary subject’s face visible. Blur every other detected face throughout the entire video while maintaining smooth tracking across frames. The selected face should remain completely unblurred even when moving, turning, or partially occluded. Apply a strong Gaussian blur to all non-selected faces and ensure no flickering or missed frames occur."
            )

        data_prompt = "DATA:\n" + json.dumps(payload_data, ensure_ascii=True)

        request_payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": data_prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }

        try:
            model_name = self._normalize_model(self.model)
            query = urllib.parse.urlencode({"key": self.api_key})
            url = f"{self.api_base}/{model_name}:generateContent?{query}"
            response = self._post_json_with_retries(url, request_payload)
        except (urllib.error.URLError, ValueError, RuntimeError) as ex:
            meta["error"] = f"request_failed: {ex}"
            return None, meta

        try:
            parts = response["candidates"][0]["content"]["parts"]
            content = "".join(str(part.get("text", "")) for part in parts)
        except (KeyError, IndexError, TypeError):
            meta["error"] = "invalid_response"
            return None, meta

        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            meta["error"] = "invalid_json"
            return None, meta

        keep_ids = parsed.get("keep_track_ids") or parsed.get("keep_track_id") or []
        normalized_ids = []
        for item in keep_ids:
            try:
                normalized_ids.append(int(item))
            except (TypeError, ValueError):
                continue

        if normalized_ids:
            normalized_ids = normalized_ids[: max(1, int(primary_subject_count))]

        meta["reason"] = parsed.get("reason")
        confidence = parsed.get("confidence")
        if isinstance(confidence, (float, int)):
            meta["confidence"] = max(0.0, min(1.0, float(confidence)))

        return normalized_ids or None, meta
