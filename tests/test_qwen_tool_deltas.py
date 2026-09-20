#!/usr/bin/env python3
"""Opt-in live Qwen SSE test. Reports tool calls; never executes them."""

import argparse
import json
import time
import urllib.request


def check(base_url, api_key, model, thinking, anthropic):
    schema = {
        "type": "object",
        "properties": {"content": {"type": "string"}},
        "required": ["content"],
    }
    tool = {"name": "write_report", "description": "Write a report."}
    message = {
        "role": "user",
        "content": "Call write_report once. Set content to a roughly 100-word explanation "
        "of how rain forms. Approximate length is fine; do not count words. "
        "Do not answer in plain text.",
    }
    body = {
        "model": model,
        "messages": [message],
        "stream": True,
        "temperature": 0,
        "max_tokens": 2048 if thinking else 1024,
    }
    if anthropic:
        tool["input_schema"] = schema
        body["tools"] = [tool]
        body["thinking"] = {"type": "enabled", "budget_tokens": 1024} if thinking else {"type": "disabled"}
        path = "/v1/messages"
    else:
        tool["parameters"] = schema
        body["tools"] = [{"type": "function", "function": tool}]
        body["reasoning_effort"] = "high" if thinking else "none"
        path = "/v1/chat/completions"
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}",
                 "x-api-key": api_key, "anthropic-version": "2023-06-01"},
    )
    calls = {}
    first_args = last_args = finished = None
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                continue
            event = json.loads(data)
            now = time.monotonic() - started
            if anthropic:
                if event["type"] == "error":
                    raise AssertionError(event)
                if event["type"] == "content_block_start":
                    block = event["content_block"]
                    if block["type"] == "tool_use":
                        calls[event["index"]] = {"id": block["id"], "name": block["name"], "parts": []}
                if event["type"] == "content_block_delta" and event["delta"]["type"] == "input_json_delta":
                    frag = event["delta"]["partial_json"]
                    calls[event["index"]]["parts"].append(frag)
                    if frag:
                        first_args = now if first_args is None else first_args
                        last_args = now
                if event["type"] == "message_delta":
                    assert event["delta"]["stop_reason"] == "tool_use", event
                    finished = now
            else:
                assert "error" not in event, event
                for choice in event.get("choices", []):
                    for call in choice.get("delta", {}).get("tool_calls", []):
                        index = call["index"]
                        fn = call.get("function", {})
                        if "id" in call:
                            assert index not in calls, "duplicate tool start"
                            calls[index] = {"id": call["id"], "name": fn["name"], "parts": []}
                        frag = fn.get("arguments", "")
                        if frag:
                            calls[index]["parts"].append(frag)
                            first_args = now if first_args is None else first_args
                            last_args = now
                    if choice.get("finish_reason"):
                        assert choice["finish_reason"] == "tool_calls", choice
                        finished = now
    assert calls and finished is not None, "no completed tool call"
    for call in calls.values():
        assert call["name"] == "write_report" and call["id"], call
        args = json.loads("".join(call["parts"]))
        assert isinstance(args.get("content"), str) and len(args["content"]) > 100, args
        assert len(call["parts"]) > 5, "arguments were buffered until completion"
    assert first_args < last_args <= finished
    print(json.dumps({
        "api": "anthropic" if anthropic else "openai", "thinking": thinking,
        "calls": len(calls), "argument_deltas": sum(len(c["parts"]) for c in calls.values()),
        "first_arguments_s": round(first_args, 3), "last_arguments_s": round(last_args, 3),
        "finish_s": round(finished, 3),
    }), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18081")
    parser.add_argument("--api-key", default="test")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    for anthropic in (False, True):
        check(args.base_url, args.api_key, args.model, args.thinking, anthropic)
