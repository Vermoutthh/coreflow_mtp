"""Measure real external tool calls without implementing a new agent framework."""
import time

import httpx


async def observe_tool_call(controller_url, agent_id, tool_name, tool, *args,
                            observation_client=None, **kwargs):
    """Call an async tool, learn only successful durations, preserve its result.

The caller supplies the actual tool implementation and normal authorization.
Reporting failure must not destroy a successful tool result; it is returned
separately for the workflow's audit log. No arbitrary shell/code endpoint is
introduced by this helper.
    """
    started = time.perf_counter()
    value = await tool(*args, **kwargs)
    elapsed = time.perf_counter() - started
    owns = observation_client is None
    client = observation_client or httpx.AsyncClient(timeout=10)
    report = {"agent_id": agent_id, "tool_name": tool_name, "duration_seconds": elapsed}
    try:
        response = await client.post(controller_url.rstrip("/") + "/tools/observe", json=report)
        response.raise_for_status()
        report["reported"] = True
        report["response"] = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        report.update(reported=False, error=str(exc))
    finally:
        if owns:
            await client.aclose()
    return value, report
