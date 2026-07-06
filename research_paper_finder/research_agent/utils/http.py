import httpx

from research_agent.config import REQUEST_TIMEOUT


client = httpx.AsyncClient(

    timeout=REQUEST_TIMEOUT,
    follow_redirects=True,

    headers={
        "User-Agent": "Research-Agent/1.0"
    },
)