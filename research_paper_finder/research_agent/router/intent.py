from enum import Enum


class SearchIntent(str, Enum):

    EXACT_TITLE = "exact_title"

    TOPIC = "topic"

    LATEST = "latest"

    MOST_CITED = "most_cited"

    AUTHOR = "author"

    UNKNOWN = "unknown"