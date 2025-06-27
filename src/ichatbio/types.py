import re
from typing import Optional, Type, Annotated

from annotated_types import MinLen
from pydantic import BaseModel, AnyHttpUrl, StringConstraints, WithJsonSchema

IDString = Annotated[str, StringConstraints(min_length=2, pattern=r"^[a-zA-Z0-9_-]+$")]


class AgentEntrypoint(BaseModel):
    """
    Defines how iChatBio interacts with an agent. Messages from iChatBio are required to comply with
    this data model. Validation of the model is performed by the agent. Messages that violate this model
    will be returned to iChatBio.
    """
    id: IDString
    """The identifier for this entrypoint. Can only contain letters, numbers, and underscores. Try to make the ID informative and concise. For example, "search_idigbio"."""

    description: str
    """An explanation of what the agent can do through this entrypoint."""

    parameters: Optional[Type[BaseModel]] = None
    """Structured information that iChatBio must provide to use this entrypoint."""


class AgentCard(BaseModel):
    """
    Provides iChatBio with information about an agent and rules for interacting with it.
    """
    name: str
    """The name used to identify the agent to iChatBio users."""

    description: str
    """Describes the agent to both the iChatBio assistant and users."""

    icon: Optional[str] = None
    """URL for the image shown to iChatBio users to visually reference this agent."""

    url: Annotated[Optional[str], AnyHttpUrl] = None
    """URL at which the agent receives requests."""

    entrypoints: Annotated[list[AgentEntrypoint], MinLen(1)]
    """Defines how iChatBio can interact with this agent."""


_URL_PATTERN = re.compile(r"^https?://")


class _ArtifactDescription(BaseModel):
    local_id: str
    """Locally identifies the artifact in the context of an iChatBio conversation."""

    description: str
    """A brief (~50 characters) description of the artifact."""

    mimetype: str
    """The MIME type of the artifact, e.g. ``text/plain``, ``application/json``, ``image/png``."""

    uris: list[str]
    """Identifiers associated with the artifact. Usually, one of these URIs is a resolvable URL."""

    metadata: dict
    """Anything related to the artifact, e.g. provenance, schema, landing page URLs, related artifact URIs."""

    def get_urls(self) -> list[str]:
        return [uri for uri in self.uris if _URL_PATTERN.match(uri)]


Artifact = Annotated[
    _ArtifactDescription,
    WithJsonSchema(
        {
            "type": "string",
            "pattern": "^#[0-9a-f]{4}$",
            "examples": ["#0a9f"]
        }
    )
]
"""
An entrypoint parameter to instruct iChatBio to provide an artifact description, which contains metadata about the
artifact and how to access its content.
"""
