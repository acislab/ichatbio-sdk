import re
from typing import Optional, Type, Annotated

import pydantic
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


class ProcessMessage(BaseModel):
    """
    Tells iChatBio users what the agent is doing. Send multiple process messages to provide updates for long-running
    processes.
    """

    summary: Optional[str] = None
    """A brief summary of what the agent is doing, e.g. "Searching iDigBio". Overrides the summary set by any prior
    ProcessMessages for the current request. Set to None to preserve the current summary (if one exists)."""

    description: Optional[str] = None
    """Freeform text to more thoroughly describe agent processes. Uses Markdown formatting."""

    data: Optional[dict] = None
    """Structured information related to the process."""


class TextMessage(BaseModel):
    """
    Responds directly to the iChatBio assistant, not the user. Text messages can be used to:
    - Request more information
    - Refuse the assistant's request
    - Provide context for process and artifact messages
    - Provide advice on what to do next
    - etc.
    """

    text: Optional[str]
    """A natural language response to the assistant's request."""

    data: Optional[dict] = None
    """Structured information related to the message."""


class ArtifactMessage(BaseModel):
    """
    Provides any kind of content that should be identifiable via one or more URIs. If content is not included,
    a resolvable URI must be specified. If no resolvable URIs are provided, iChatBio will store the content and use its
    SHA-256 hash as its identifier.
    """

    mimetype: str
    """The MIME type of the artifact, e.g. ``text/plain``, ``application/json``, ``image/png``."""

    description: str
    """A brief (~50 characters) description of the artifact."""

    uris: Optional[list[str]] = None
    """Unique identifiers for the artifact. If URIs are resolvable, content can be omitted."""

    content: Optional[bytes] = None
    """The raw content of the artifact."""

    metadata: Optional[dict] = None
    """Anything related to the artifact, e.g. provenance, schema, landing page URLs, related artifact URIs."""

    @pydantic.model_validator(mode="after")
    def validate_content(self):
        if not self.content and not self.uris:
            raise ValueError("Either content or uris must be specified.")
        return self


Message = ProcessMessage | TextMessage | ArtifactMessage

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
