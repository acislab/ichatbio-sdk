from typing import Optional

from pydantic import BaseModel


class ProcessMessage(BaseModel):
    """
    Tells iChatBio users what the agent is doing. Send multiple process messages to provide updates for long-running
    processes.
    """

    summary: str
    """A brief summary ("Searching iDigBio") of what the agent is doing."""

    description: Optional[str]
    """Freeform text to more thoroughly describe agent processes. Uses Markdown formatting."""

    data: Optional[dict]
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
    """The MIME type of the artifact, e.g. text/plain, application/json, image/png."""

    description: str
    """A brief (~50 characters) description of the artifact."""

    uris: Optional[list[str]] = None
    """Unique identifiers for the artifact. If URIs are resolvable, content can be omitted."""

    content: Optional[bytes] = None
    """The raw content of the artifact, e.g. JSON records or a Base64 encoded image."""

    metadata: Optional[dict] = None
    """Anything related to the artifact, e.g. provenance, schema, landing page URLs, related artifact URIs."""


Message = ProcessMessage | TextMessage | ArtifactMessage
