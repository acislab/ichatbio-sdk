import importlib.resources

import instructor
from instructor import AsyncInstructor
from instructor.exceptions import InstructorRetryException
from openai import AsyncOpenAI
from pydantic import Field, BaseModel
from tenacity import AsyncRetrying

from examples.util.ai import StopOnTerminalErrorOrMaxAttempts, AIGenerationException
from ichatbio.types import AgentEntrypoint, ProcessMessage, TextMessage, ArtifactMessage
from ..schema import IDigBioRecordsApiParameters
from ..util import query_idigbio_api, make_idigbio_api_url, make_idigbio_portal_url

description = """\
Searches for species occurrence records using the iDigBio Portal or the iDigBio records API. Returns the total number 
of records that were found, the URL used to call the iDigBio Records API to perform the search, and a URL to view the 
results in the iDigBio Search Portal.
"""

entrypoint = AgentEntrypoint(
    id="find_occurrence_records",
    description=description,
    parameters=None
)


async def run(request: str):
    yield ProcessMessage(summary="Searching iDigBio",
                         description="Generating search parameters for species occurrences")
    try:
        params, artifact_description = await _generate_records_search_parameters(request)
    except AIGenerationException as e:
        yield ProcessMessage(description=e.message)
        return

    yield ProcessMessage(description=f"Generated search parameters", data=params)

    records_api_url = make_idigbio_api_url("/v2/search/records")
    yield ProcessMessage(description=f"Sending a POST request to the iDigBio records API at {records_api_url}")

    response_code, success, response_data = query_idigbio_api("/v2/search/records", params)
    matching_count = response_data.get("itemCount", 0)
    record_count = len(response_data.get("items", []))

    if success:
        yield ProcessMessage(description=f"Response code: {response_code}")
    else:
        yield ProcessMessage(description=f"Response code: {response_code} - something went wrong!")
        return

    api_query_url = make_idigbio_api_url("/v2/search/records", params)
    yield TextMessage(
        text=f"The API query returned {record_count} out of {matching_count} matching records in iDigBio using the"
             f" URL {api_query_url}"
    )

    portal_url = make_idigbio_portal_url(params)
    yield ProcessMessage(
        description=f"[View {record_count} out of {matching_count} matching records]({api_query_url})"
                    f" | [Show in iDigBio portal]({portal_url})"
    )
    yield TextMessage(
        text=f"The records can be viewed in the iDigBio portal at {portal_url}. The portal shows the records in an"
             f" interactive list and plots them on a map. The raw records returned returned by the API can be found"
             f" at {api_query_url}"
    )
    yield ArtifactMessage(
        mimetype="application/json",
        description=artifact_description,
        uris=[api_query_url],
        metadata={
            "data_source": "iDigBio",
            "portal_url": portal_url,
            "retrieved_record_count": record_count,
            "total_matching_count": matching_count
        }
    )


class LLMResponseModel(BaseModel):
    plan: str = Field(description="A brief explanation of what API parameters you plan to use")
    search_parameters: IDigBioRecordsApiParameters = Field()
    artifact_description: str = Field(
        description="A concise characterization of the retrieved occurrence record data",
        examples=["Occurrence records of Rattus rattus",
                  "Occurrence records modified in 2025"])


async def _generate_records_search_parameters(request: str) -> (dict, str):
    client: AsyncInstructor = instructor.from_openai(AsyncOpenAI())

    try:
        result = await client.chat.completions.create(
            model="gpt-4.1",
            temperature=0,
            response_model=LLMResponseModel,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": request}
            ],
            max_retries=AsyncRetrying(stop=StopOnTerminalErrorOrMaxAttempts(3))
        )
    except InstructorRetryException as e:
        raise AIGenerationException(e)

    generation = result.model_dump(exclude_none=True, by_alias=True)
    return generation["search_parameters"], generation["artifact_description"]


SYSTEM_PROMPT_TEMPLATE = """
You translate user requests into parameters for the iDigBio record search API.

# Query format

Here is a description of how iDigBio queries are formatted:

[BEGIN QUERY FORMAT DOC]

{query_format_doc}

[END QUERY FORMAT DOC]

# Examples

{examples_doc}
"""


def get_system_prompt():
    query_format_doc = importlib.resources.files().joinpath("..", "resources", "records_query_format.md").read_text()
    examples_doc = importlib.resources.files().joinpath("..", "resources", "records_examples.md").read_text()

    return SYSTEM_PROMPT_TEMPLATE.format(
        query_format_doc=query_format_doc,
        examples_doc=examples_doc
    ).strip()
