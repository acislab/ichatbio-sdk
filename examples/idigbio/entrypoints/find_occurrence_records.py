import importlib.resources

import instructor
from instructor import AsyncInstructor
from instructor.exceptions import InstructorRetryException
from openai import AsyncOpenAI
from pydantic import Field, BaseModel
from tenacity import AsyncRetrying

from examples.util.ai import StopOnTerminalErrorOrMaxAttempts, AIGenerationException
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.types import AgentEntrypoint
from ..schema import IDigBioRecordsApiParameters
from ..util import query_idigbio_api, make_idigbio_api_url, make_idigbio_portal_url

# This description helps iChatBio understand when to call this entrypoint
description = """\
Searches for species occurrence records using the iDigBio Portal or the iDigBio records API. Returns the total number 
of records that were found, the URL used to call the iDigBio Records API to perform the search, and a URL to view the 
results in the iDigBio Search Portal.
"""

# This gets included in the agent card
entrypoint = AgentEntrypoint(
    id="find_occurrence_records",
    description=description,
    parameters=None
)


async def run(context: ResponseContext, request: str):
    """
    Executes this specific entrypoint. See description above. This function yields a sequence of messages that are
    returned one-by-one to iChatBio in response to the request, logging the retrieval process in real time. Any records
    retrieved from the iDigBio API are packaged as an JSON artifact that iChatBio can interact with.
    """
    async with context.begin_process(summary="Searching iDigBio") as process:
        process: IChatBioAgentProcess

        await process.log("Generating search parameters for species occurrences")
        try:
            params, artifact_description = await _generate_records_search_parameters(request)
        except AIGenerationException as e:
            raise ValueError(e.message) from e

        await process.log("Generated search parameters", data=params)

        records_api_url = make_idigbio_api_url("/v2/search/records")
        await process.log(f"Sending a POST request to the iDigBio records API at {records_api_url}")

        response_code, success, response_data = query_idigbio_api("/v2/search/records", params)
        matching_count = response_data.get("itemCount", 0)
        record_count = len(response_data.get("items", []))

        if success:
            await process.log(f"Response code: {response_code}")
        else:
            await process.log(f"Response code: {response_code} - something went wrong!")
            return

        api_query_url = make_idigbio_api_url("/v2/search/records", params)
        await context.reply(
            f"The API query returned {record_count} out of {matching_count} matching records in iDigBio using the URL"
            f" {api_query_url}"
        )

        portal_url = make_idigbio_portal_url(params)
        await process.log(
            f"[View {record_count} out of {matching_count} matching records]({api_query_url})"
            f" | [Show in iDigBio portal]({portal_url})"
        )
        await context.reply(
            f"The records can be viewed in the iDigBio portal at {portal_url}. The portal shows the records in an"
            f" interactive list and plots them on a map. The raw records returned returned by the API can be found at"
            f" {api_query_url}"
        )
        await process.create_artifact(
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
