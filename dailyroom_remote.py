#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp

from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator, LLMUserResponseAggregator)
from pipecat.processors.frameworks.langchain import LangchainProcessor
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate



from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.agents.openai_assistant.base import OpenAIAssistantV2Runnable
from langchain.agents import AgentExecutor



from loguru import logger

from runner import configure

from dotenv import load_dotenv
load_dotenv(override=True)


logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

message_store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in message_store:
        message_store[session_id] = ChatMessageHistory()
    return message_store[session_id]


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),      
            ),
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            #
            # English
            #
            voice_id="pNInz6obpgDQGcFmaJgB",

            #
            # Spanish
            #
            # model="eleven_multilingual_v2",
            # voice_id="gD1IexrzCvsXPHUuT0s3",
        )

        


        tma_in = LLMUserResponseAggregator()
        tma_out = LLMAssistantResponseAggregator()
        agent = OpenAIAssistantV2Runnable(assistant_id=os.environ.get("OPENAI_ASSISTANT_ID"), as_agent=True)
        agent_executor = AgentExecutor(agent=agent, tools=[])
        lc = LangchainProcessor(agent_executor, transcript_key='content')

        pipeline = Pipeline(
            [
                transport.input(),      # Transport user input
                tma_in,                 # User response aggregator
                lc,                     # Langchain
                tts,                    # TTS
                transport.output(),     # Transport bot output
                tma_out,                # Assistant response aggregator
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])
            lc.set_participant_id(participant["id"])
            # Kick off the conversation.
            # the `LLMMessagesFrame` will be picked up by the LangchainProcessor using
            # only the content of the last message to inject it in the prompt defined
            # above. So no role is required here.
            messages = [(
                {
                    "content": "Please briefly introduce yourself to the user."
                }
            )]
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))