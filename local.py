#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import aiohttp
import os
import sys
import tkinter as tk

from pipecat.frames.frames import EndFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.xtts import XTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.local.audio import LocalAudioTransport
from pipecat.transports.local.tk import TkLocalTransport

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    async with aiohttp.ClientSession() as session:
        tk_root = tk.Tk()
        tk_root.title("Calendar")
        #transport = LocalAudioTransport(TransportParams(audio_out_enabled=True,audio_in_enabled=True))
        transport = TkLocalTransport(
            tk_root,
            TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1024,
                camera_out_height=1024))

        tts = XTTSService(
            aiohttp_session=session,
            voice_id="Claribel Dervla",
            language="en",
            base_url="http://localhost:8000"
        )

        pipeline = Pipeline([tts, transport.output()])

        task = PipelineTask(pipeline)

        async def say_something():
            await asyncio.sleep(1)
            await task.queue_frames([TextFrame("Hello there!"), EndFrame()])

        runner = PipelineRunner(handle_sigint=False)

        await asyncio.gather(runner.run(task), say_something())

if __name__ == "__main__":
    asyncio.run(main())