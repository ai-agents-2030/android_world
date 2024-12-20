# Copyright 2024 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Some LLM inference interface."""

import abc
import base64
import io
import os
import time
from typing import Any
import google.generativeai as genai
from google.generativeai.types import generation_types
import numpy as np
from PIL import Image
import requests


ERROR_CALLING_LLM = 'Error calling LLM'


def _array_to_jpeg_bytes(image: np.ndarray) -> bytes:
  """Converts a numpy array into a byte string for a JPEG image."""
  image = Image.fromarray(image)
  in_mem_file = io.BytesIO()
  image.save(in_mem_file, format='JPEG')
  # Reset file pointer to start
  in_mem_file.seek(0)
  img_bytes = in_mem_file.read()
  return img_bytes


class LlmWrapper(abc.ABC):
  """Abstract interface for (text only) LLM."""

  @abc.abstractmethod
  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Any]:
    """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.

    Returns:
      Text output and raw output.
    """


class MultimodalLlmWrapper(abc.ABC):
  """Abstract interface for Multimodal LLM."""

  @abc.abstractmethod
  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Any]:
    """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.
      images: List of images as numpy ndarray.

    Returns:
      Text output and raw output.
    """


class GeminiGcpWrapper(LlmWrapper, MultimodalLlmWrapper):
  """Gemini GCP interface."""

  # As of 05/15/2024, there is a limit of 5 RPM:
  # https://cloud.google.com/vertex-ai/generative-ai/docs/quotas.
  TIME_BETWEEN_REQUESTS = 15

  def __init__(
      self,
      model_name: str | None = None,
      max_retry: int = 3,
      temperature: float = 0.0,
      top_p: float = 0.95,
  ):
    if 'GCP_API_KEY' not in os.environ:
      raise RuntimeError('GCP API key not set.')
    genai.configure(api_key=os.environ['GCP_API_KEY'])
    self.llm = genai.GenerativeModel(
        model_name,
        generation_config=generation_types.GenerationConfig(
            temperature=temperature, top_p=top_p
        ),
    )
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)
    self.time_of_last_request = time.time() - self.TIME_BETWEEN_REQUESTS

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Any]:
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Any]:
    counter = self.max_retry
    retry_delay = self.TIME_BETWEEN_REQUESTS
    while counter > 0:
      try:
        time_since_last_request = time.time() - self.time_of_last_request
        if time_since_last_request < self.TIME_BETWEEN_REQUESTS:
          time.sleep(self.TIME_BETWEEN_REQUESTS - time_since_last_request)
        self.time_of_last_request = time.time()
        output = self.llm.generate_content(
            [text_prompt] + [Image.fromarray(image) for image in images]
        )
        return output.text, output
      except Exception as e:  # pylint: disable=broad-exception-caught
        counter -= 1
        print('Error calling LLM, will retry in {retry_delay} seconds')
        print(e)
        if counter > 0:
          time.sleep(retry_delay)
          retry_delay *= 2
    return ERROR_CALLING_LLM, None


class Gpt4Wrapper(LlmWrapper, MultimodalLlmWrapper):
  """OpenAI GPT4 wrapper.

  Attributes:
    openai_api_key: The class gets the OpenAI api key either explicitly, or
      through env variable in which case just leave this empty.
    max_retry: Max number of retries when some error happens.
    temperature: The temperature parameter in LLM to control result stability.
    model: GPT model to use based on if it is multimodal.
  """

  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      model_name: str,
      max_retry: int = 3,
      temperature: float = 0.0,
  ):
    if 'OPENAI_API_KEY' not in os.environ:
      raise RuntimeError('OpenAI API key not set.')
    self.openai_api_key = os.environ['OPENAI_API_KEY']
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)
    self.temperature = temperature
    self.model = model_name

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    return base64.b64encode(_array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Any]:
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Any]:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {self.openai_api_key}',
    }

    payload = {
        'model': self.model,
        'temperature': self.temperature,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': text_prompt},
            ],
        }],
        'max_tokens': 1000,
    }

    # Gpt-4v supports multiple images, just need to insert them in the content
    # list.
    for image in images:
      payload['messages'][0]['content'].append({
          'type': 'image_url',
          'image_url': {
              'url': f'data:image/jpeg;base64,{self.encode_image(image)}'
          },
      })

    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS
    while counter > 0:
      try:
        response = requests.post(
            #'https://api.openai.com/v1/chat/completions',
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") + '/chat/completions',
            headers=headers,
            json=payload,
        )
        if response.ok and 'choices' in response.json():
          return response.json()['choices'][0]['message']['content'], response
        print(
            'Error calling OpenAI API with error message: '
            + response.json()['error']['message']
        )
        time.sleep(wait_seconds)
        wait_seconds *= 2
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Want to catch all exceptions happened during LLM calls.
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter -= 1
        print('Error calling LLM, will retry soon...')
        print(e)
    return ERROR_CALLING_LLM, None
