"""
Google Gemini LLM Module for PSW Policy Assistant.
Uses the official google-genai SDK, implements grounded generation,
conservative parameters, model fallbacks, and resilient error handling.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

from config import (
    DEFAULT_GEMINI_MODEL,
    FALLBACK_GEMINI_MODELS,
    GENERATION_TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    get_gemini_api_key,
)
from src.prompts import SYSTEM_INSTRUCTION, construct_user_prompt
from src.retrieval import RetrievedPassage

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response object from LLM invocation."""
    content: str
    model_used: str
    success: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class GeminiClient:
    """Wrapper for Google GenAI SDK with grounded system instruction and resilient error handling."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        if api_key is not None:
            self.api_key = api_key.strip()
        else:
            self.api_key = (get_gemini_api_key() or "").strip()
        self.model_name = (model_name or DEFAULT_GEMINI_MODEL).strip()
        self._client: Optional[genai.Client] = None

        if self.api_key:
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Google GenAI client: {self._sanitize_error(e)}")

    def _sanitize_error(self, err: Exception) -> str:
        """Strips potential API keys from error messages before display or logging."""
        msg = str(err)
        if self.api_key and len(self.api_key) > 8:
            msg = msg.replace(self.api_key, "[PROTECTED_API_KEY]")
        return msg

    @property
    def is_configured(self) -> bool:
        """True if an API key is provided and the client is ready."""
        return self._client is not None and bool(self.api_key)

    def generate_grounded_answer(
        self,
        question: str,
        passages: List[RetrievedPassage],
        conversation_context: Optional[str] = None,
        override_model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Sends the grounded prompt containing retrieved policy evidence to Gemini.
        Handles rate limits, quota limits, model availability, and errors gracefully.
        """
        if not self.is_configured:
            return LLMResponse(
                content=(
                    "⚠️ **Gemini API Key Required**\n\n"
                    "Please configure a valid Google Gemini API key to generate grounded answers. "
                    "You can enter your API key in the sidebar or set `GEMINI_API_KEY` in your `.env` file."
                ),
                model_used="none",
                success=False,
                error_type="missing_api_key",
                error_message="Gemini API key is not configured.",
            )

        user_prompt = construct_user_prompt(
            question=question,
            passages=passages,
            conversation_context=conversation_context,
        )

        models_to_try = [override_model or self.model_name]
        for fb in FALLBACK_GEMINI_MODELS:
            if fb not in models_to_try:
                models_to_try.append(fb)

        last_error_type = None
        last_error_msg = None

        for target_model in models_to_try:
            try:
                config = types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=GENERATION_TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                )

                response = self._client.models.generate_content(
                    model=target_model,
                    contents=user_prompt,
                    config=config,
                )

                response_text = response.text if response and hasattr(response, "text") and response.text else ""

                if not response_text.strip():
                    # Handle case where response might be filtered or empty
                    if response and response.candidates:
                        for cand in response.candidates:
                            if cand.content and cand.content.parts:
                                for part in cand.content.parts:
                                    if hasattr(part, "text") and part.text:
                                        response_text += part.text

                if response_text.strip():
                    return LLMResponse(
                        content=response_text.strip(),
                        model_used=target_model,
                        success=True,
                    )

            except ClientError as ce:
                sanitized_msg = self._sanitize_error(ce)
                code = getattr(ce, "code", None) or (ce.args[0] if ce.args else 0)

                # Check 404 (Model not found or deprecated) -> try next fallback
                if "404" in sanitized_msg or "NOT_FOUND" in sanitized_msg:
                    logger.warning(f"Model '{target_model}' not found or deprecated. Trying fallback...")
                    last_error_type = "model_not_found"
                    last_error_msg = f"Model '{target_model}' not available."
                    continue

                # Check 429 Quota exhaustion or Rate Limit
                if "429" in sanitized_msg or "RESOURCE_EXHAUSTED" in sanitized_msg:
                    logger.warning("Gemini rate limit or quota exceeded.")
                    return LLMResponse(
                        content=(
                            "⚠️ **Gemini Free-Tier Quota Limit Reached**\n\n"
                            "The Google Gemini free-tier rate limit or quota has been reached. "
                            "Please wait 30–60 seconds before submitting another question. "
                            "The system remains fully functional and will process your query once the limit resets."
                        ),
                        model_used=target_model,
                        success=False,
                        error_type="quota_exhausted",
                        error_message="Gemini API rate limit or quota exceeded (429 Resource Exhausted).",
                    )

                # Invalid API key
                if "400" in sanitized_msg and ("API_KEY_INVALID" in sanitized_msg or "API key not valid" in sanitized_msg):
                    return LLMResponse(
                        content=(
                            "⚠️ **Invalid Gemini API Key**\n\n"
                            "The provided Google Gemini API key appears to be invalid. "
                            "Please check your API key in the sidebar or update the `.env` file."
                        ),
                        model_used=target_model,
                        success=False,
                        error_type="invalid_api_key",
                        error_message="Provided Gemini API key was rejected by Google GenAI.",
                    )

                last_error_type = "client_error"
                last_error_msg = sanitized_msg

            except APIError as ae:
                sanitized_msg = self._sanitize_error(ae)
                logger.error(f"Google GenAI APIError with {target_model}: {sanitized_msg}")
                last_error_type = "api_error"
                last_error_msg = sanitized_msg

            except Exception as e:
                sanitized_msg = self._sanitize_error(e)
                logger.error(f"Unexpected error generating answer with {target_model}: {sanitized_msg}")
                last_error_type = "general_error"
                last_error_msg = sanitized_msg

        # If all candidate models failed
        return LLMResponse(
            content=(
                f"⚠️ **Error Generating Response**\n\n"
                f"Unable to generate answer from Gemini ({last_error_type or 'Service Error'}). "
                f"Details: {last_error_msg or 'Please check your connection and configuration.'}"
            ),
            model_used=models_to_try[0],
            success=False,
            error_type=last_error_type,
            error_message=last_error_msg,
        )
