from __future__ import annotations

import asyncio
import json
import logging
import math
import string
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

import jinja2
import msgspec
from fastapi import Request
from fastapi.responses import ORJSONResponse
from transformers import PreTrainedTokenizerBase

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageContentImageURL,
    DecisionAnswer,
    DecisionChoiceQuestion,
    DecisionQuestion,
    DecisionRequest,
    DecisionResponse,
    DecisionScoreQuestion,
    DecisionText,
    UsageInfo,
    is_blank_decision_text,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
# Ported from v0.5.23: this tree keeps the chat-template client-error tuple inline
# in serving_chat.py, so define the same tuple here instead of importing it.
_CHAT_TEMPLATE_CLIENT_ERRORS: Tuple[type[BaseException], ...] = (
    jinja2.TemplateError,
    TypeError,
)

from sglang.srt.parser.jinja_template_utils import process_content_for_template_format
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.utils import ImageData

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    from sglang.srt.multimodal.processors.base_processor import (
        MultimodalSpecialTokens,
    )

logger = logging.getLogger(__name__)

# (rendered prompt text, prompt token ids, label token ids, images)
EncodedQuestion = Tuple[str, List[int], List[int], List[ImageData]]

# Version of the server-owned prompt wording and answer labels.
# Any change to a rendering /v1/decisions can produce needs a new version.
PROMPT_FORMAT_VERSION = 1

# Parser defaults that name the chat template kwarg toggling reasoning.
_PARSER_TOGGLE_MODES = (
    "thinking",
    "enable_thinking",
    "explicit_thinking",
    "explicit_enable_thinking",
)
# Answer text of a finished reply, rendered only to see what precedes an answer.
_REPLY_SENTINEL = "DECISION_ANSWER"


class QuestionView(msgspec.Struct, frozen=True):
    """A question as the renderer and scorer see it, shared by the decision routes."""

    # choice, score, or yes_no
    kind: str
    # None or blank when the question has no text of its own
    question: Any
    # Option names, level indices, or yes and no, in candidate order
    names: List[str]
    # Option descriptions, levels, or the yes and no descriptions
    details: List[Any]


def _probe_prompt_text_round_trip(tokenizer_manager) -> bool:
    """Does rendering the chat template to text and re-encoding lose anything?"""
    tokenizer = tokenizer_manager.tokenizer
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return True
    probe = [{"role": "user", "content": "x"}]
    try:
        rendered = tokenizer.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True, return_dict=False
        )
        encode_kwargs = (
            {"add_special_tokens": False}
            if getattr(tokenizer, "add_bos_token", False)
            else {}
        )
        via_text = tokenizer.encode(rendered, **encode_kwargs)
        via_ids = tokenizer.apply_chat_template(
            probe, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        return list(via_text) != list(via_ids)
    except Exception:
        # A template that needs kwargs this probe does not supply tells us
        # nothing; keep the long-standing text path.
        return False


class OpenAIServingDecisions(OpenAIServingBase):
    """Handler for /v1/decisions requests, answered by candidate scoring without generation"""

    # Named in setup refusals, so each decision route reports itself.
    route = "/v1/decisions"

    def __init__(self, chat_serving: OpenAIServingChat):
        super().__init__(chat_serving.tokenizer_manager)
        # Render the way the chat route does, and refuse where it renders differently.
        self.template_manager = chat_serving.template_manager
        # Ported from v0.5.23 onto this tree's serving architecture. The three
        # chat-server accessors and the three decision-head configs belong to the
        # newer runtime; absent here, fall back to the conservative behaviour:
        # no server-wide template kwargs, no special prompt encoder, verify the
        # full rendered prompt (no lossy-text shortcut), and no decision/joint/
        # pointer heads (answers come from candidate scoring through /v1/score).
        self.default_chat_template_kwargs = getattr(
            chat_serving, "default_chat_template_kwargs", {}
        )
        self.chat_encoding_spec = getattr(chat_serving, "chat_encoding_spec", None)
        # v0.5.23 keeps this probe in the chat server; compute the same flag
        # here so the route still refuses tokenizers that cannot round-trip the
        # rendered prompt text (e.g. mistral_common control tokens).
        self.prompt_text_is_lossy = _probe_prompt_text_round_trip(
            self.tokenizer_manager
        )
        model_config = self.tokenizer_manager.model_config
        self.decision_config = getattr(model_config, "decision_config", None)
        self.joint_head_config = getattr(model_config, "joint_head_config", None)
        self.pointer_head = getattr(model_config, "pointer_head", None)
        tokenizer = self.tokenizer_manager.tokenizer
        # Other tokenizers skip the shortcut in _encode_labels and check the full prompt.
        self.added_tokens = (
            {i: token for token, i in tokenizer.get_added_vocab().items()}
            if isinstance(tokenizer, PreTrainedTokenizerBase)
            else {}
        )
        # The configured reasoning parser, or the one the chat template suggests,
        # tells where reasoning blocks start and end and whether answers open one.
        parser = (
            chat_serving.reasoning_parser
            or self.template_manager.suggested_reasoning_parser
        )
        self.reasoning_markers = None
        self.answers_open_reasoning = False
        # The kwarg that turns reasoning on and off: the template's, else, when
        # detection finds no config, the one the configured or suggested parser names.
        config = self.template_manager.reasoning_config
        self.reasoning_toggle = config.toggle_param if config is not None else None
        if parser is not None:
            try:
                # v0.5.23 passes the tokenizer so detectors can read special
                # tokens; this tree's ReasoningParser takes only the model type,
                # which is enough for the think-block markers used here.
                detector = ReasoningParser(model_type=parser).detector
            except ValueError as e:
                logger.warning(
                    "No reasoning block check for %s with parser '%s': %s",
                    self.route,
                    parser,
                    e,
                )
            else:
                mode = detector.reasoning_default
                if config is None and mode in _PARSER_TOGGLE_MODES:
                    self.reasoning_toggle = mode.removeprefix("explicit_")
                if detector.think_start_token and detector.think_end_token:
                    self.reasoning_markers = (
                        detector.think_start_token,
                        detector.think_end_token,
                    )
                    self.answers_open_reasoning = detector.reasoning_default == "always"

    def _request_id_prefix(self) -> str:
        return "decision-"

    def _validate_request(self, request: DecisionRequest) -> Optional[str]:
        error = self._validate_server(request.model) or self._validate_images(
            request.images
        )
        if error is not None:
            return error
        if self.decision_config is not None:
            return (
                f"{self.route} does not serve decision checkpoints, use /v1/systemone, "
                "which renders the prompt and answer codes they were trained with"
            )
        if request.images and request.return_prompt_token_ids:
            return (
                "return_prompt_token_ids cannot replay image inputs through /v1/score"
            )
        version = request.prompt_format_version
        if version is not None and version != PROMPT_FORMAT_VERSION:
            return (
                f"prompt_format_version {version} is not served, this server "
                f"uses version {PROMPT_FORMAT_VERSION}"
            )
        return self._validate_reasoning(request.chat_template_kwargs)

    def _validate_server(self, model: str) -> Optional[str]:
        """Refuse servers this route cannot render faithfully."""
        route = self.route
        if not self.tokenizer_manager.is_generation:
            return f"{route} requires a generation model"
        if self.tokenizer_manager.tokenizer is None:
            return f"{route} requires the server tokenizer"
        if self.chat_encoding_spec is not None:
            return (
                f"{route} requires a chat template, but this model's chat "
                f"route uses the {self.chat_encoding_spec!r} encoder"
            )
        if self.prompt_text_is_lossy:
            return (
                f"{route} places answer labels on the rendered chat text, "
                "which this tokenizer does not encode back to the same ids"
            )
        if self.template_manager.chat_template_name is not None:
            return (
                f"{route} renders the tokenizer's Jinja chat template, but "
                "this server uses the built-in chat template "
                f"{self.template_manager.chat_template_name!r}"
            )
        # Ported from v0.5.23: `get_exec()` is a newer runtime-context accessor
        # that this tree does not have; the two guards read the same resolved
        # flags from the serving object's server_args.
        server_args = self.tokenizer_manager.server_args
        if getattr(server_args, "enable_mis", False):
            return f"{route} does not support --enable-mis"
        dllm = getattr(server_args, "dllm_algorithm", None)
        if dllm is not None:
            return (
                f"{route} does not support diffusion language models "
                "served with --dllm-algorithm"
            )
        _, adapter = self._parse_model_parameter(model)
        if adapter is not None:
            return (
                f"model names the LoRA adapter {adapter!r}, which {route} "
                "does not support"
            )
        return None

    def _validate_images(
        self, images: List[ChatCompletionMessageContentImageURL]
    ) -> Optional[str]:
        if not images:
            return None
        if not self.tokenizer_manager.model_config.is_multimodal:
            return (
                f"{self.route} images require a model served with multimodal "
                "input, and some vision models need --enable-multimodal"
            )
        if self.template_manager.jinja_template_content_format != "openai":
            return f"{self.route} images require a chat template accepting image parts"
        if self.tokenizer_manager.allow_auto_truncate:
            return (
                f"{self.route} images require --allow-auto-truncate to be disabled "
                "to preserve the answer position after image processing"
            )
        return None

    def _validate_reasoning(
        self, chat_template_kwargs: Dict[str, Any]
    ) -> Optional[str]:
        """The answer position must follow the reasoning block, not sit inside it."""
        config = self.template_manager.reasoning_config
        if config is not None and config.always_on:
            return (
                f"{self.route} does not support chat templates that always "
                "reason before answering"
            )
        toggle = self.reasoning_toggle
        if toggle in chat_template_kwargs and chat_template_kwargs[toggle] is not False:
            return (
                f"chat_template_kwargs sets {toggle!r} to "
                f"{chat_template_kwargs[toggle]!r}, but decisions need it false or unset"
            )
        return None

    def _chat_template_kwargs(self, request_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Reasoning off, then the server defaults, then the request kwargs."""
        kwargs = {}
        if self.reasoning_toggle is not None:
            kwargs[self.reasoning_toggle] = False
        for key, value in self.default_chat_template_kwargs.items():
            kwargs.setdefault(key, value)
        kwargs.update(request_kwargs)
        return kwargs

    def _convert_to_internal_request(
        self,
        request: DecisionRequest,
        raw_request: Request = None,
    ) -> Tuple[Iterator[EncodedQuestion], DecisionRequest]:
        # Lazy, so the handler can let other requests run between questions.
        return self._encoded_questions(request), request

    def _encoded_questions(self, request: DecisionRequest) -> Iterator[EncodedQuestion]:
        """Prompt and label ids for each question, in request order."""
        text = render_text(request.input)
        chat_template_kwargs = self._chat_template_kwargs(request.chat_template_kwargs)
        for question in request.questions:
            view = _decision_view(question)
            labels = default_labels(view)
            try:
                encoded = self._encode_question(
                    content=render_question(text=text, view=view, labels=labels),
                    labels=labels,
                    chat_template_kwargs=chat_template_kwargs,
                    images=request.images,
                )
            except ValueError as e:
                raise ValueError(f"question {question.id!r}: {e}") from e
            yield encoded

    def _encode_question(
        self,
        content: str,
        labels: List[str],
        chat_template_kwargs: Dict[str, Any],
        images: List[ChatCompletionMessageContentImageURL],
        system: Optional[str] = None,
    ) -> EncodedQuestion:
        tokenizer = self.tokenizer_manager.tokenizer
        image_data = []
        message_content = content
        if images:
            message = process_content_for_template_format(
                {
                    "role": "user",
                    "content": [
                        *(
                            {"type": "image_url", "image_url": image.model_dump()}
                            for image in images
                        ),
                        {"type": "text", "text": content},
                    ],
                },
                self.template_manager.jinja_template_content_format,
                image_data,
                [],
                [],
                [],
            )
            message_content = message["content"]
        prompt = self._apply_chat_template(
            message_content, chat_template_kwargs, system
        )
        if images:
            # Image marker metadata is an optional native processor capability.
            tokens = getattr(self.tokenizer_manager.mm_processor, "mm_tokens", None)
            if tokens is None or tokens.image_token_regex is None:
                raise ValueError("images require a processor exposing image markers")
            if _has_media_marker(content, tokens, tokenizer):
                raise ValueError(
                    "the input and questions must not contain multimodal marker "
                    "text when images are attached"
                )
            if sum(1 for _ in tokens.image_token_regex.finditer(prompt)) != len(images):
                raise ValueError(
                    "the chat template must preserve one complete marker per image"
                )
        if self.reasoning_markers is not None:
            # Look only after the message, whose last line is fixed text.
            closing = content.rsplit("\n", 1)[-1]
            cut = prompt.rfind(closing)
            generation_prompt = prompt if cut < 0 else prompt[cut + len(closing) :]
            start, end = self.reasoning_markers
            opened = generation_prompt.rfind(start)
            closed = generation_prompt.rfind(end)
            if opened > closed:
                raise ValueError(
                    "the chat template leaves a reasoning block open at the "
                    "answer position, so this model is not supported with these "
                    "chat_template_kwargs"
                )
            if self.answers_open_reasoning and closed < 0:
                raise ValueError(
                    "the reasoning parser for this model expects answers to start "
                    "with a reasoning block, and the chat template does not close "
                    "one. Send chat_template_kwargs that turn thinking off, if the "
                    "template supports it"
                )
            # The template's own finished reply shows whether answers start with
            # a reasoning block that the generation prompt leaves out.
            reply = self._render_reply(closing, chat_template_kwargs)
            begin = reply.rfind(closing) if reply is not None else -1
            answer = reply.find(_REPLY_SENTINEL, begin) if begin >= 0 else -1
            if answer >= 0 and reply[begin:answer].count(start) > (
                generation_prompt.count(start)
            ):
                raise ValueError(
                    "the chat template starts every answer with a reasoning "
                    "block, so this model is not supported"
                )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        # Refuse here because --allow-auto-truncate would cut off the answer position.
        context_len = self.tokenizer_manager.context_len
        if len(prompt_ids) + self.tokenizer_manager.num_reserved_tokens >= context_len:
            raise ValueError(
                f"the prompt has {len(prompt_ids)} tokens, which does not fit "
                f"the context length of {context_len} tokens"
            )
        label_ids = _encode_labels(
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_ids=prompt_ids,
            labels=labels,
            added_tokens=self.added_tokens,
        )
        return prompt, prompt_ids, label_ids, image_data

    def _apply_chat_template(
        self,
        content: str | List[Dict[str, Any]],
        chat_template_kwargs: Dict[str, Any],
        system: Optional[str] = None,
    ) -> str:
        messages = [] if system is None else [{"role": "system", "content": system}]
        try:
            return self.tokenizer_manager.tokenizer.apply_chat_template(
                [*messages, {"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        except _CHAT_TEMPLATE_CLIENT_ERRORS as e:
            raise ValueError(f"the chat template failed: {e}") from e

    def _render_reply(
        self, message: str, chat_template_kwargs: Dict[str, Any]
    ) -> Optional[str]:
        try:
            return self.tokenizer_manager.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": _REPLY_SENTINEL},
                ],
                tokenize=False,
                **chat_template_kwargs,
            )
        except _CHAT_TEMPLATE_CLIENT_ERRORS:
            # The generation prompt checks above still apply.
            return None

    async def _score(
        self,
        adapted_request: Iterator[EncodedQuestion],
        raw_request: Request,
        temperature: float = 1.0,
    ):
        """Encode every question, then score them all in one call."""
        texts, prompts, label_token_ids, image_data = await _encode_all(adapted_request)
        if any(image_data):
            raise ValueError(f"{self.route} does not support images on this server")
        # v0.5.23 asks the scoring API for `return_token_logprobs`; this tree's
        # scoring API returns per-label probabilities (apply_softmax=False), so
        # take their log here. That keeps upstream's semantics: temperature
        # scales the option probabilities only, label_mass stays natural.
        result = await self.tokenizer_manager.score_prompts(
            prompts=texts,
            label_token_ids=label_token_ids,
            apply_softmax=False,
            request=raw_request,
        )
        probabilities_raw = result.scores
        token_logprobs = [
            [math.log(p) if p > 0.0 else float("-inf") for p in row]
            for row in probabilities_raw
        ]
        probabilities = [_softmax_with_temperature(row, temperature)
                         for row in token_logprobs]
        return prompts, label_token_ids, probabilities, token_logprobs, result.prompt_tokens

    async def _handle_non_streaming_request(
        self,
        adapted_request: Iterator[EncodedQuestion],
        request: DecisionRequest,
        raw_request: Request,
    ) -> ORJSONResponse:
        (
            prompts,
            label_token_ids,
            probabilities,
            token_logprobs,
            prompt_tokens,
        ) = await self._score(
            adapted_request=adapted_request,
            raw_request=raw_request,
            temperature=request.temperature,
        )
        answers = {}
        for i, question in enumerate(request.questions):
            answer = _build_answer(
                question=question,
                view=_decision_view(question),
                probabilities=probabilities[i],
                token_logprobs=token_logprobs[i],
            )
            if request.return_prompt_token_ids:
                answer.prompt_token_ids = prompts[i]
                answer.label_token_ids = label_token_ids[i]
            answers[question.id] = answer
        response = DecisionResponse(
            model=request.model,
            prompt_format_version=PROMPT_FORMAT_VERSION,
            answers=answers,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
            ),
        )
        return ORJSONResponse(content=response.model_dump(exclude_none=True))


def render_text(value: Optional[DecisionText]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


async def _encode_all(
    encoded: Iterator[EncodedQuestion],
) -> Tuple[List[str], List[List[int]], List[List[int]], List[List[ImageData]]]:
    """Collect prompt text and label ids, letting other requests run between questions."""
    texts, prompts, label_token_ids, image_data = [], [], [], []
    for text, prompt_ids, label_ids, images in encoded:
        texts.append(text)
        prompts.append(prompt_ids)
        label_token_ids.append(label_ids)
        image_data.append(images)
        # Each question renders and tokenizes the whole input on the event loop.
        await asyncio.sleep(0)
    return texts, prompts, label_token_ids, image_data


def _has_media_marker(
    text: str, tokens: MultimodalSpecialTokens, tokenizer: Any
) -> bool:
    """Whether text holds a processor media marker, which would misalign the images."""
    if tokens.get_combined_regex().search(text):
        return True
    ids = [tokens.image_token_id, tokens.video_token_id, tokens.audio_token_id]
    return any(
        marker and marker in text
        for marker in (tokenizer.decode([i]) for i in ids if i is not None)
    )


def _decision_view(question: DecisionQuestion) -> QuestionView:
    if isinstance(question, DecisionChoiceQuestion):
        return QuestionView(
            kind="choice",
            question=question.question,
            names=[option.name for option in question.options],
            details=[option.description for option in question.options],
        )
    if isinstance(question, DecisionScoreQuestion):
        return QuestionView(
            kind="score",
            question=question.question,
            names=[str(level) for level in range(len(question.levels))],
            details=list(question.levels),
        )
    return QuestionView(
        kind="yes_no",
        question=question.question,
        names=["yes", "no"],
        details=[question.yes, question.no],
    )


def default_labels(view: QuestionView) -> List[str]:
    """Single-token labels in candidate order: A to Z, level indices, or yes and no."""
    if view.kind == "choice":
        return list(string.ascii_uppercase[: len(view.names)])
    return list(view.names)


def render_question(text: str, view: QuestionView, labels: List[str]) -> str:
    """Prompt wording of PROMPT_FORMAT_VERSION."""
    # Every /v1/decisions question has text. A question without its own text
    # drops the question line, and a yes or no question keeps its lead in.
    question_text = (
        "" if is_blank_decision_text(view.question) else render_text(view.question)
    )
    if view.kind == "choice":
        lines = [f"Question: {question_text}"] if question_text else []
        for label, name, description in zip(labels, view.names, view.details):
            detail = render_text(description)
            lines.append(
                f"{label}: {name} - {detail}" if detail else f"{label}: {name}"
            )
        lines.append("Answer with the letter of one option only.")
    elif view.kind == "score":
        lines = [f"Question: {question_text}"] if question_text else []
        lines += [
            f"{label}: {render_text(level)}"
            for label, level in zip(labels, view.details)
        ]
        lines.append("Answer with the number of one level only.")
    else:
        lines = [
            f"Is the following true? {question_text}"
            if question_text
            else "Is the following true?"
        ]
        for label, description in zip(labels, view.details):
            detail = render_text(description)
            if detail:
                lines.append(f"{label}: {detail}")
        lines.append("Answer with yes or no only.")
    return "\n".join([text, "", *lines])


def label_context(
    tokenizer: Any,
    prompt: str,
    prompt_ids: List[int],
    added_tokens: Dict[int, str],
) -> Tuple[str, List[int], bool]:
    """Text and ids after which labels are checked, and whether the shortcut applies."""
    # Added tokens are split off before tokenization.
    # The text after the last one tokenizes on its own,
    # so the check does not grow with the input.
    # When the prompt ends with an added token, each label starts a new segment,
    # which is how the model continues after that token.
    last = next(
        (i for i in reversed(range(len(prompt_ids))) if prompt_ids[i] in added_tokens),
        None,
    )
    if last is not None:
        token = added_tokens[prompt_ids[last]]
        start = prompt.rfind(token)
        suffix = prompt[start + len(token) :]
        if start >= 0 and (
            tokenizer.encode(suffix, add_special_tokens=False) == prompt_ids[last + 1 :]
        ):
            return suffix, prompt_ids[last + 1 :], True
    return prompt, prompt_ids, False


def label_token_id(
    tokenizer: Any, text: str, text_ids: List[int], label: str
) -> Optional[int]:
    """The token a label adds after the text, or None when it is not exactly one."""
    ids = tokenizer.encode(text + label, add_special_tokens=False)
    if len(ids) != len(text_ids) + 1 or ids[:-1] != text_ids:
        return None
    return ids[-1]


def _encode_labels(
    tokenizer: Any,
    prompt: str,
    prompt_ids: List[int],
    labels: List[str],
    added_tokens: Dict[int, str],
) -> List[int]:
    """Check that each label adds exactly one distinct token after the prompt."""
    text, text_ids, _ = label_context(tokenizer, prompt, prompt_ids, added_tokens)
    label_ids = []
    for label in labels:
        token_id = label_token_id(tokenizer, text, text_ids, label)
        if token_id is None or token_id in label_ids:
            raise ValueError(
                f"the answer label {label!r} is not one distinct token after the "
                "chat prompt for this tokenizer, so this model is not supported"
            )
        label_ids.append(token_id)
    return label_ids


def label_mass(token_logprobs: List[float]) -> float:
    return math.fsum(math.exp(logprob) for logprob in token_logprobs)


def _softmax_with_temperature(values: List[float], temperature: float) -> List[float]:
    """Softmax over label logprobs with an optional temperature."""
    scaled = [(-1e30 if math.isinf(v) and v < 0 else v) / temperature for v in values]
    peak = max(scaled)
    exps = [math.exp(v - peak) for v in scaled]
    total = math.fsum(exps)
    if total <= 0:
        return [0.0 for _ in values]
    return [e / total for e in exps]


def _build_answer(
    question: DecisionQuestion,
    view: QuestionView,
    probabilities: List[float],
    token_logprobs: List[float],
) -> DecisionAnswer:
    names = view.names
    value = {}
    if view.kind == "choice":
        value["choice"] = names[probabilities.index(max(probabilities))]
    elif view.kind == "score":
        value["score"] = math.fsum(i * p for i, p in enumerate(probabilities))
    return DecisionAnswer(
        type=question.type,
        probabilities=dict(zip(names, probabilities)),
        label_mass=label_mass(token_logprobs),
        **value,
    )
