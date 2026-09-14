"""Canonical TRACE prompts and output contracts from the upstream repository."""

TASK_PROMPTS = {
    "FOMC": "What is the monetary policy stance for the following text? A. dovish, B. hawkish, C. neutral. Choose one from A, B and C.\n",
    "C-STANCE": "判断以下文本对指定对象的态度，选择一项：A.支持，B.反对，C.中立。输出A，B或者C。\n",
    "ScienceQA": "Choose an answer for the following question and give your reasons.\n\n",
    "NumGLUE-cm": "Solve the following math problem.\n",
    "NumGLUE-ds": "Solve the following math problem.\n",
    "MeetingBank": "Write a summary of the following meeting transcripts.\n",
    # TRACE's official naive protocol adds no task prefix for Py150.
    "Py150": "",
    "20Minuten": "Provide a simplified version of the following paragraph in German.\n\n",
}

OUTPUT_CONTRACTS = {
    "C-STANCE": "Output exactly one label: A, B, or C.",
    "FOMC": "Output exactly one label: A, B, or C.",
    "MeetingBank": "Output only the requested meeting summary.",
    "Py150": "Output only the immediate code continuation, with no explanation or Markdown fence.",
    "ScienceQA": "Output the answer choice first, followed by its explanation.",
    "NumGLUE-cm": "Solve the problem and put the final numeric answer last.",
    "NumGLUE-ds": "Solve the problem and put the final numeric answer last.",
    "20Minuten": "Output only the simplified German paragraph.",
}


def ensure_task_prompt(task: str, prompt: str) -> str:
    prefix = TASK_PROMPTS[task]
    return prompt if prompt.startswith(prefix) else prefix + prompt


def privileged_prompt(task: str, prompt: str, answer: str) -> str:
    prompt = ensure_task_prompt(task, prompt)
    return (
        f"{prompt}\n\nPrivileged reference response:\n{answer}\n\n"
        "Now answer the original request yourself. " + OUTPUT_CONTRACTS[task]
    )
