namespace Glacier.Rag.Engine;

using System;
using System.Buffers;
using Glacier.Inference.Tokenizer;

/// <summary>
/// Idempotent prompt-injection sanitizer that delegates to <see cref="BpeTokenizer.SanitizePrompt"/>
/// and enforces defense-in-depth escaping for all M3.4 control tokens (including tokens missing
/// from published NuGet <c>Glacier.Inference 1.10.1</c> when built standalone).
/// </summary>
internal static class PromptSanitizer
{
    private static readonly SearchValues<char> TriggerChars = SearchValues.Create("<[\uFF5C");

    private static readonly (string Token, string Replacement)[] ControlTokens =
    [
        // DeepSeek family
        ("<｜begin of sentence｜>", "< ｜begin of sentence｜ >"),
        ("<｜begin▁of▁sentence｜>", "< ｜begin▁of▁sentence｜ >"),
        ("<｜end of sentence｜>", "< ｜end of sentence｜ >"),
        ("<｜end▁of▁sentence｜>", "< ｜end▁of▁sentence｜ >"),
        ("<｜System｜>", "< ｜System｜ >"),
        ("<｜User｜>", "< ｜User｜ >"),
        ("<｜Assistant｜>", "< ｜Assistant｜ >"),
        ("<｜latest_reminder｜>", "< ｜latest_reminder｜ >"),
        ("｜DSML｜", "｜ DSML ｜"),
        ("<｜ref｜>", "< ｜ref｜ >"),
        ("<｜/ref｜>", "< ｜/ref｜ >"),
        ("<｜det｜>", "< ｜det｜ >"),
        ("<｜/det｜>", "< ｜/det｜ >"),
        ("<｜box｜>", "< ｜box｜ >"),
        ("<｜/box｜>", "< ｜/box｜ >"),
        ("<｜point｜>", "< ｜point｜ >"),
        ("<｜/point｜>", "< ｜/point｜ >"),
        ("<｜polygon｜>", "< ｜polygon｜ >"),
        ("<｜/polygon｜>", "< ｜/polygon｜ >"),
        ("<｜grounding｜>", "< ｜grounding｜ >"),
        ("<｜/grounding｜>", "< ｜/grounding｜ >"),

        // Thinking / reasoning tags
        ("<think>", "< think >"),
        ("</think>", "< /think >"),
        ("<|think|>", "< |think| >"),
        ("<|thought|>", "< |thought| >"),
        ("<|/thought|>", "< |/thought| >"),

        // ChatML / Qwen / OpenAI / Phi
        ("<|im_start|>", "< |im_start| >"),
        ("<|im_end|>", "< |im_end| >"),
        ("<|im_sep|>", "< |im_sep| >"),
        ("<|endoftext|>", "< |endoftext| >"),

        // LLaMA 3
        ("<|start_header_id|>", "< |start_header_id| >"),
        ("<|end_header_id|>", "< |end_header_id| >"),
        ("<|eot_id|>", "< |eot_id| >"),
        ("<|begin_of_text|>", "< |begin_of_text| >"),
        ("<|end_of_text|>", "< |end_of_text| >"),

        // Gemma 2 / 3 / 4
        ("<start_of_turn>", "< start_of_turn >"),
        ("<end_of_turn>", "< end_of_turn >"),
        ("<|turn>", "< |turn >"),
        ("<turn|>", "< turn| >"),
        ("<|channel>", "< |channel >"),
        ("<channel|>", "< channel| >"),
        ("<|tool>", "< |tool >"),
        ("<tool|>", "< tool| >"),
        ("<|tool_call>", "< |tool_call >"),
        ("<tool_call|>", "< tool_call| >"),
        ("<|tool_response>", "< |tool_response >"),
        ("<tool_response|>", "< tool_response| >"),
        ("<bos>", "< bos >"),
        ("<eos>", "< eos >"),

        // Inkling-2
        ("<|message_user|>", "< |message_user| >"),
        ("<|message_model|>", "< |message_model| >"),
        ("<|message_system|>", "< |message_system| >"),
        ("<|message_tool|>", "< |message_tool| >"),
        ("<|content_text|>", "< |content_text| >"),
        ("<|content_thinking|>", "< |content_thinking| >"),
        ("<|content_image|>", "< |content_image| >"),
        ("<|content_xml|>", "< |content_xml| >"),
        ("<|content_invoke_tool_json|>", "< |content_invoke_tool_json| >"),
        ("<|content_invoke_tool_text|>", "< |content_invoke_tool_text| >"),
        ("<|content_tool_error|>", "< |content_tool_error| >"),
        ("<|end_message|>", "< |end_message| >"),
        ("<|content_model_end_sampling|>", "< |content_model_end_sampling| >"),
        ("<|unused_200054|>", "< |unused_200054| >"),

        // Mistral / SentencePiece
        ("[INST]", "[ INST ]"),
        ("[/INST]", "[ /INST ]"),
        ("<s>", "< s >"),
        ("</s>", "< /s >"),
        ("<unk>", "< unk >"),
        ("[TOOL_CALLS]", "[ TOOL_CALLS ]"),
        ("[TOOL_RESULTS]", "[ TOOL_RESULTS ]"),
        ("[/TOOL_RESULTS]", "[ /TOOL_RESULTS ]")
    ];

    private static readonly SearchValues<string> ControlTokenSearchValues =
        SearchValues.Create(Array.ConvertAll(ControlTokens, static t => t.Token), StringComparison.Ordinal);

    public static string Sanitize(string input)
    {
        if (string.IsNullOrEmpty(input))
            return input;

        // Zero-alloc fast path when no trigger delimiter appears in the text.
        if (input.AsSpan().IndexOfAny(TriggerChars) < 0)
            return input;

        string sanitized = BpeTokenizer.SanitizePrompt(input);

        if (sanitized.AsSpan().IndexOfAny(ControlTokenSearchValues) < 0)
            return sanitized;

        for (int i = 0; i < ControlTokens.Length; i++)
        {
            var (token, replacement) = ControlTokens[i];
            if (sanitized.Contains(token, StringComparison.Ordinal))
            {
                sanitized = sanitized.Replace(token, replacement, StringComparison.Ordinal);
            }
        }

        return sanitized;
    }
}
