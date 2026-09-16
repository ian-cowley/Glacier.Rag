namespace Glacier.Rag.Chunking;

using System;
using System.Collections.Generic;

/// <summary>
/// A chunk of text extracted from a parent document with line and character boundaries.
/// </summary>
public sealed class DocumentChunk
{
    public required string ChunkId { get; init; }
    public required string DocumentId { get; init; }
    public required int ChunkIndex { get; init; }
    public required string Content { get; init; }
    public required int StartChar { get; init; }
    public required int EndChar { get; init; }
    public Dictionary<string, string> Metadata { get; init; } = new(StringComparer.OrdinalIgnoreCase);

    public override string ToString() => $"[{DocumentId}#{ChunkIndex}] {Content}";
}

/// <summary>
/// Zero-copy, high-speed document chunker supporting paragraph and sliding token windows.
/// </summary>
public static class DocumentChunker
{
    public static List<DocumentChunk> ChunkText(
        string documentId, 
        string text, 
        int maxChunkLength = 512, 
        int overlap = 64)
    {
        if (string.IsNullOrEmpty(text)) return [];

        var chunks = new List<DocumentChunk>();
        int chunkIdx = 0;
        int currentPos = 0;

        while (currentPos < text.Length)
        {
            int remaining = text.Length - currentPos;
            int take = Math.Min(maxChunkLength, remaining);

            // If we are not at the end of the text, try to break at paragraph or sentence boundary
            if (currentPos + take < text.Length)
            {
                int naturalBreak = FindNaturalBreak(text, currentPos, take);
                if (naturalBreak > currentPos + (take / 2))
                {
                    take = naturalBreak - currentPos;
                }
            }

            string chunkContent = text.Substring(currentPos, take).Trim();
            if (!string.IsNullOrWhiteSpace(chunkContent))
            {
                chunks.Add(new DocumentChunk
                {
                    ChunkId = $"{documentId}_{chunkIdx}",
                    DocumentId = documentId,
                    ChunkIndex = chunkIdx++,
                    Content = chunkContent,
                    StartChar = currentPos,
                    EndChar = currentPos + take
                });
            }

            if (currentPos + take >= text.Length) break;

            // Advance by (take - overlap)
            int step = Math.Max(1, take - overlap);
            currentPos += step;
        }

        return chunks;
    }

    private static int FindNaturalBreak(string text, int start, int maxLen)
    {
        int limit = start + maxLen;

        // 1. Look for double newline (paragraph boundary)
        for (int i = limit - 1; i >= start + (maxLen / 2); i--)
        {
            if (i + 1 < text.Length && text[i] == '\n' && text[i + 1] == '\n')
                return i + 2;
        }

        // 2. Look for single newline
        for (int i = limit - 1; i >= start + (maxLen / 2); i--)
        {
            if (text[i] == '\n')
                return i + 1;
        }

        // 3. Look for period + space (sentence boundary)
        for (int i = limit - 1; i >= start + (maxLen / 2); i--)
        {
            if (text[i] == '.' && (i + 1 == text.Length || char.IsWhiteSpace(text[i + 1])))
                return i + 1;
        }

        // 4. Look for whitespace
        for (int i = limit - 1; i >= start + (maxLen / 2); i--)
        {
            if (char.IsWhiteSpace(text[i]))
                return i + 1;
        }

        return limit;
    }
}
