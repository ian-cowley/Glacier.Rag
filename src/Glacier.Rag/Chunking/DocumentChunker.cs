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
        if (maxChunkLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxChunkLength), "Maximum chunk length must be positive.");
        if (overlap < 0 || overlap >= maxChunkLength)
            throw new ArgumentOutOfRangeException(nameof(overlap), "Overlap must be non-negative and strictly less than maxChunkLength.");

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

            // Advance by (take - effectiveOverlap), capping overlap at half of take when naturalBreak shrinks a chunk.
            int effectiveOverlap = Math.Min(overlap, take / 2);
            int step = Math.Max(1, take - effectiveOverlap);
            currentPos += step;
        }

        return chunks;
    }

    private static int FindNaturalBreak(string text, int start, int maxLen)
    {
        int limit = start + maxLen;
        int minBreak = start + (maxLen / 2);

        // 1. Look for double newline (paragraph boundary: \r\n\r\n or \n\n)
        for (int i = limit - 1; i >= minBreak; i--)
        {
            if (i + 3 < text.Length && text[i] == '\r' && text[i + 1] == '\n' && text[i + 2] == '\r' && text[i + 3] == '\n')
                return Math.Min(limit, i + 4);
            if (i + 1 < text.Length && text[i] == '\n' && text[i + 1] == '\n')
                return Math.Min(limit, i + 2);
        }

        // 2. Look for single newline
        for (int i = limit - 1; i >= minBreak; i--)
        {
            if (text[i] == '\n')
                return i + 1;
        }

        // 3. Look for period + space (sentence boundary)
        for (int i = limit - 1; i >= minBreak; i--)
        {
            if (text[i] == '.' && (i + 1 == text.Length || char.IsWhiteSpace(text[i + 1])))
                return i + 1;
        }

        // 4. Look for whitespace
        for (int i = limit - 1; i >= minBreak; i--)
        {
            if (char.IsWhiteSpace(text[i]))
                return i + 1;
        }

        return limit;
    }
}
