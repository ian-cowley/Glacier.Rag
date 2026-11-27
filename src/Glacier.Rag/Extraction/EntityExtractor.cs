namespace Glacier.Rag.Extraction;

using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

public readonly record struct EntityTriplet(string Subject, string Predicate, string Object);

/// <summary>
/// High-speed entity and relational triplet extractor for enterprise domain text.
/// Extracts key concepts, types, APIs, acronyms, quoted terms, and multi-hop relationships for insertion into Glacier.Graph.
/// </summary>
public static partial class EntityExtractor
{
    [GeneratedRegex(@"\b[I]?[A-Z][a-zA-Z0-9]{2,}\b")]
    private static partial Regex PascalCaseRegex();

    [GeneratedRegex(@"\b(?=[A-Z0-9_]*[A-Z])[A-Z0-9_]{2,8}\b")]
    private static partial Regex AcronymRegex();

    [GeneratedRegex(@"""([^""\r\n]{2,50})""|'([^'\r\n]{2,50})'")]
    private static partial Regex QuotedRegex();

    private static readonly (string Phrase, string Relation)[] RelationTriggers =
    [
        ("implements", "IMPLEMENTS"),
        ("inherits from", "INHERITS_FROM"),
        ("depends on", "DEPENDS_ON"),
        ("streams from", "STREAMS_FROM"),
        ("queries", "QUERIES"),
        ("validates", "VALIDATES"),
        ("records", "RECORDS"),
        ("routes to", "ROUTES_TO"),
        ("contains", "CONTAINS"),
        ("uses", "USES"),
        ("produces", "PRODUCES"),
        ("stores", "STORES"),
        ("manages", "MANAGES"),
        ("connects to", "CONNECTS_TO"),
        ("communicates with", "COMMUNICATES_WITH"),
        ("invokes", "INVOKES"),
        ("calls", "CALLS"),
        ("publishes to", "PUBLISHES_TO"),
        ("subscribes to", "SUBSCRIBES_TO"),
        ("writes to", "WRITES_TO"),
        ("reads from", "READS_FROM"),
        ("processes", "PROCESSES"),
        ("transforms", "TRANSFORMS"),
        ("authenticates with", "AUTHENTICATES_WITH"),
        ("authorizes", "AUTHORIZES"),
        ("loads", "LOADS"),
        ("persists to", "PERSISTS_TO"),
        ("serializes", "SERIALIZES"),
        ("deserializes", "DESERIALIZES"),
        ("dispatches to", "DISPATCHES_TO"),
        ("indexes", "INDEXES"),
        ("integrates with", "INTEGRATES_WITH"),
        ("exposes", "EXPOSES"),
        ("consumes", "CONSUMES"),
        ("notifies", "NOTIFIES")
    ];

    /// <summary>
    /// Extracts entities (nodes) and relational triplets (edges) from document text.
    /// </summary>
    public static (List<string> Entities, List<EntityTriplet> Triplets) Extract(string text)
    {
        var entities = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var triplets = new List<EntityTriplet>();

        if (string.IsNullOrWhiteSpace(text))
            return ([], triplets);

        var rawOccurrences = new List<(int Index, string Value)>(16);

        // 1. Extract quoted terms
        foreach (Match match in QuotedRegex().Matches(text))
        {
            string val = (match.Groups[1].Success ? match.Groups[1].Value : match.Groups[2].Value).Trim();
            if (val.Length >= 2 && !IsCommonKeyword(val))
            {
                rawOccurrences.Add((match.Index, val));
            }
        }

        // 2. Extract PascalCase entities
        foreach (Match match in PascalCaseRegex().Matches(text))
        {
            string val = match.Value;
            if (val.Length > 2 && !IsCommonKeyword(val))
            {
                rawOccurrences.Add((match.Index, val));
            }
        }

        // 3. Extract uppercase technical acronyms
        foreach (Match match in AcronymRegex().Matches(text))
        {
            string val = match.Value;
            if (val.Length >= 2 && !IsCommonKeyword(val))
            {
                rawOccurrences.Add((match.Index, val));
            }
        }

        rawOccurrences.Sort(static (a, b) => a.Index.CompareTo(b.Index));

        var foundEntitiesInOrder = new List<string>(rawOccurrences.Count);
        for (int i = 0; i < rawOccurrences.Count; i++)
        {
            if (entities.Add(rawOccurrences[i].Value))
            {
                foundEntitiesInOrder.Add(rawOccurrences[i].Value);
            }
        }

        // 4. Extract relation triplets based on sentence proximity and word-bounded triggers
        var sentences = text.Split(['.', ';', '\n'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        foreach (var sentence in sentences)
        {
            foreach (var (phrase, rel) in RelationTriggers)
            {
                int searchFrom = 0;
                while (searchFrom < sentence.Length)
                {
                    int phraseIdx = sentence.IndexOf(phrase, searchFrom, StringComparison.OrdinalIgnoreCase);
                    if (phraseIdx < 0) break;

                    if (phraseIdx > 0 && IsWordBounded(sentence, phraseIdx, phrase.Length))
                    {
                        string left = sentence.Substring(0, phraseIdx);
                        string right = sentence.Substring(phraseIdx + phrase.Length);

                        string? subj = FindClosestEntity(left, fromEnd: true);
                        string? obj = FindClosestEntity(right, fromEnd: false);

                        if (subj != null && obj != null && !subj.Equals(obj, StringComparison.OrdinalIgnoreCase))
                        {
                            triplets.Add(new EntityTriplet(subj, rel, obj));
                        }
                    }

                    searchFrom = phraseIdx + phrase.Length;
                }
            }
        }

        // 5. Fallback: If entities co-occur in the same chunk, link adjacent document occurrences with RELATED_TO
        for (int i = 0; i < Math.Min(foundEntitiesInOrder.Count - 1, 6); i++)
        {
            string a = foundEntitiesInOrder[i];
            string b = foundEntitiesInOrder[i + 1];
            if (!a.Equals(b, StringComparison.OrdinalIgnoreCase))
            {
                triplets.Add(new EntityTriplet(a, "RELATED_TO", b));
            }
        }

        return (foundEntitiesInOrder, triplets);
    }

    private static bool IsWordBounded(string text, int index, int length)
    {
        bool leftOk = index == 0 || !char.IsLetterOrDigit(text[index - 1]);
        int end = index + length;
        bool rightOk = end >= text.Length || !char.IsLetterOrDigit(text[end]);
        return leftOk && rightOk;
    }

    private static string? FindClosestEntity(string slice, bool fromEnd)
    {
        string? bestVal = null;
        int bestPos = fromEnd ? -1 : int.MaxValue;

        foreach (Match m in QuotedRegex().Matches(slice))
        {
            string val = (m.Groups[1].Success ? m.Groups[1].Value : m.Groups[2].Value).Trim();
            if (val.Length < 2 || IsCommonKeyword(val)) continue;
            int pos = fromEnd ? m.Index + m.Length : m.Index;
            if ((fromEnd && pos > bestPos) || (!fromEnd && pos < bestPos))
            {
                bestPos = pos;
                bestVal = val;
            }
        }

        foreach (Match m in PascalCaseRegex().Matches(slice))
        {
            string val = m.Value;
            if (val.Length <= 2 || IsCommonKeyword(val)) continue;
            int pos = fromEnd ? m.Index + m.Length : m.Index;
            if ((fromEnd && pos > bestPos) || (!fromEnd && pos < bestPos))
            {
                bestPos = pos;
                bestVal = val;
            }
        }

        foreach (Match m in AcronymRegex().Matches(slice))
        {
            string val = m.Value;
            if (val.Length < 2 || IsCommonKeyword(val)) continue;
            int pos = fromEnd ? m.Index + m.Length : m.Index;
            if ((fromEnd && pos > bestPos) || (!fromEnd && pos < bestPos))
            {
                bestPos = pos;
                bestVal = val;
            }
        }

        return bestVal;
    }

    private static bool IsCommonKeyword(string word) => word switch
    {
        "The" or "This" or "That" or "There" or "These" or "Those" or "When" or "Where"
        or "What" or "With" or "From" or "Then" or "Step" or "Example" or "Method" or "Below"
        or "AND" or "OR" or "NOT" or "FOR" or "THE" or "ALL" or "ANY" or "TRUE" or "FALSE"
        or "NULL" or "GET" or "SET" or "USE" or "NEW" or "HOW" or "WHY" or "CAN" or "OUT" => true,
        _ => false
    };
}
