namespace Glacier.Rag.Extraction;

using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

public readonly record struct EntityTriplet(string Subject, string Predicate, string Object);

/// <summary>
/// High-speed entity and relational triplet extractor for enterprise domain text.
/// Extracts key concepts, types, APIs, and relationships for insertion into Glacier.Graph.
/// </summary>
public static class EntityExtractor
{
    // Regex for matching PascalCase types / interfaces (e.g. IAsyncEnumerable, LedgerAccountDto, SqlServer)
    private static readonly Regex PascalCaseRegex = new(@"\b[I]?[A-Z][a-zA-Z0-9]{2,}\b", RegexOptions.Compiled);

    // Common relational trigger phrases
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
        ("manages", "MANAGES")
    ];

    /// <summary>
    /// Extracts entities (nodes) and relational triplets (edges) from document text.
    /// </summary>
    public static (List<string> Entities, List<EntityTriplet> Triplets) Extract(string text)
    {
        var entities = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var triplets = new List<EntityTriplet>();

        if (string.IsNullOrWhiteSpace(text))
            return (new List<string>(), triplets);

        // 1. Extract PascalCase entities
        var matches = PascalCaseRegex.Matches(text);
        var foundEntitiesInOrder = new List<string>();

        foreach (Match match in matches)
        {
            string val = match.Value;
            if (val.Length > 2 && !IsCommonKeyword(val))
            {
                entities.Add(val);
                foundEntitiesInOrder.Add(val);
            }
        }

        // 2. Extract relation triplets based on sentence proximity and triggers
        var sentences = text.Split(['.', ';', '\n'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        foreach (var sentence in sentences)
        {
            foreach (var (phrase, rel) in RelationTriggers)
            {
                int phraseIdx = sentence.IndexOf(phrase, StringComparison.OrdinalIgnoreCase);
                if (phraseIdx > 0)
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
            }
        }

        // 3. Fallback: If entities co-occur in the same chunk, link them with CO_OCCURS
        for (int i = 0; i < Math.Min(foundEntitiesInOrder.Count - 1, 4); i++)
        {
            string a = foundEntitiesInOrder[i];
            string b = foundEntitiesInOrder[i + 1];
            if (!a.Equals(b, StringComparison.OrdinalIgnoreCase))
            {
                triplets.Add(new EntityTriplet(a, "RELATED_TO", b));
            }
        }

        return (new List<string>(entities), triplets);
    }

    private static string? FindClosestEntity(string slice, bool fromEnd)
    {
        var matches = PascalCaseRegex.Matches(slice);
        if (matches.Count == 0) return null;

        Match targetMatch = fromEnd ? matches[matches.Count - 1] : matches[0];
        string val = targetMatch.Value;
        return IsCommonKeyword(val) ? null : val;
    }

    private static bool IsCommonKeyword(string word) => word switch
    {
        "The" or "This" or "That" or "There" or "These" or "Those" or "When" or "Where" 
        or "What" or "With" or "From" or "Then" or "Step" or "Example" or "Method" or "Below" => true,
        _ => false
    };
}
