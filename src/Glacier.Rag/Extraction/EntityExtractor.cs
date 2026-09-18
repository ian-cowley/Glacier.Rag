namespace Glacier.Rag.Extraction;

using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

public readonly record struct EntityTriplet(string Subject, string Predicate, string Object);

/// <summary>
/// High-speed entity and relational triplet extractor for enterprise domain text.
/// Extracts key concepts, types, APIs, acronyms, quoted terms, and multi-hop relationships for insertion into Glacier.Graph.
/// </summary>
public static class EntityExtractor
{
    // Regex for matching PascalCase types / interfaces (e.g. IAsyncEnumerable, LedgerAccountDto, SqlServer)
    private static readonly Regex PascalCaseRegex = new(@"\b[I]?[A-Z][a-zA-Z0-9]{2,}\b", RegexOptions.Compiled);

    // Regex for matching technical acronyms (e.g. SQL, HTTP, VRAM, SIMD, AVX, CSR, DTO, API)
    private static readonly Regex AcronymRegex = new(@"\b[A-Z0-9_]{2,8}\b", RegexOptions.Compiled);

    // Regex for matching quoted enterprise terms or phrases (e.g. "Customer Balance", "Account Table")
    private static readonly Regex QuotedRegex = new(@"""([^""\r\n]{2,50})""|'([^'\r\n]{2,50})'", RegexOptions.Compiled);

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
            return (new List<string>(), triplets);

        var foundEntitiesInOrder = new List<string>();

        // 1. Extract quoted terms
        var quotedMatches = QuotedRegex.Matches(text);
        foreach (Match match in quotedMatches)
        {
            string val = match.Groups[1].Success ? match.Groups[1].Value : match.Groups[2].Value;
            val = val.Trim();
            if (val.Length >= 2 && !IsCommonKeyword(val))
            {
                if (entities.Add(val))
                    foundEntitiesInOrder.Add(val);
            }
        }

        // 2. Extract PascalCase entities
        var pascalMatches = PascalCaseRegex.Matches(text);
        foreach (Match match in pascalMatches)
        {
            string val = match.Value;
            if (val.Length > 2 && !IsCommonKeyword(val))
            {
                if (entities.Add(val))
                    foundEntitiesInOrder.Add(val);
            }
        }

        // 3. Extract uppercase technical acronyms
        var acronymMatches = AcronymRegex.Matches(text);
        foreach (Match match in acronymMatches)
        {
            string val = match.Value;
            if (val.Length >= 2 && !IsCommonKeyword(val))
            {
                if (entities.Add(val))
                    foundEntitiesInOrder.Add(val);
            }
        }

        // 4. Extract relation triplets based on sentence proximity and triggers
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

        // 5. Fallback: If entities co-occur in the same chunk, link them with RELATED_TO
        for (int i = 0; i < Math.Min(foundEntitiesInOrder.Count - 1, 6); i++)
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
        // Check quoted first
        var qMatches = QuotedRegex.Matches(slice);
        if (qMatches.Count > 0)
        {
            Match target = fromEnd ? qMatches[qMatches.Count - 1] : qMatches[0];
            string val = (target.Groups[1].Success ? target.Groups[1].Value : target.Groups[2].Value).Trim();
            if (!IsCommonKeyword(val)) return val;
        }

        // Check PascalCase
        var pMatches = PascalCaseRegex.Matches(slice);
        if (pMatches.Count > 0)
        {
            Match target = fromEnd ? pMatches[pMatches.Count - 1] : pMatches[0];
            string val = target.Value;
            if (!IsCommonKeyword(val)) return val;
        }

        // Check Acronym
        var aMatches = AcronymRegex.Matches(slice);
        if (aMatches.Count > 0)
        {
            Match target = fromEnd ? aMatches[aMatches.Count - 1] : aMatches[0];
            string val = target.Value;
            if (!IsCommonKeyword(val)) return val;
        }

        return null;
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
