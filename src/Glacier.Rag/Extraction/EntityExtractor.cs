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

    private readonly record struct EntityOccurrence(int Start, int End, string Value, int Priority);

    /// <summary>
    /// Extracts entities (nodes) and relational triplets (edges) from document text.
    /// </summary>
    public static (List<string> Entities, List<EntityTriplet> Triplets) Extract(string text)
    {
        var entities = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var triplets = new List<EntityTriplet>();

        if (string.IsNullOrWhiteSpace(text))
            return ([], triplets);

        var rawOccurrences = new List<EntityOccurrence>(16);

        // 1. Extract quoted terms (single regex pass)
        foreach (Match match in QuotedRegex().Matches(text))
        {
            string val = (match.Groups[1].Success ? match.Groups[1].Value : match.Groups[2].Value).Trim();
            if (val.Length >= 2 && !IsCommonKeyword(val))
            {
                rawOccurrences.Add(new EntityOccurrence(match.Index, match.Index + match.Length, val, Priority: 0));
            }
        }

        // 2. Extract PascalCase entities (single regex pass)
        foreach (Match match in PascalCaseRegex().Matches(text))
        {
            string val = match.Value;
            if (val.Length > 2 && !IsCommonKeyword(val))
            {
                rawOccurrences.Add(new EntityOccurrence(match.Index, match.Index + match.Length, val, Priority: 1));
            }
        }

        // 3. Extract uppercase technical acronyms (single regex pass)
        foreach (Match match in AcronymRegex().Matches(text))
        {
            string val = match.Value;
            if (val.Length >= 2 && !IsCommonKeyword(val))
            {
                rawOccurrences.Add(new EntityOccurrence(match.Index, match.Index + match.Length, val, Priority: 2));
            }
        }

        rawOccurrences.Sort(static (a, b) =>
        {
            int cmp = a.Start.CompareTo(b.Start);
            return cmp != 0 ? cmp : a.Priority.CompareTo(b.Priority);
        });

        var foundEntitiesInOrder = new List<string>(rawOccurrences.Count);
        for (int i = 0; i < rawOccurrences.Count; i++)
        {
            if (entities.Add(rawOccurrences[i].Value))
            {
                foundEntitiesInOrder.Add(rawOccurrences[i].Value);
            }
        }

        // 4. Extract relation triplets using cached occurrences & zero-regex sentence-span scans
        int segStart = 0;
        for (int pos = 0; pos <= text.Length; pos++)
        {
            if (pos == text.Length || text[pos] == '.' || text[pos] == ';' || text[pos] == '\n')
            {
                int sentStart = segStart;
                int sentEnd = pos;
                while (sentStart < sentEnd && char.IsWhiteSpace(text[sentStart]))
                    sentStart++;
                while (sentEnd > sentStart && char.IsWhiteSpace(text[sentEnd - 1]))
                    sentEnd--;

                if (sentStart < sentEnd)
                {
                    ScanSentenceTriplets(text, sentStart, sentEnd, rawOccurrences, triplets);
                }

                segStart = pos + 1;
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

    private static void ScanSentenceTriplets(
        string text,
        int sentStart,
        int sentEnd,
        List<EntityOccurrence> occurrences,
        List<EntityTriplet> triplets)
    {
        if (occurrences.Count == 0)
            return;

        int firstInSentence = LowerBound(occurrences, sentStart);
        if (firstInSentence >= occurrences.Count || occurrences[firstInSentence].Start >= sentEnd)
            return;

        ReadOnlySpan<char> textSpan = text.AsSpan();
        foreach (var (phrase, rel) in RelationTriggers)
        {
            int searchFrom = sentStart;
            while (searchFrom < sentEnd)
            {
                int relIdx = textSpan.Slice(searchFrom, sentEnd - searchFrom).IndexOf(phrase, StringComparison.OrdinalIgnoreCase);
                if (relIdx < 0)
                    break;

                int phraseIdx = searchFrom + relIdx;
                if (phraseIdx > sentStart && IsWordBoundedInRange(text, sentStart, sentEnd, phraseIdx, phrase.Length))
                {
                    string? subj = FindClosestLeft(occurrences, firstInSentence, phraseIdx);
                    string? obj = FindClosestRight(occurrences, phraseIdx + phrase.Length, sentEnd);

                    if (subj != null && obj != null && !subj.Equals(obj, StringComparison.OrdinalIgnoreCase))
                    {
                        triplets.Add(new EntityTriplet(subj, rel, obj));
                    }
                }

                searchFrom = phraseIdx + phrase.Length;
            }
        }
    }

    private static int LowerBound(List<EntityOccurrence> occurrences, int minStart)
    {
        int lo = 0;
        int hi = occurrences.Count;
        while (lo < hi)
        {
            int mid = lo + ((hi - lo) >> 1);
            if (occurrences[mid].Start < minStart)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }

    private static string? FindClosestLeft(List<EntityOccurrence> occurrences, int firstInSentence, int phraseIdx)
    {
        string? bestVal = null;
        int bestEnd = -1;
        int bestPriority = int.MaxValue;

        for (int i = firstInSentence; i < occurrences.Count; i++)
        {
            var occ = occurrences[i];
            if (occ.Start >= phraseIdx)
                break;

            if (occ.End <= phraseIdx)
            {
                if (occ.End > bestEnd || (occ.End == bestEnd && occ.Priority < bestPriority))
                {
                    bestEnd = occ.End;
                    bestPriority = occ.Priority;
                    bestVal = occ.Value;
                }
            }
        }

        return bestVal;
    }

    private static string? FindClosestRight(List<EntityOccurrence> occurrences, int phraseEnd, int sentEnd)
    {
        int idx = LowerBound(occurrences, phraseEnd);
        for (int i = idx; i < occurrences.Count; i++)
        {
            var occ = occurrences[i];
            if (occ.Start >= sentEnd)
                break;

            if (occ.End <= sentEnd)
                return occ.Value;
        }

        return null;
    }

    private static bool IsWordBoundedInRange(string text, int sentStart, int sentEnd, int index, int length)
    {
        bool leftOk = index == sentStart || !char.IsLetterOrDigit(text[index - 1]);
        int end = index + length;
        bool rightOk = end >= sentEnd || !char.IsLetterOrDigit(text[end]);
        return leftOk && rightOk;
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
